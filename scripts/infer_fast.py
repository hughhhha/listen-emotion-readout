#!/usr/bin/env python3
"""Run frozen Qwen2.5-Omni Thinker on a condition-expanded LISTEN manifest.

The prompt, choice expansion, and generated-letter parser follow LISTEN commit
32966584488bd3b3c67a150bd55f1f18ca09ef76. General max_new_tokens=3 is what
LISTEN requests. On Transformers 4.52.4 the top-level Omni wrapper implicitly
overrides that cap with 1024; use --max-new-tokens 1024 for that behavior.
"""
import argparse
import contextlib
import functools
import gc
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import string
import sys
import time

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
          "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
MODES = {
    "1_text": "text", "1_audio": "audio", "1_audio_and_text": "audio_and_text",
    "2A": "text", "2B": "audio", "2C": "audio_and_text",
    "3A": "text", "3B": "audio", "3C": "audio_and_text", "4": "audio",
}
INTRO = {
    "audio": "Listen to the audio and classify the emotion.",
    "text": "Read the transcription and classify the emotion.",
    "audio_and_text": "Listen to the audio and read the transcription, then classify the emotion.",
}
EXTRA = ["neutral", "sadness", "surprise", "happiness", "fear", "anger", "excitement", "frustration"]
ALIASES = {"angry": "anger", "happy": "happiness", "sad": "sadness", "pleasant_surprise": "surprise"}


def normalize(label):
    label = str(label).strip().lower()
    return ALIASES.get(label, label)


def prompt_for(row, choices):
    mode = MODES[row["condition"]]
    letters = string.ascii_uppercase[:len(choices)]
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    instruction = f"Respond with only the letter ({', '.join(letters)}):"
    prompt = "\n\n".join((INTRO[mode], row["question"], options, instruction))
    if mode != "audio":
        prompt = f"Transcription: {row['transcription']}\n\n{prompt}"
    user = [] if mode == "text" else [{"type": "audio", "audio": "in_memory"}]
    user.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": user}]


def extract_letter(text, size):
    valid = string.ascii_uppercase[:size]
    return next((c for c in text.upper() if c in valid), None)


def ordered_rows(manifest, conditions, seed):
    grouped = {condition: [] for condition in conditions}
    seen = set()
    with manifest.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            condition = row["condition"]
            if condition not in grouped:
                continue
            if row["id"] in seen:
                raise ValueError(f"Duplicate identifier in expanded manifest: {row['id']}")
            seen.add(row["id"])
            grouped[condition].append(row)
    rng = random.Random(seed)
    for condition, rows in grouped.items():
        for row in rows:
            choices = list(row["choices"])
            if condition in ("3A", "3B", "3C"):
                choices = list(set(choices + EXTRA))  # Same union as LISTEN, fixed PYTHONHASHSEED.
            if not 1 <= len(choices) <= 26:
                raise ValueError(f"Invalid choice count on {row['id']}")
            expected = row.get("explicit_emotion") if condition in ("2A", "3A") else row.get("answer")
            if expected is None:
                expected = row.get("label")
            # An empty or absent official target is skipped before any random draw in LISTEN.
            if not expected or normalize(expected) not in {normalize(c) for c in choices}:
                continue
            rng.shuffle(choices)
            supplied = row.get("ordered_choices_by_experiment", {}).get(condition)
            if supplied is None:
                supplied = row.get("ordered_choices")
            if supplied is not None:
                if sorted(choices) != sorted(supplied):
                    raise ValueError(f"Ordered choices disagree with candidates on {row['id']}")
                choices = supplied
            row["ordered_choices"] = choices
            if expected in choices:
                index = choices.index(expected)
            else:
                index = [normalize(c) for c in choices].index(normalize(expected))
            row["expected_letter"] = string.ascii_uppercase[index]
            row["label"] = normalize(choices[index])
            yield row


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_output(f):
    """Discard only a torn final record. Reject duplicates or corrupt committed rows."""
    completed = set()
    f.seek(0)
    while True:
        position = f.tell()
        line = f.readline()
        if not line:
            break
        if not line.endswith(b"\n"):
            f.truncate(position)
            break
        record = json.loads(line)
        key = record["id"]
        if key in completed:
            raise ValueError(f"Duplicate output record {key}")
        completed.add(key)
    return completed


def load_audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1, dtype=np.float32)
    if rate != 16000:
        common = math.gcd(16000, rate)
        wave = resample_poly(wave, 16000 // common, rate // common).astype(np.float32)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Empty or non-finite audio file: {path}")
    return wave


def trimmed_audio_kwargs(extractor, waves):
    # Whitening uses the clip maximum and STFT uses centered windows. Leave
    # silent samples beyond every valid window so no reflected edge enters it.
    longest = max(map(len, waves))
    hop = extractor.hop_length
    guard = extractor.n_fft + hop
    cap = min(extractor.n_samples, ((longest + guard + hop - 1) // hop) * hop)
    return {"audio_kwargs": {"max_length": cap}}


def base_record(row):
    keys = ("id", "condition", "label", "group", "speaker_group", "experiment_type",
            "dataset_source", "split", "expected_letter")
    return {key: row[key] for key in keys if key in row}


def save_vector(path, vector):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part")
    with temp.open("wb") as writer:
        np.save(writer, vector, allow_pickle=False)
    os.replace(temp, path)


class Inference:
    def __init__(self, args):
        import torch
        from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        self.args = args
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[args.dtype]
        config = Qwen2_5OmniConfig.from_pretrained(args.model)
        self.thinker, loading_info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model, config=config.thinker_config, torch_dtype=dtype, attn_implementation=args.attention,
            device_map={"": args.device}, output_loading_info=True,
        )
        if loading_info["missing_keys"] or loading_info["mismatched_keys"]:
            raise RuntimeError(f"Incomplete Thinker weights: {loading_info['missing_keys']}, {loading_info['mismatched_keys']}")
        self.thinker.eval().requires_grad_(False)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
        self.processor.tokenizer.padding_side = "left"
        tokenizer = self.processor.tokenizer
        self.letters = []
        for letter in string.ascii_uppercase:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Answer letter {letter} is not a single token")
            self.letters.append(ids[0])
        self.read = functools.lru_cache(maxsize=48)(load_audio)
        self.embdir = args.output.parent / (args.output.stem + "_embeddings")

    def run(self, rows):
        import torch
        mode = MODES[rows[0]["condition"]]
        has_audio = mode != "text"
        chats = [prompt_for(row, row["ordered_choices"]) for row in rows]
        texts = self.processor.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
        waves = [self.read(self.args.audio_root / row["audio_path"]) for row in rows] if has_audio else None
        padding_kwargs = trimmed_audio_kwargs(self.processor.feature_extractor, waves) if has_audio and self.args.trim_audio_padding else {}
        inputs = self.processor(text=texts, audio=waves, padding=True, return_tensors="pt",
                                use_audio_in_video=has_audio, **padding_kwargs)
        inputs = inputs.to(self.thinker.device)
        if has_audio:
            inputs["input_features"] = inputs["input_features"].to(self.thinker.dtype)
        scores, vectors = [], []
        stopping_args = {}
        stopped_with_prefix = [False] * len(rows)
        if self.args.stop_on_letter:
            from transformers import StoppingCriteria, StoppingCriteriaList
            tokenizer = self.processor.tokenizer
            input_width = inputs.input_ids.shape[1]
            sizes = [len(r["ordered_choices"]) for r in rows]
            class FirstAnswerLetter(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    tails = tokenizer.batch_decode(input_ids[:, input_width:], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                    done = [extract_letter(t, n) is not None for t,n in zip(tails,sizes)]
                    for i, field in enumerate(done):
                        stopped_with_prefix[i] = stopped_with_prefix[i] or field
                    return torch.tensor(done, dtype=torch.bool, device=input_ids.device)
            stopping_args["stopping_criteria"] = StoppingCriteriaList([FirstAnswerLetter()])
        cap = max(self.args.max_new_tokens, self.args.fallback_cap) if self.args.stop_on_letter else self.args.max_new_tokens
        # Select only the valid letter columns from prefill, avoiding full-vocabulary caches.
        def capture_logits(module, args, out):
            if not scores:
                scores.append(out[:, -1, self.letters].detach().float().cpu())

        def capture_audio(module, args, out):
            if vectors:
                return
            features = getattr(out, "last_hidden_state", None)
            if features is None:
                features = out.pooler_output
            lengths = inputs.input_ids.eq(self.thinker.config.audio_token_index).sum(1).tolist()
            if sum(lengths) != features.shape[0] or min(lengths) < 1:
                raise ValueError(f"Audio-token alignment failed: {lengths}, encoder shape {features.shape}")
            vectors.extend(part.float().mean(0).cpu().numpy() for part in features.split(lengths))

        with contextlib.ExitStack() as hooks, torch.inference_mode():
            hooks.enter_context(self.thinker.lm_head.register_forward_hook(capture_logits))
            if self.args.embeddings and has_audio:
                hooks.enter_context(self.thinker.audio_tower.register_forward_hook(capture_audio))
            output = self.thinker.generate(
                **inputs, max_new_tokens=cap, do_sample=False, num_beams=1, **stopping_args,
                pad_token_id=self.processor.tokenizer.eos_token_id, use_audio_in_video=has_audio,
            )
        decoded = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for index, (row, response) in enumerate(zip(rows, decoded)):
            choices = row["ordered_choices"]
            letter = extract_letter(response.strip(), len(choices))
            logits = scores[0][index, :len(choices)].tolist()
            class_scores = {}
            for name, value in zip(choices, logits):
                name = normalize(name)
                class_scores[name] = max(class_scores.get(name, float("-inf")), value)
            prediction = normalize(choices[ord(letter) - 65]) if letter else None
            record = base_record(row)
            record.update(prediction=prediction, scores=class_scores, choices=choices,
                          raw_response=response, raw_response_is_prefix=stopped_with_prefix[index], predicted_letter=letter, letter_logits=logits)
            if vectors:
                pathkey = hashlib.sha256(str(row["audio_path"]).encode()).hexdigest()
                path = self.embdir / (pathkey + ".npy")
                if not path.exists():
                    save_vector(path, vectors[index])
                record["embedding_path"] = str(path.relative_to(self.args.output.parent))
            yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=Path("."))
    parser.add_argument("--model", required=True, help="Local full model checkpoint folder or Hugging Face ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--batch-size", type=int, default=16, help="Maximum audio or joint batch size")
    parser.add_argument("--text-batch-size", type=int, default=48)
    parser.add_argument("--max-audio-seconds", type=float, default=140.0,
                        help="Maximum sum of audio duration per batch; a longer clip runs alone")
    parser.add_argument("--bucket-size", type=int, default=128,
                        help="Length-sort within condition-local windows; use 1 for original order")
    parser.add_argument("--trim-audio-padding", action=argparse.BooleanOptionalAction, default=True,
                        help="Avoid fixed 300s log-mel padding, retaining a full silent STFT guard")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-on-letter", action="store_true",
                        help="Stop each greedy output at its first official-parser letter; unresolved rows keep decoding")
    parser.add_argument("--fallback-cap", type=int, default=1024,
                        help="Maximum generation length for unresolved rows with --stop-on-letter (native wrapper default)")
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.text_batch_size, args.bucket_size, args.max_audio_seconds, args.fallback_cap) <= 0:
        parser.error("batch size and generation cap must be positive")
    if os.getenv("PYTHONHASHSEED") in (None, "", "random"):
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    conditions = [c for c in MODES if c in args.experiments]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    import fcntl
    import torch
    import transformers
    config_file = Path(args.model) / "config.json"
    run_info = dict(script_sha256=hash_file(Path(__file__)), manifest_sha256=hash_file(args.manifest),
                    model=args.model, config_sha256=hash_file(config_file) if config_file.is_file() else None,
                    seed=args.seed, hash_seed=os.environ["PYTHONHASHSEED"], experiments=conditions,
                    max_new_tokens=args.max_new_tokens, attention=args.attention, dtype=args.dtype,
                    embeddings=args.embeddings, trim_audio_padding=args.trim_audio_padding, stop_on_letter=args.stop_on_letter, fallback_cap=args.fallback_cap, torch=torch.__version__, transformers=transformers.__version__)
    with args.output.open("a+b") as output:
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = restore_output(output)
        if meta_path.exists():
            if json.loads(meta_path.read_text()) != run_info:
                raise RuntimeError("Existing run was produced with different inputs or configuration; choose a new output path.")
        elif completed:
            raise RuntimeError("Existing output is missing run metadata")
        else:
            temp = meta_path.with_suffix(".tmp")
            temp.write_text(json.dumps(run_info, indent=2) + "\n")
            os.replace(temp, meta_path)
        initial = len(completed)
        print(f"Resuming with {initial} completed rows.", flush=True)
        source = (r for r in ordered_rows(args.manifest, conditions, args.seed) if r["id"] not in completed)
        timing_path = args.output.with_suffix(args.output.suffix + ".timing.jsonl")
        with timing_path.open("a") as timing:
            run_batches(args, output, timing, source, initial)


def run_batches(args, output, timing, source, initial):
    import torch
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    start = time.monotonic()
    run_id = time.time_ns()
    processed = peak = splits = 0
    engine = None
    def event(kind, **values):
        timing.write(json.dumps(dict(event=kind, run_id=run_id, **values)) + "\n")
        timing.flush()
    event("start", initial=initial, device=str(device), batch_size=args.batch_size,
          text_batch_size=args.text_batch_size, max_audio_seconds=args.max_audio_seconds,
          bucket_size=args.bucket_size, trim_audio_padding=args.trim_audio_padding)
    def commit(records):
        nonlocal processed
        output.write(b"".join((json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n").encode() for r in records))
        output.flush()
        processed += len(records)
    def attempt(batch):
        nonlocal engine, peak, splits
        if not batch:
            return
        if engine is None:
            engine = Inference(args)
        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        before = time.monotonic()
        failed = False
        try:
            predictions = list(engine.run(batch))
            if use_cuda:
                torch.cuda.synchronize(device)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            failed = True
        batch_peak = torch.cuda.max_memory_allocated(device) if use_cuda else 0
        peak = max(peak, batch_peak)
        if failed:
            splits += 1
            event("oom", condition=batch[0]["condition"], size=len(batch), peak_allocated_bytes=batch_peak)
            # The exception frames must expire before releasing the allocator cache.
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
            half = len(batch) // 2
            attempt(batch[:half])
            attempt(batch[half:])
            return
        commit(predictions)
        elapsed = time.monotonic()-before
        event("batch", condition=batch[0]["condition"], size=len(batch), first_id=batch[0]["id"],
              total_audio_seconds=round(sum(r.get("_seconds", 0) for r in batch), 4),
              elapsed_sec=round(elapsed, 4), cumulative_sec=round(time.monotonic()-start, 4),
              peak_allocated_bytes=batch_peak)
        if processed <= len(batch) or processed//128 != (processed-len(batch))//128:
            print(f"{initial+processed} committed, {processed / max(time.monotonic()-start, .001):.2f} new items/s",flush=True)
    def ordered(group, has_audio):
        while True:
            block = list(itertools.islice(group, args.bucket_size))
            if not block:
                return
            for row in block:
                if has_audio:
                    duration = row.get("duration")
                    if duration is None:
                        import soundfile as sf
                        info = sf.info(args.audio_root / row["audio_path"])
                        duration = info.duration
                    row["_seconds"] = float(duration)
            key = (lambda r:r["_seconds"]) if has_audio else (lambda r:len(r.get("transcription") or "") + len(r["question"]))
            yield from sorted(block, key=key)
    for cond, rows in itertools.groupby(source,key=lambda r:r["condition"]):
        has_audio = MODES[cond] != "text"
        limit = args.batch_size if has_audio else args.text_batch_size
        batch = []
        total_seconds = 0
        for row in ordered(rows,has_audio):
            if MODES[cond] != "audio" and not row.get("transcription"):
                record = base_record(row)
                record.update(prediction=None, scores={}, choices=row["ordered_choices"],
                              predicted_letter=None, raw_response="", error="No transcription available")
                commit([record])
                continue
            seconds = row.get("_seconds", 0)
            # Whisper uses a 160-sample hop. Fewer than three frames give zero
            # projected audio tokens after the Qwen2.5-Omni encoder pooling step.
            if has_audio and math.ceil(round(seconds * 16000) / 160) < 3:
                record = base_record(row)
                record.update(prediction=None, scores={}, choices=row["ordered_choices"],
                              predicted_letter=None, raw_response="",
                              error="No audio tokens after encoder pooling")
                commit([record])
                continue
            if batch and (len(batch) >= limit or total_seconds + seconds > args.max_audio_seconds):
                attempt(batch)
                batch = []
                total_seconds = 0
            batch.append(row)
            total_seconds += seconds
        attempt(batch)
    elapsed = time.monotonic()-start
    event("finish", initial=initial, new_rows=processed, total_rows=initial+processed,
          elapsed_sec=round(elapsed,4), peak_allocated_bytes=peak, oom_splits=splits)
    print(f"Finished {initial+processed} rows. This invocation {elapsed:.1f}s, peak batch GPU memory {peak/2**30:.2f} GiB.",flush=True)


if __name__ == "__main__":
    main()
