#!/usr/bin/env python3
r"""Classify one WAV with the same frozen Qwen Thinker and fitted LISTEN head.

    python scripts/predict_clip.py --model models/Qwen2.5-Omni-3B \
      --bundle results/fitted_heads.joblib --wav speaker.wav

    python scripts/predict_clip.py --model models/Qwen2.5-Omni-3B \
      --bundle results/fitted_heads.joblib --manifest data/test.jsonl --id test-00001-1_audio \
      --baseline-choices runs/test.jsonl

Place infer_fast.py, methods.py, prosody.py and this script in one directory. A fitted
joblib file contains Python objects and must come from your trusted training run.
"""
import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CHOICES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")


def find_row(path, key):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("id") == key:
                    return row
    raise ValueError(f"No row with id {key!r} in {path}")


def input_spec(args):
    from infer_fast import EXTRA, MODES, normalize

    if args.wav is not None:
        wav = args.wav
        condition = "1_audio"
        question = "Which emotion is the speaker conveying through their voice?"
        transcript = ""
        choices = list(DEFAULT_CHOICES)
        group = None
        order = "provided"
    else:
        row = find_row(args.manifest, args.id)
        condition = row["condition"]
        if MODES.get(condition) not in ("audio", "audio_and_text"):
            raise ValueError(f"Row {args.id} is not an audio-input condition")
        root = args.audio_root if args.audio_root is not None else args.manifest.parent
        wav = root / row["audio_path"]
        question = row["question"]
        transcript = row.get("transcription", "") if MODES[condition] == "audio_and_text" else ""
        if MODES[condition] == "audio_and_text" and not transcript:
            raise ValueError(f"Audio+text row {args.id} has no transcript.")
        group = row.get("group")  # Only for choosing a saved CV fold.
        offered = list(row["choices"])
        if condition in ("3B", "3C"):
            offered = list(dict.fromkeys([*offered, *EXTRA]))
        if args.baseline_choices:
            choices = find_row(args.baseline_choices, args.id)["choices"]
            order = "saved baseline order"
        else:
            saved = row.get("ordered_choices_by_experiment", {}).get(condition) or row.get(
                "ordered_choices"
            )
            choices = list(saved) if saved is not None else offered
            order = "saved manifest order" if saved is not None else "manifest order"
        if sorted(choices) != sorted(offered):
            raise ValueError("Saved order does not contain the same options as the source row.")
    if args.labels:
        choices = [v.strip() for v in args.labels.split(",")]
        order = "provided"
        if len(set(map(normalize, choices))) != len(choices):
            raise ValueError("Custom options must represent different canonical emotion classes.")
    if not 2 <= len(choices) <= 26 or any(
        not re.fullmatch(r"[a-zA-Z][a-zA-Z _-]*", v) for v in choices
    ):
        raise ValueError("Supply two to 26 English emotion labels.")
    if not wav.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav}")
    # Use an anonymous row ID and a fixed filename. Only the question, displayed
    # choices, optional transcript, and in-memory waveform enter prompt_for().
    clean = {
        "id": "clip",
        "condition": condition,
        "question": question,
        "transcription": transcript,
        "ordered_choices": choices,
        "audio_path": "clip.wav",
    }
    return wav, clean, group, order


def get_head(saved, group, fold):
    if fold is None:
        if len(saved["folds"]) == 1:
            fold = 0
        else:
            fold = saved["group_to_fold"].get(group)
    if fold is None or not 0 <= fold < len(saved["folds"]):
        raise ValueError("Choose --fold for a multi-fold bundle when the source group is unknown.")
    candidate = saved["folds"][fold]
    name = candidate["selected_head"]
    return name, candidate["candidates"][name]


def classify(args, engine_factory=None):
    import joblib
    import numpy as np
    import soundfile as sf
    from infer_fast import Inference, load_audio, normalize
    from methods import checkpoint_identity, mixed_probabilities, predict_acoustic, predict_bundle
    from prosody import extract

    wav, row, group, order = input_spec(args)
    saved = joblib.load(args.bundle)
    identity = saved.get("checkpoint_identity")
    if identity and checkpoint_identity(args.model) != identity:
        raise ValueError(
            "Checkpoint config or weight index differs from the one attached to this fitted head."
        )
    name, head = get_head(saved, group, args.fold)
    labels = head["classes"]
    unknown = {normalize(v) for v in row["ordered_choices"]} - set(labels)
    if unknown:
        raise ValueError(
            f"Classes unavailable in the fitted head: {sorted(unknown)}. Supported: {labels}"
        )
    with tempfile.TemporaryDirectory(prefix="listen-clip-") as directory:
        temp = Path(directory)
        # Model and prosody estimator receive exactly the same float waveform.
        sf.write(temp / "clip.wav", load_audio(wav), 16000, subtype="FLOAT")
        _, features = extract(("clip", temp / "clip.wav"))
        config = SimpleNamespace(
            model=str(args.model),
            device="cuda:0",
            dtype="bfloat16",
            attention=args.attention,
            embeddings=head["uses_embeddings"],
            max_new_tokens=3,
            fallback_cap=1024,
            stop_on_letter=True,
            trim_audio_padding=True,
            output=temp / "qwen.jsonl",
            audio_root=temp,
        )
        if engine_factory is None:
            engine = Inference(config)
            # The full Omni wrapper builds this setting from its Thinker config.
            # Loading the standalone weights can otherwise leave EOS unset.
            engine.thinker.generation_config.eos_token_id = engine.thinker.config.eos_token_id
        else:
            engine = engine_factory(config)
        (predicted,) = engine.run([row])
        numeric = np.array([features[k] for k in head["features"]], dtype=np.float32)
        if head["uses_embeddings"]:
            embedding = np.load(temp / predicted["embedding_path"], allow_pickle=False)
            if embedding.shape != (head["embedding_size"],):
                raise ValueError("The encoder embedding dimension disagrees with the fitted head.")
            numeric = np.concatenate((numeric, embedding)).astype(np.float32, copy=False)
        x = numeric[None, :]
        logits = np.array([[predicted["scores"].get(k, -np.inf) for k in labels]])
        categories = predict_bundle(x, logits, head)
        calibrated_logits = logits * head["calibrator"]["scale"] + head["calibrator"]["biases"]
        acoustic = predict_acoustic(x, head["acoustic"], len(labels))
        probs = mixed_probabilities(calibrated_logits, acoustic, head["fusion_weight"])[0]
    ranking = sorted(
        ((k, float(p)) for k, p, offered in zip(labels, probs, np.isfinite(logits[0])) if offered),
        key=lambda item: -item[1],
    )
    return {
        "condition": row["condition"],
        "choice_order": order,
        "choices": row["ordered_choices"],
        "original_model": predicted["prediction"],
        "calibrated": str(categories["calibrated"][0]),
        "acoustic_head": str(categories["acoustic"][0]),
        "fused": str(categories["fused"][0]),
        "fused_probabilities": {k: round(p, 6) for k, p in ranking},
        "selected_head": name,
        "acoustic_weight": head["fusion_weight"],
        "prosody": features,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model", required=True, type=Path, help="Matching local frozen Qwen2.5-Omni checkpoint."
    )
    p.add_argument(
        "--bundle", required=True, type=Path, help="Matching trusted fitted_heads.joblib."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path, help="A recording to classify.")
    source.add_argument("--manifest", type=Path, help="A condition-expanded LISTEN JSONL manifest.")
    p.add_argument("--id", help="Expanded sample ID for --manifest.")
    p.add_argument(
        "--audio-root", type=Path, help="Manifest audio root. Defaults to its parent directory."
    )
    p.add_argument("--labels", help="Comma-separated English candidates in display order.")
    p.add_argument(
        "--baseline-choices",
        type=Path,
        help="Baseline JSONL with the original shuffled choice order for the selected id.",
    )
    p.add_argument("--fold", type=int, help="Explicit fold for an experimental multi-fold bundle.")
    p.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    args = p.parse_args(argv)
    if args.manifest is not None and not args.id:
        p.error("--manifest requires --id")
    if args.wav is not None and (args.id or args.audio_root or args.baseline_choices):
        p.error("--id, --audio-root and --baseline-choices apply to --manifest")
    if args.baseline_choices and args.labels:
        p.error("Use one candidate list: --baseline-choices or --labels")
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This single-clip benchmark demo requires CUDA GPU 0.")
    print(json.dumps(classify(args), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
