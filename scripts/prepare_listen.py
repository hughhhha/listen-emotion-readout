"""Expand the pinned LISTEN release to its 10 author-defined input conditions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

CONDITIONS = {"1": ["1_text","1_audio","1_audio_and_text"], "2A":["2A"], "2B":["2B","2C"], "3A":["3A"], "3B":["3B","3C"], "4":["4"]}
EXTRA = {"neutral","sadness","surprise","happiness","fear","anger","excitement","frustration"}
ALIASES = {"angry":"anger","happy":"happiness","sad":"sadness","pleasant_surprise":"surprise"}
def normalized(x):
    v = (x or "").lower().strip()
    return ALIASES.get(v,v)

def save_audio(data,root,encoded_cache):
    blob = data.get("bytes") if data else None
    if not blob:
        return None
    key = hashlib.sha256(blob).digest()
    if key in encoded_cache:
        return encoded_cache[key]
    samples, sr = sf.read(io.BytesIO(blob),always_2d=True,dtype="float32")
    samples = samples.mean(axis=1)
    if sr != 16000:
        samples = soxr.resample(samples,sr,16000,quality="HQ")
    samples = np.ascontiguousarray(samples,dtype="<f4")
    digest = hashlib.sha256(samples.tobytes()).hexdigest()
    name = f"audio/{digest[:2]}/{digest}.wav"
    dest = root/name
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        sf.write(dest,samples,16000,subtype="FLOAT")
    item = (name, digest, len(samples)/16000)
    encoded_cache[key] = item
    return item

def protect_repeated_clips(root, meta):
    # Rows from different question types may use the same underlying recording ID.
    # Union exact waveform duplicates and matching clip identifiers before splitting.
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        while x != parent[x]:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    paths=[root/"train.jsonl",root/"test.jsonl"]
    for path in paths:
        with path.open() as f:
            for line in f:
                row=json.loads(line)
                if not row["audio_path"]: continue
                clip=re.sub(r"_(?:2A|2B|3A|3B)$","",row["source_id"])
                union("a:"+row["group"],"id:"+row["dataset_source"]+"/"+clip)
    hist={}
    for path in paths:
        groups=set()
        with path.open() as f, path.with_suffix(".grouping").open("w") as out:
            for line in f:
                row=json.loads(line)
                row["audio_hash"]=row["group"] if row["audio_path"] else None
                if row["audio_path"]:
                    row["group"]=find("a:"+row["group"])[2:]
                if row["condition"] not in {"1_text","2A","3A"}:
                    groups.add(row["group"])
                out.write(json.dumps(row,ensure_ascii=False)+"\n")
        path.with_suffix(".grouping").replace(path)
        hist[path.stem]=groups
        meta[path.stem]["protected_groups"]=len(groups)
    meta["both_split_protected_groups"]=len(hist["train"]&hist["test"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    args=p.parse_args()
    root=args.data
    meta={"source":"VibeCheck1/LISTEN_full","revision":"f6397b5bfedd85ae41b80dac13e95bb55adf0a35"}
    encoded_cache={}
    seen=defaultdict(set)
    for split in ("train","test"):
        files=sorted((root/"parquet").glob(f"{split}-*.parquet"))
        if len(files) != (3 if split=="train" else 1):
            raise RuntimeError(f"Missing shards for {split}")
        raw=Counter(); skipped=Counter(); count=Counter(); sources=Counter()
        rowno=0
        with (root/f"{split}.jsonl.partial").open("w") as out:
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        et=row["experiment_type"]
                        raw[et]+=1
                        rowno+=1
                        if et not in CONDITIONS:
                            continue
                        gold=row["answer"]
                        lexical=row["explicit_emotion"]
                        audio=save_audio(row.get("audio"),root,encoded_cache)
                        # Only neutral variants reuse exact '1' rows. Separate task-specific
                        # records in the source retain their original wording and order.
                        for cc in CONDITIONS[et]:
                            target = lexical if cc in {"2A","3A"} else gold
                            options=row["choices"] or []
                            available = set(options)
                            if cc in {"3A","3B","3C"}: available.update(EXTRA)
                            if not target or not target.strip() or target not in available:
                                skipped[cc]+=1
                                continue
                            if audio is None and cc not in {"1_text","2A","3A"}:
                                raise RuntimeError(f"audio missing: {split}/{rowno}/{cc}")
                            name,g,secs=audio or (None,f"text-{rowno}",0.0)
                            if cc not in {"1_text","2A","3A"}: seen[split].add(g)
                            uid=f"{split}-{rowno:05d}-{cc}"
                            sample={"id":uid,"source_id":row["id"],"condition":cc,
                                    "experiment_type":et,"split":split,"group":g,
                                    "dataset_source":row["dataset_source"],"question":row["question"],
                                    "choices":options,"answer":gold,"explicit_emotion":lexical,
                                    "implicit_emotion":row.get("implicit_emotion"),"label":normalized(target),
                                    "transcription":row["transcription"] or "",
                                    "audio_path":name,"duration":secs}
                            out.write(json.dumps(sample,ensure_ascii=False)+"\n")
                            count[cc]+=1
                            sources[str(row["dataset_source"])]+=1
        (root/f"{split}.jsonl.partial").replace(root/f"{split}.jsonl")
        meta[split]={"raw":dict(raw),"skipped_by_upstream":dict(skipped),"conditions":dict(count),"sources":dict(sources),"audio_groups":len(seen[split])}
        print(split,meta[split],flush=True)
    meta["both_split_audio_groups"]=len(seen["train"] & seen["test"])
    meta["unique_audio_groups_written"]=len({v[1] for v in encoded_cache.values()})
    protect_repeated_clips(root,meta)
    (root/"data_stats.json").write_text(json.dumps(meta,indent=2))
if __name__=="__main__": main()
