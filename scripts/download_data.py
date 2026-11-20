"""Fetch and verify the exact four Parquet shards of the public LISTEN release."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

target, tree = Path(sys.argv[1]), Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
version = "f6397b5bfedd85ae41b80dac13e95bb55adf0a35"
files = [r for r in json.loads(tree.read_text()) if r["path"].endswith(".parquet")]

def fetch(entry):
    path = target / Path(entry["path"]).name
    partial = path.with_suffix(".part")
    if not path.exists():
        url = f"https://huggingface.co/datasets/VibeCheck1/LISTEN_full/resolve/{version}/{entry['path']}?download=true&fetch={int(time.time())}"
        subprocess.run(["curl", "-fLsS", "--connect-timeout", "12", "--retry", "4", "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "45", "-C", "-", "-o", str(partial), url], check=True)
        with partial.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == entry["lfs"]["oid"], entry["path"]
        partial.rename(path)
    assert path.stat().st_size == entry["size"]
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == entry["lfs"]["oid"], path
    print(path.name, path.stat().st_size, digest, flush=True)

with concurrent.futures.ThreadPoolExecutor(4) as pool:
    list(pool.map(fetch, files))
