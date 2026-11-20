"""Three waveform-only prosody statistics, computed once per unique 16 kHz recording."""
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import find_peaks
import soundfile as sf

def extract(item):
    group,path=item
    signal, rate=sf.read(path,dtype="float32")
    if rate != 16000 or signal.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz: {path}")
    frame_size,hop=640,160
    if len(signal)<frame_size:
        signal=np.pad(signal,(0,frame_size-len(signal)))
    frames=np.lib.stride_tricks.sliding_window_view(signal,frame_size)[::hop]
    rms_db=10*np.log10(np.maximum(np.mean(frames*frames,axis=1),1.e-10))
    threshold=max(-70., float(np.percentile(rms_db,90))-30.)
    active=rms_db>threshold
    active_idx=np.flatnonzero(active)
    pause=0. if not len(active_idx) else 1.-active[active_idx[0]:active_idx[-1]+1].mean()
    rms_span=np.ptp(np.percentile(rms_db[active],[10,90])) if active.any() else 0.
    voiced=frames[active] if active.any() else frames[:0]
    pitches=[]
    if len(voiced):
        window=(voiced-voiced.mean(axis=1,keepdims=True))*np.hanning(frame_size).astype(np.float32)
        power=np.abs(rfft(window,n=1280,axis=1))**2
        autocorr=irfft(power,n=1280,axis=1)[:,:267]
        autocorr/=np.maximum(autocorr[:,:1],1.e-12)
        for corr in autocorr:
            # Choose the first local peak close to the strongest plausible F0 peak.
            peaks,_=find_peaks(corr[31:266])
            peaks+=31
            if len(peaks):
                strongest=corr[peaks].max()
                if strongest>=.35:
                    lag=peaks[corr[peaks]>=.9*strongest][0]
                    pitches.append(12*np.log2(rate/lag))
    span=np.ptp(np.percentile(pitches,[10,90])) if len(pitches)>1 else 0.
    return group,{"pitch_span":float(span),"rms_span":float(rms_span),"pause_fraction":float(pause)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path); p.add_argument("--workers",type=int,default=6)
    args=p.parse_args()
    rows=[]
    for split in ("train","test"):
        with (args.data/f"{split}.jsonl").open() as reader:
            rows.extend(json.loads(line) for line in reader)
    paths={r["audio_hash"]:str(args.data/r["audio_path"]) for r in rows if r["audio_path"] and r["condition"] not in {"1_text","2A","3A"}}
    with concurrent.futures.ProcessPoolExecutor(args.workers) as executor:
        by_group=dict(executor.map(extract,paths.items(),chunksize=16))
    target=args.data/"features.jsonl"
    with target.with_suffix(".partial").open("w") as out:
        for row in rows:
            keys=("id","group","split","condition","label","dataset_source","source_id","explicit_emotion","audio_hash")
            rec={k:row[k] for k in keys}
            rec["features"]=by_group.get(row["audio_hash"],{})
            out.write(json.dumps(rec)+"\n")
    target.with_suffix(".partial").replace(target)
    print(len(rows),"rows;",len(paths),"unique acoustic inputs",flush=True)
if __name__=="__main__":main()
