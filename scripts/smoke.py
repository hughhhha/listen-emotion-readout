"""Local synthetic shape/split/persistence tests. No dataset or model claims."""
import json, tempfile, subprocess, sys, hashlib
from pathlib import Path
import numpy as np
rng = np.random.default_rng(4)
conds = ["1_audio","1_audio_and_text","2B","2C","3B","3C","4","1_text","2A","3A"]
here = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(dir=here) as tmp:
 p=Path(tmp)
 with (p/'feat').open('w') as f,(p/'pred').open('w') as g:
  for split, groups in [('train',range(35)),('test',range(34,43))]:
   for i in groups:
    for j,c in enumerate(conds):
     if c=='4' and split=='test': continue
     group=hashlib.sha256(str(i).encode()).hexdigest()
     key=f"{i}/{c}/{split}"; labels=['anger','fear','sadness']; y=i%3
     z=rng.normal(size=3); z[y]+=0.5
     a=dict(id=key,condition=c,group=group,split=split,label=labels[y],features=dict(pitch_span=float(rng.normal()),rms_span=float(rng.normal()),pause_fraction=.2))
     b=dict(id=key,prediction=labels[z.argmax()],scores=dict(zip(labels,z)),embedding_path=f'{i}.npy')
     if i==42 and j==1: b.update(prediction=None,scores={})
     print(json.dumps(a),file=f); print(json.dumps(b),file=g)
     np.save(p/f'{i}.npy',np.eye(3)[y])
 for extra in ([],['--embeddings'],['--select-linear-head']):
  args=[sys.executable,str(here/'methods.py'),'--features',str(p/'feat'),'--predictions',str(p/'pred'),'--output',str(p/'res'),'--bootstrap','64','--protocol','official_plus4',*extra]
  subprocess.run(args,check=True,stdout=subprocess.DEVNULL)
  m=json.load(open(p/'res'/'metrics.json')); r=m['mixed_seven']
  assert r['complete'] and m['native_six']['complete']
  assert r['n_by_condition']['4']==7 and r['n_by_condition']['2C']==9
  assert r['n_by_condition']['1_audio_and_text']==8
  assert m['audit']['groups_crossing_input_splits']==0
  assert m['alignment']['jointly_skipped_null_prediction_and_scores']=={'native_test/1_audio_and_text':1}
  assert m['artifact_replay']['mismatches']==0
  print(extra,r['n_by_condition'],m['artifact_replay'])
  if '--select-linear-head' in extra:
   assert m['folds'][0]['selected_head']=="prosody_plus_encoder"
   assert 'fused_prosody' in m['per_condition']['3B']
   with (p/'feat_unlabelled').open('w') as out:
    for line in (p/'feat').open():
     row=json.loads(line); del row["label"]
     print(json.dumps(row),file=out)
   apply=[sys.executable,str(here/'methods.py'),'apply','--features',str(p/'feat_unlabelled'),'--predictions',str(p/'pred'),'--bundle',str(p/'res'/'fitted_heads.joblib'),'--expected',str(p/'res'/'heldout_predictions.jsonl'),'--selected-only','--output',str(p/'replay')]
   subprocess.run(apply,check=True)
