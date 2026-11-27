import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

root = Path(__file__).resolve().parents[1]
run = root/'research_outputs/relgate_v2_final'
def rows(p):
    return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]

records = rows(run/'scores/3way.jsonl')
scores = {r['id']:r for r in records}
assert len(records) == len(scores) == 4733 and all(r['finite'] for r in records)
occ = rows(root/'research_outputs/routeB_step2b_v1/report/gate_candidate_occurrences.jsonl')
arr = [dict(x, score=scores[x['triple']]['q']['CONTRADICTED']) for x in occ if x['swapped'] is not None]
def vals(view, swapped):
    return np.array([x['score'] for x in arr if x['view'] == view and x['swapped'] == swapped])
def metrics(neg, pos):
    assert len(neg) and len(pos)
    y=np.r_[np.zeros(len(neg)),np.ones(len(pos))]
    s=np.r_[neg,pos]
    return {'n_negative':len(neg),'n_positive':len(pos),'auroc':float(roc_auc_score(y,s)),'average_precision':float(average_precision_score(y,s))}
clean, coarse, swap = vals('Clean',False), vals('Coarse-Desc',False), vals('Name-Swapped',True)
result = {'clean_vs_swap':metrics(clean,swap)}
for view in ['Target-Swapped','Distractor-Swapped']:
    result[view] = metrics(vals(view,False), vals(view,True))
thresholds = np.unique(np.r_[coarse,swap,np.nextafter(1.,np.inf)])
valid = [float(t) for t in thresholds if np.mean(coarse>=t)<=.15 and np.mean(swap>=t)>=.65]
result['threshold_gate'] = {'pass':bool(valid),'min':min(valid) if valid else None,'max':max(valid) if valid else None}
result['gate_pass'] = bool(result['clean_vs_swap']['auroc']>=.88 and result['Target-Swapped']['auroc']>=.72 and result['Distractor-Swapped']['auroc']>=.80 and valid)
labels = {r['id']:r['label'] for r in rows(run/'score_inputs.jsonl') if r['id'].startswith('validation:')}
assert len(labels)==1000
pred = {i:max(scores[i]['q'],key=scores[i]['q'].get) for i in labels}
unverified = [i for i in labels if labels[i]=='UNVERIFIED']
result['validation'] = {'n':len(labels),'likelihood_argmax_accuracy':sum(pred[i]==labels[i] for i in labels)/len(labels),'unverified_contradicted_rate':sum(pred[i]=='CONTRADICTED' for i in unverified)/len(unverified),'note':'Argmax of full label-sequence likelihood; not free-generation accuracy.'}
result['note'] = 'Occurrence-level synthetic swap discrimination; swap is not independently verified semantic contradiction. Label-normalized scores are not calibrated probabilities.'
out=run/'audit_v2.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
