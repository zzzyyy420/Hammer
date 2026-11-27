import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from routeb_stage1 import rows, indexed, split_prompt, dump
from routeb_validation import R

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_outputs/step_a_v1'
train=rows(ROOT/'research_outputs/relgate_v2_final/data/train_3way.jsonl')
def words(s): return re.findall(r'[a-z0-9]+',s.lower())
def grams(s):
    w=words(s); return set(zip(w,w[1:],w[2:],w[3:]))
used=set()
for x in train:
    if x['output']=='UNVERIFIED':
        desc=x['instruction'].split('\nDescription: ',1)[1].split('\n\nParameter schema:',1)[0]
        used.update(grams(desc))
templates=[
 'This endpoint is part of an online resource service. Its interface accepts structured inputs and returns a result.',
 'An accessible service interface for resource information, with supplied fields and a corresponding response.',
 'A resource-oriented interface is available here; callers supply structured fields and receive a response.'
]
assert all(75<=len(t)<=120 and not (grams(t)&used) for t in templates)
opposites={'get':['delete','remove'],'fetch':['delete','remove'],'list':['delete','remove'],'read':['delete','remove'],'query':['delete','remove'],'search':['delete','remove'],'create':['delete','cancel'],'add':['delete','remove'],'insert':['delete','remove'],'post':['delete','remove'],'delete':['get','create'],'remove':['get','create'],'cancel':['get','create'],'enable':['disable','stop'],'start':['stop','disable'],'open':['close','stop'],'disable':['enable','start'],'stop':['start','enable'],'close':['open','start'],'upload':['download','fetch'],'send':['receive','fetch'],'push':['pull','fetch'],'download':['upload','send'],'encrypt':['decrypt'],'decrypt':['encrypt']}
src=indexed(rows(R/'inputs/keep.jsonl'))
cases=[c for c in rows(R/'cases.jsonl') if c['view']=='Clean']
keys={(x['id'],x['view']):x for x in rows(ROOT/'research_outputs/routeB_stage1_v1/eval_keys.jsonl') if x['mode']=='Keep-Name'}
assert len(cases)==197
out=[]; examples=[]
for c in cases:
    pre,ts,post=split_prompt(src[c['prompt']]); g=keys[(c['id'],'Clean')]['g']; original=ts[g]['name']
    verbose=deepcopy(ts)
    for i,t in enumerate(verbose): t['description']=templates[i%3]
    out.append(dict(id=c['id'],view='Verbose-Coarse',tools=verbose,g=g,instruction=pre+dump(verbose)+post))
    m=re.match(r'^([a-z]+)(?=(_|[A-Z]|$))',original)
    if not m or m.group(1) not in opposites: continue
    old=m.group(1); suffix=original[len(old):]; occupied={t['name'] for t in ts}
    options=[v+suffix for v in opposites[old]]
    replacement=next((n for n in options if n not in occupied),None)
    if replacement is None: continue
    flipped=deepcopy(ts); flipped[g]['name']=replacement
    assert len({t['name'] for t in flipped})==len(flipped) and replacement not in occupied
    out.append(dict(id=c['id'],view='Verb-Flip-Target',tools=flipped,g=g,original_name=original,instruction=pre+dump(flipped)+post))
    examples.append(dict(id=c['id'],old=original,new=replacement))
OUT.mkdir(parents=True,exist_ok=True)
dest=OUT/'probes.jsonl'
if dest.exists(): raise RuntimeError('Refusing to overwrite frozen probes')
dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf8')
audit={'clean_n':197,'flip_n':len(examples),'flip_examples':examples[:10],'verbose_description_lengths':sorted(map(len,templates)),'verbose_examples':templates,'fourgram_overlap':0,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
(OUT/'probe_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(audit,indent=2))
