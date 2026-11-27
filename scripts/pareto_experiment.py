"""Prepare exact-prompt inference queues and score the frozen exploratory IDs."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pareto_policies import apply_name_mask, random_mask, dose_matched_mask
from relgate_evaluate import cache, adapter, summarise
from routeb_validation import R, gold
from routeb_stage1 import rows, indexed, split_prompt, msg, dump
from routeb_step2a import fingerprint, save_table

H=Path(__file__).resolve().parents[1]
P=H/'research_outputs/relgate_pareto'
CALLERS=['Pure-NameMask','B1-100','B2']
MASK_CALLERS=['Mask-50','Mask-100']
MAIN={'Coarse-Desc','Name-Swapped'}
MIXED={'Target-Swapped','Distractor-Swapped'}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for r in data:f.write(dump(r)+'\n')
    temp.replace(path)


def predictions(model):
    result={} if model in MASK_CALLERS else cache(model)
    for f in sorted((H/'research_outputs/relgate_v1/results').glob(model+'__*.jsonl')):
        for r in rows(f):result.setdefault(r['id'],r)
    for f in sorted((P/'results').glob(model+'__*.jsonl')):
        for r in rows(f):result.setdefault(r['id'],r)
    return result


def prepare(stage):
    cfg=json.loads((P/'experiment_config.json').read_bytes())
    cases=rows(R/'cases.jsonl');original=indexed(rows(R/'inputs/keep.jsonl'))
    assert Counter(c['view'] for c in cases)==cfg['evaluation']['denominators']
    scores={}
    if stage=='semantic':
        for way in ('3way','2way'):
            seq=rows(P/'scores'/f'{way}.jsonl')
            scores[way]={r['id']:r for r in seq if not r['id'].startswith('validation:')}
            assert len(scores[way])==3733
    schedules=[];jobs={}
    for model in (MASK_CALLERS if stage=='maskeval' else CALLERS):
        cc=[];requests={};known=predictions(model)
        for c in cases:
            primary=c['view'] in MAIN
            mixed=c['view'] in MIXED
            if stage!='maskeval' and not primary and not (stage=='semantic' and mixed):continue
            pre,tools,post=split_prompt(original[c['prompt']]);n=len(tools)
            choices=[]
            if stage=='maskeval':
                choices=[('Keep-Name',[False]*n)]
            elif stage=='baseline':
                choices=[('Keep-Name',[False]*n),('Always-Mask',[True]*n)]
                choices += [(f'Random-{ratio}-seed{seed}',random_mask(c['id'],n,ratio,seed))
                    for ratio in (.25,.5,.75) for seed in (42,123,2026)]
                choices += [(f'Length-{tau}',[len(t.get('description') or '')>tau for t in tools]) for tau in (32,64,128)]
            else:
                for way in ('3way','2way'):
                    if way=='2way' and model=='B1-100':continue
                    for tau in (.1,.3,.5,.7,.9):
                        if mixed and tau not in (.5,.9):continue
                        selected=[scores[way][key]['q']['CONTRADICTED']>=tau
                                  if scores[way][key]['finite'] else False for key in c['triples']]
                        choices.append((f'RelGate-{way}-tau{tau}',selected))
                        if way=='3way':
                            choices += [(f'Matched-{way}-tau{tau}-seed{seed}',
                                dose_matched_mask(c['id'],n,sum(selected),seed)) for seed in (42,123,2026)]
            for policy,selected in choices:
                new,inverse=apply_name_mask(tools,c['aliases'],selected)
                messages=msg(pre+dump(new)+post) if any(selected) else original[c['prompt']]['messages']
                prompt=fingerprint(dict(messages=messages))
                requests[prompt]=dict(id=prompt,messages=messages)
                cc.append(dict(id=c['id'],view=c['view'],policy=policy,prompt=prompt,
                    inverse=inverse,masked_count=sum(selected),candidate_count=n))
        job=model+'__'+stage
        todo=[requests[h] for h in sorted(requests) if h not in known]
        save(P/'inputs'/f'{job}.jsonl',todo);save(P/'cases'/f'{job}.jsonl',cc)
        model_adapter=(json.loads((P/model/'verified.json').read_bytes())['adapter']
                       if model in MASK_CALLERS else adapter(model))
        jobs[job]=dict(base='/root/autodl-tmp/Hammer/models/Qwen2-1.5B-Instruct',adapter=model_adapter)
        schedules.append(dict(job=job,cells=len(cc),unique_prompts=len(requests),
            reused_unique=len(requests)-len(todo),new_unique=len(todo)))
    (P/f'jobs_{stage}.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    (P/f'schedule_{stage}.json').write_text(json.dumps(schedules,indent=2),encoding='utf-8')
    print(json.dumps(schedules,indent=2),flush=True)


def score():
    keys=gold();metrics=[];actions=[]
    # summarise() enforces the original per-view denominators and fixed strict-match denominator.
    for model in CALLERS+MASK_CALLERS:
        src=predictions(model)
        for stage in ('baseline','semantic','maskeval'):
            f=P/'cases'/f'{model}__{stage}.jsonl'
            if not f.exists():continue
            policies=defaultdict(list)
            for c in rows(f):policies[c['policy']].append(c)
            for policy,cc in policies.items():
                missing={c['prompt'] for c in cc if c['prompt'] not in src}
                if missing:continue
                result=summarise(model,policy,cc,src,keys,prediction_dir=P/'predictions')
                metrics+=result
                for view in {c['view'] for c in cc}:
                    seq=[c for c in cc if c['view']==view]
                    actions.append(dict(model=model,policy=policy,view=view,
                        masked=sum(c['masked_count'] for c in seq),candidates=sum(c['candidate_count'] for c in seq),
                        request_masked=sum(bool(c['masked_count']) for c in seq),requests=len(seq)))
    if metrics:
        save_table(P/'metrics.csv',metrics);save_table(P/'action_rates.csv',actions)
    print(json.dumps(dict(scored_metric_rows=len(metrics)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['baseline','semantic','maskeval','score']);a=p.parse_args()
    score() if a.action=='score' else prepare(a.action)
