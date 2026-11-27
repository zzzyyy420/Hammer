import json
from collections import defaultdict
from pathlib import Path
from pareto_experiment import P, save, predictions
from pareto_policies import apply_name_mask, dose_matched_mask
from relgate_evaluate import adapter, summarise
from routeb_validation import R, gold
from routeb_stage1 import rows, indexed, split_prompt, msg, dump
from routeb_step2a import fingerprint, save_table

H = Path(__file__).resolve().parents[1]
RUN = H / 'research_outputs/relgate_v2_final'
VIEWS = {'Clean', 'Coarse-Desc', 'Name-Swapped', 'Target-Swapped', 'Distractor-Swapped', 'Desc-Degraded'}
TAUS = (.30, .50, .70)
JOB = 'B2__relgate_v2'

def prepare_matched():
    source=indexed(rows(R/'inputs/keep.jsonl')); known=predictions('B2')
    originals={(c['id'],c['view'],c['policy']):c for c in rows(RUN/'b2_cases.jsonl')}
    keys={(x['id'],x['view']):x for x in rows(H/'research_outputs/routeB_stage1_v1/eval_keys.jsonl') if x['mode']=='Keep-Name'}
    requests={}; output=[]
    for c in rows(R/'cases.jsonl'):
        if c['view'] not in {'Target-Swapped','Distractor-Swapped'}: continue
        pre,tools,post=split_prompt(source[c['prompt']]); n=len(tools); key=keys[(c['id'],c['view'])]; g=key['g']
        for tau in (.30,.50):
            k=originals[(c['id'],c['view'],f'RelGate-{tau:.2f}')]['masked_count']
            for seed in (42,123,2026):
                sel=dose_matched_mask(c['id'],n,k,seed)
                new,inv=apply_name_mask(tools,c['aliases'],sel)
                messages=msg(pre+dump(new)+post) if any(sel) else source[c['prompt']]['messages']
                h=fingerprint(dict(messages=messages)); requests[h]=dict(id=h,messages=messages)
                output.append(dict(id=c['id'],view=c['view'],policy=f'Matched-{tau:.2f}-seed{seed}',prompt=h,inverse=inv,masked_count=sum(sel),candidate_count=n,target_masked=sel[g],target_swapped=key['flags'][g]))
    job='B2__relgate_v2matched'; todo=[requests[h] for h in sorted(requests) if h not in known]
    save(P/'inputs'/f'{job}.jsonl',todo); save(RUN/'b2_matched_cases.jsonl',output)
    (P/'jobs_relgate_v2matched.json').write_text(json.dumps({job:dict(base='/root/autodl-tmp/Hammer/models/Qwen2-1.5B-Instruct',adapter=adapter('B2'))},indent=2),encoding='utf8')
    print(json.dumps(dict(cells=len(output),unique=len(requests),reused=len(requests)-len(todo),new=len(todo))))

def prepare_default():
    labels=indexed(rows(RUN/'greedy_labels.jsonl'))
    assert len(labels)==4733
    source=indexed(rows(R/'inputs/keep.jsonl')); known=predictions('B2')
    cases=[c for c in rows(R/'cases.jsonl') if c['view'] in VIEWS]
    keys={(x['id'],x['view']):x for x in rows(H/'research_outputs/routeB_stage1_v1/eval_keys.jsonl') if x['mode']=='Keep-Name'}
    requests={}; output=[]
    for c in cases:
        pre,tools,post=split_prompt(source[c['prompt']]); key=keys[(c['id'],c['view'])]; g=key['g']
        sel=[labels[x]['raw']=='CONTRADICTED' for x in c['triples']]
        new,inv=apply_name_mask(tools,c['aliases'],sel)
        messages=msg(pre+dump(new)+post) if any(sel) else source[c['prompt']]['messages']
        h=fingerprint(dict(messages=messages)); requests[h]=dict(id=h,messages=messages)
        output.append(dict(id=c['id'],view=c['view'],policy='RelGate-default',prompt=h,inverse=inv,masked_count=sum(sel),candidate_count=len(tools),target_masked=sel[g],target_swapped=key['flags'][g]))
    job='B2__relgate_v2default'; todo=[requests[h] for h in sorted(requests) if h not in known]
    save(P/'inputs'/f'{job}.jsonl',todo); save(RUN/'b2_default_cases.jsonl',output)
    (P/'jobs_relgate_v2default.json').write_text(json.dumps({job:dict(base='/root/autodl-tmp/Hammer/models/Qwen2-1.5B-Instruct',adapter=adapter('B2'))},indent=2),encoding='utf8')
    print(json.dumps(dict(cells=len(output),unique=len(requests),reused=len(requests)-len(todo),new=len(todo))))

def prepare():
    scores = indexed(rows(RUN/'scores/3way.jsonl'))
    source = indexed(rows(R/'inputs/keep.jsonl'))
    known = predictions('B2')
    cases = [c for c in rows(R/'cases.jsonl') if c['view'] in VIEWS]
    keys = {(x['id'],x['view']):x for x in rows(H/'research_outputs/routeB_stage1_v1/eval_keys.jsonl') if x['mode']=='Keep-Name'}
    requests, output = {}, []
    for c in cases:
        pre, tools, post = split_prompt(source[c['prompt']]); n=len(tools)
        key=keys[(c['id'],c['view'])]; g=key['g']
        choices=[('Keep-Name',[False]*n),('Always-Mask',[True]*n),('Length-64',[len(t.get('description') or '')>64 for t in tools])]
        choices += [(f'RelGate-{t:.2f}',[scores[x]['q']['CONTRADICTED']>=t for x in c['triples']]) for t in TAUS]
        for policy, sel in choices:
            new, inv=apply_name_mask(tools,c['aliases'],sel)
            messages=msg(pre+dump(new)+post) if any(sel) else source[c['prompt']]['messages']
            h=fingerprint(dict(messages=messages)); requests[h]=dict(id=h,messages=messages)
            output.append(dict(id=c['id'],view=c['view'],policy=policy,prompt=h,inverse=inv,masked_count=sum(sel),candidate_count=n,target_masked=sel[g],target_swapped=key['flags'][g]))
    todo=[requests[h] for h in sorted(requests) if h not in known]
    save(P/'inputs'/f'{JOB}.jsonl',todo); save(RUN/'b2_cases.jsonl',output)
    (P/'jobs_relgate_v2.json').write_text(json.dumps({JOB:dict(base='/root/autodl-tmp/Hammer/models/Qwen2-1.5B-Instruct',adapter=adapter('B2'))},indent=2),encoding='utf8')
    print(json.dumps(dict(cases=len(cases),cells=len(output),unique=len(requests),reused=len(requests)-len(todo),new=len(todo))))

def score():
    src=predictions('B2'); keys=gold(); by=defaultdict(list)
    paths=[RUN/'b2_cases.jsonl', RUN/'b2_default_cases.jsonl']
    if (RUN/'b2_matched_cases.jsonl').exists(): paths.append(RUN/'b2_matched_cases.jsonl')
    for path in paths:
        for c in rows(path): by[c['policy']].append(c)
    metrics=[]; actions=[]
    for policy,cc in by.items():
        assert all(c['prompt'] in src for c in cc), policy
        metrics.extend(summarise('B2',policy,cc,src,keys,prediction_dir=RUN/'predictions'))
        for view in sorted(VIEWS):
            group=[c for c in cc if c['view']==view]; un=[c for c in group if not c['target_swapped']]
            actions.append(dict(policy=policy,view=view,n=len(group),masked=sum(c['masked_count'] for c in group),candidates=sum(c['candidate_count'] for c in group),target_unchanged_n=len(un),target_false_masks=sum(c['target_masked'] for c in un)))
    save_table(RUN/'b2_metrics.csv',metrics); save_table(RUN/'b2_actions.csv',actions)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'prepare_default':prepare_default,'prepare_matched':prepare_matched,'score':score}[sys.argv[1]]()
