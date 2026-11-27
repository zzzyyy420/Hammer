"""Build metadata-only gate requests and frozen caller policies; score matched IDs."""
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

from routeb_validation import R,VIEWS,BASELINES,existing,gold,tid
from routeb_step2a import fingerprint,save_table,outcome
from routeb_stage1 import rows,indexed,split_prompt,msg,dump,remap,wrapper
from relgate_cpu_audit import DEST,write,jsonl,hm,fixed_length

REMOTE='/root/autodl-tmp/Hammer'
RR=REMOTE+'/research_outputs/relgate_v1'
BASE=REMOTE+'/models/Qwen2-1.5B-Instruct'
MOUNTS={'3way':['Pure-NameMask','B2','B1-100'],'2way':['Pure-NameMask','B2']}


def adapter(model):
    if model in ('Pure-NameMask','Pure-NoMask'):
        return REMOTE+'/research_outputs/phase1_gpu_v1/'+model
    return json.loads((R/model/'verified.json').read_bytes())['adapter']


def cache(model):
    if model in BASELINES:return existing(model)
    return indexed(rows(R/'results'/f'{model}.jsonl'))


def prepare():
    jobs={}
    for way in ('3way','2way'):
        val=rows(DEST/'data'/f'validation_{way}.jsonl')
        inp=rows(R/'inputs/self_gate.jsonl')
        inp+= [dict(id='validation:'+str(i),instruction=r['instruction']) for i,r in enumerate(val)]
        jsonl(DEST/'inputs'/f'gate_{way}.jsonl',inp)
        jobs['gate_'+way]=dict(base=BASE,adapter=RR+'/RelGate-'+('3Way' if way=='3way' else '2Way')+'/adapter')
    for model in BASELINES:
        job=model+'__Length-64'
        jobs[job]=dict(base=REMOTE+'/models/Hammer-1.5b') if model=='Hammer-1.5b' else dict(base=BASE,adapter=adapter(model))
    write(DEST/'jobs.json',jobs)


def gate_quality():
    detail=rows(R/'report/gate_candidate_occurrences.jsonl')
    results=[];validation=[];unique=[]
    for way in MOUNTS:
        predictions=indexed(rows(DEST/'results'/f'gate_{way}.jsonl'))
        labels={k:r['raw'].strip() for k,r in predictions.items() if not k.startswith('validation:')}
        valid={'SUPPORTED','CONTRADICTED'}|({'UNVERIFIED'} if way=='3way' else set())
        assert set(labels)==set(indexed(rows(R/'inputs/self_gate.jsonl')))
        counters=defaultdict(Counter);occurrences=[];flags=defaultdict(set)
        for r in detail:
            raw=labels[r['triple']];hit=raw=='CONTRADICTED';bad=raw not in valid
            target=r['swapped']
            if target is not None:flags[r['triple']].add(target)
            occurrences.append(dict(r,raw=raw,parse_failure=bad,masked=hit))
            for group in (r['view'],'all'):
                cc=counters[group];cc['n']+=1;cc['invalid']+=bad
                if target is not None:cc['tp' if target and hit else 'fn' if target else 'fp' if hit else 'tn']+=1
        def summarize(group,cc,unit):
            tp,fp,tn,fn=[cc[x] for x in ('tp','fp','tn','fn')]
            div=lambda a,b:a/b if b else None
            return dict(gate=way,view=group,unit=unit,n=cc['n'],tp=tp,fp=fp,tn=tn,fn=fn,
                precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),fpr=div(fp,fp+tn),
                invalid=cc['invalid'],parse_failure_rate=div(cc['invalid'],cc['n']))
        results.extend(summarize(v,c,'candidate_occurrence') for v,c in counters.items())
        cc=Counter();conflicts=0;anonymous=0
        for key,raw in labels.items():
            hit=raw=='CONTRADICTED';cc['n']+=1;cc['invalid']+=raw not in valid
            if len(flags[key])==1:
                target=next(iter(flags[key]));cc['tp' if target and hit else 'fn' if target else 'fp' if hit else 'tn']+=1
            elif flags[key]:conflicts+=1
            else:anonymous+=1
        u=summarize('all_unique',cc,'unique_triple');u.update(conflicting_status=conflicts,no_swap_binary_label=anonymous)
        unique.append(u)
        jsonl(DEST/'report'/f'gate_{way}_candidate_occurrences.jsonl',occurrences)
        vals=rows(DEST/'data'/f'validation_{way}.jsonl')
        confusion=Counter((v['output'],predictions['validation:'+str(i)]['raw'].strip()) for i,v in enumerate(vals))
        for target in sorted(valid):
            tp=sum(n for (g,p),n in confusion.items() if g==p==target)
            fp=sum(n for (g,p),n in confusion.items() if p==target and g!=target)
            fn=sum(n for (g,p),n in confusion.items() if g==target and p!=target)
            validation.append(dict(gate=way,label=target,n=sum(n for (g,p),n in confusion.items() if g==target),
                tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
    save_table(DEST/'report/gate_quality.csv',results)
    save_table(DEST/'report/gate_quality_unique.csv',unique)
    save_table(DEST/'report/internal_validation.csv',validation)


def gated():
    jobs=json.loads((DEST/'jobs.json').read_bytes())
    original=indexed(rows(R/'inputs/keep.jsonl'));cases=rows(R/'cases.jsonl')
    allcases=[]
    for way,models in MOUNTS.items():
        labels={r['id']:r['raw'].strip() for r in rows(DEST/'results'/f'gate_{way}.jsonl')}
        assert {k for k in labels if not k.startswith('validation:')} == set(indexed(rows(R/'inputs/self_gate.jsonl')))
        cc=[];requests={}
        for c in cases:
            prefix,tools,suffix=split_prompt(original[c['prompt']]);inverse={}
            assert len(tools)==len(c['triples'])==len(c['aliases'])
            for t,alias,key in zip(tools,c['aliases'],c['triples'],strict=True):
                assert tid(t)[0]==key
                if labels[key]=='CONTRADICTED':inverse[alias]=t['name'];t['name']=alias
            assert len({t['name'] for t in tools})==len(tools)
            messages=msg(prefix+dump(tools)+suffix) if inverse else original[c['prompt']]['messages']
            h=fingerprint(dict(messages=messages));requests[h]=dict(id=h,messages=messages)
            cc.append(dict(id=c['id'],view=c['view'],prompt=h,inverse=inverse))
        for model in models:
            job=model+'__RelGate-'+way
            src=cache(model);missing={c['prompt'] for c in cc if c['prompt'] not in src}
            jsonl(DEST/'inputs'/f'{job}.jsonl',(requests[k] for k in sorted(missing)))
            jsonl(DEST/'cases'/f'{job}.jsonl',cc)
            jobs[job]=dict(base=BASE,adapter=adapter(model))
            allcases.append(dict(job=job,n_cases=len(cc),new_unique_prompts=len(missing),
                                 reused_cases=sum(c['prompt'] in src for c in cc)))
    write(DEST/'jobs.json',jobs);write(DEST/'schedule.json',allcases)
    print(json.dumps(allcases,indent=2),flush=True)


def summarise(model,mode,cc,src,keys,prediction_dir=None):
    assert len({(c['view'],c['id']) for c in cc})==len(cc), 'duplicate evaluation request'
    expected=Counter(c['view'] for c in rows(R/'cases.jsonl'))
    observed=Counter(c['view'] for c in cc)
    assert all(n==expected[v] for v,n in observed.items()), 'partial evaluation denominator'
    predictions=[];cells=defaultdict(list)
    for c in cc:
        p=src[c['prompt']]
        res=outcome(dict(p,parsed_pred=remap(p['parsed_pred'],c.get('inverse',{}))),keys[c['view'],c['id']])
        cells[c['view']].append(res)
        predictions.append(dict(id=c['id'],view=c['view'],prompt=c['prompt'],raw=p.get('raw'),
            inverse=c.get('inverse',{}),tool_correct=res[0],strict_correct=res[1],
            name_hijack=res[2] if c['view'] in ('Name-Swapped','Target-Swapped') else None,invalid=res[3]))
    jsonl((prediction_dir or DEST/'report/predictions')/f'{model}__{mode}.jsonl',predictions)
    metrics=[]
    for view,rs in cells.items():
        a,b,h,inv=map(sum,zip(*rs));n=len(rs);defined=view in ('Name-Swapped','Target-Swapped')
        metrics.append(dict(model=model,mode=mode,view=view,n=n,correct=a,tool_acc=a/n,strict_correct=b,
            strict_acc=b/n,invalid=inv,hijacks=h if defined else None,nhr_abs=h/n if defined else None,
            nhr_err=h/(n-a) if defined and n>a else None,H_dual=None,mixed_mean=None,mixed_harmonic=None))
    rates={r['view']:r['tool_acc'] for r in metrics}
    h=hm(rates['Coarse-Desc'],rates['Name-Swapped']) if 'Coarse-Desc' in rates and 'Name-Swapped' in rates else None
    a,b=rates.get('Target-Swapped'),rates.get('Distractor-Swapped')
    for r in metrics:
        r.update(H_dual=h,mixed_mean=(a+b)/2 if a is not None and b is not None else None,
            mixed_harmonic=hm(a,b) if a is not None and b is not None else None)
    return metrics


def score():
    (DEST/'report').mkdir(exist_ok=True)
    gate_quality();fixed_length(emit_inputs=False)
    keys=gold();metrics=[];cc=rows(R/'cases.jsonl')
    original=indexed(rows(R/'inputs/keep.jsonl'))
    for model in [*BASELINES,'B1','B1-100','B2']:
        metrics+=summarise(model,'Keep-Name',cc,cache(model),keys)
    # Rebuild global policies from frozen aliases, not the old per-view best table.
    for model in BASELINES:
        src=cache(model)
        src.update(indexed(rows(DEST/'results'/f'{model}__Length-64.jsonl')))
        for mode in ('Always-Mask','Global-Length-64','Oracle'):
            policy=[]
            for c in cc:
                if mode=='Oracle' and c['view'] not in ('Target-Swapped','Distractor-Swapped'):continue
                pre,tools,post=split_prompt(original[c['prompt']])
                flags=keys[c['view'],c['id']].get('flags',[False]*len(tools))
                new,inverse=wrapper(tools,c['aliases'],flags,'Length-64' if mode=='Global-Length-64' else mode)
                h=fingerprint(dict(messages=msg(pre+dump(new)+post)))
                policy.append(dict(id=c['id'],view=c['view'],prompt=h,inverse=inverse))
            metrics+=summarise(model,mode,policy,src,keys)
    for way,models in MOUNTS.items():
        for model in models:
            job=model+'__RelGate-'+way;src=cache(model)
            src.update(indexed(rows(DEST/'results'/f'{job}.jsonl')))
            metrics+=summarise(model,'RelGate-'+way,rows(DEST/'cases'/f'{job}.jsonl'),src,keys)
    save_table(DEST/'report/metrics.csv',metrics)
    by={(r['model'],r['mode'],r['view']):r for r in metrics};improvements=[]
    for r in metrics:
        if not r['mode'].startswith('RelGate'):continue
        controls=[(m,'Keep-Name') for m in ('B1','B1-100','B2')]+[(r['model'],'Keep-Name')]
        controls += [('Pure-NameMask',m) for m in ('Always-Mask','Global-Length-64','Oracle')]
        for model,mode in dict.fromkeys(controls):
            b=by.get((model,mode,r['view']))
            if b:
                assert r['n']==b['n']
                improvements.append(dict(model=r['model'],mode=r['mode'],view=r['view'],control_model=model,
                    control_mode=mode,n=r['n'],delta_tool_pp=100*(r['tool_acc']-b['tool_acc']),
                    delta_strict_pp=100*(r['strict_acc']-b['strict_acc']),
                    delta_H_dual_pp=100*(r['H_dual']-b['H_dual']) if b['H_dual'] is not None else None))
    save_table(DEST/'report/improvements.csv',improvements)
    summary=['# RelGate 单基座诊断结果','','所有结果均为冻结探索诊断集；334 条留出集未读取。',
        'H_dual = HM(Coarse-Desc, Name-Swapped)，固定全局长度阈值为 64。',
        '交换标记为合成标签代理；不等同于逐候选人工确认的语义矛盾。','',
        '| Caller | Policy | Clean | Masked | Swapped | Coarse | Degraded | Target | Distractor | H_dual | Mixed HM |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model,mode in dict.fromkeys((r['model'],r['mode']) for r in metrics):
        rs={r['view']:r for r in metrics if (r['model'],r['mode'])==(model,mode)}
        val=lambda x:'—' if x is None else f'{100*x:.2f}'
        one=next(iter(rs.values()))
        summary.append('| '+model+' | '+mode+' | '+' | '.join(val(rs[v]['tool_acc']) if v in rs else '—' for v in VIEWS)+' | '+val(one['H_dual'])+' | '+val(one['mixed_harmonic'])+' |')
    summary+=['','三分类训练 12,000 条，二分类训练 8,000 条；相同 S/C 样本、同基座、同 seed，训练总样本量与优化步数随消融而改变。',
        '内部验证 1,000 条，三类分布为 200/400/400；二分类验证为相同的 200/400 条。报告 per-class 指标。',
        '未进行人工独立语义复核；本轮没有第二基座、多种子或独立最终测试。']
    (DEST/'report/结果汇总.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    print('\n'.join(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gated','score']);a=p.parse_args()
    {'prepare':prepare,'gated':gated,'score':score}[a.action]()
