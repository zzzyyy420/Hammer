"""Read-only audit of frozen Route B evidence; write only to a new RelGate run."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from routeb_validation import (R, VIEWS, EXCLUDED, BASELINES, existing, gold,
                               view_rows, tid)
from routeb_step2a import fingerprint, save_table, outcome
from routeb_stage1 import OUT, rows, indexed, split_prompt, wrapper, msg, dump, remap

DEST = R.parent / 'relgate_v1'
LABELS = {'SUPPORTED', 'UNVERIFIED', 'CONTRADICTED'}
PROMPT_SHA = '8b6238ca15111953a995cb6835b8073166aee216d0ebbf2c453d6afc819acdfa'
GLOBAL_TAU = 64  # Frozen before inspecting the resulting scores; no view-specific tuning.


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(''.join(dump(x)+'\n' for x in data).encode('utf-8'))


def hm(a,b):
    return 2*a*b/(a+b) if a+b else 0.


def audit_gate():
    prompt=(R/'external_gate_prompt_v1.txt').read_bytes()
    assert hashlib.sha256(prompt).hexdigest()==PROMPT_SHA
    triples=indexed(rows(R/'inputs/self_gate.jsonl'))
    predictions=indexed(rows(R/'results/self_gate.jsonl'))
    assert triples.keys()==predictions.keys() and len(triples)==3733
    occurrences=rows(R/'report/gate_candidate_occurrences.jsonl')
    candidates={r['triple']:r for r in occurrences if r['view']=='Name-Swapped' and r['swapped'] is True}
    sample=random.Random(42).sample(sorted(candidates),10)
    audit=[]
    for key in sample:
        t=triples[key]; raw=predictions[key]['raw']
        assert set(t)=={'id','name','description','parameters'} and tid(t)[0]==key
        c=candidates[key]
        text=prompt.decode('utf-8').format(name=t['name'],description=t['description'] or '',
                 parameters=json.dumps(t['parameters'],ensure_ascii=False,sort_keys=True))
        audit.append(dict(triple=key,request_id=c['id'],slot=c['slot'],
            gate_messages=[dict(role='user',content=text)],raw_output=raw,
            parsed_label=raw.strip() if raw.strip() in LABELS else None,
            parse_failure=raw.strip() not in LABELS,finish_reason=predictions[key]['finish_reason'],
            input_metadata=t,oracle_fields_in_gate_input=False))
    jsonl(DEST/'audit/self_gate_10_complete.jsonl',audit)
    write(DEST/'audit/self_gate_audit.json',dict(seed=42,sampled_unique_triples=len(sample),
        population_unique_swapped_triples=len(candidates),prompt_sha256=PROMPT_SHA,
        label_counts=Counter(p['raw'].strip() for p in predictions.values()),
        sampled_labels=Counter(a['parsed_label'] for a in audit),
        observations='ID/hash joins, exact prompt formatting and strict parsing agree with the frozen implementation. All ten sampled raw responses are archived.',
        inference='No parser-discarded CONTRADICTED label is observed. This does not prove a psychological confirmation-bias mechanism or establish that every synthetic swap is a semantic contradiction.',
        human_semantic_review='not performed'))


def exclusions():
    original=indexed(rows(OUT/'eval_views/Clean.jsonl'))
    # This is an explicit historical ID list, not a newly inferred keyword filter.
    reasons={
        '62aa178d54131a66':('executeSQLInjection','historical targeted attack-call exclusion'),
        '3cf7103a0acae8ca':('executeEthicalHack','historical targeted attack-call exclusion'),
        '71634fd867f412ec':('exploitVulnerability','historical exploit-candidate exclusion'),
    }
    records=[]
    for sid in sorted(EXCLUDED):
        tools=split_prompt(original[sid])[1]
        records.append(dict(id=sid,exact_predicate='request.id in EXCLUDED (explicit frozen three-ID set; applied to every model and policy)',
            historical_reason=reasons[sid][1],relevant_candidate_name=reasons[sid][0],
            candidate_names=[t['name'] for t in tools],
            original_files_modified=False,
            predicate_source='scripts/routeb_validation.py:EXCLUDED and view_rows; research_outputs/routeB_step2b_v1/evaluation_protocol.md'))
    write(DEST/'audit/exclusion_manifest.json',dict(exclusions=records,
        denominators={v:len(view_rows(v)) for v in VIEWS},
        note='No recoverable finer original textual decision rule exists in the frozen implementation; the executable rule is exactly ID membership, not a general keyword classifier.'))


def fixed_length(emit_inputs=False):
    keys=gold(); cases=rows(R/'cases.jsonl')
    originals=indexed(rows(R/'inputs/keep.jsonl'))
    assert Counter(c['view'] for c in cases)==Counter({v:len(view_rows(v)) for v in VIEWS})
    derived=[];requests={}
    for c in cases:
        original=originals[c['prompt']]
        prefix,tools,suffix=split_prompt(original)
        new,inverse=wrapper(tools,c['aliases'],[False]*len(tools),f'Length-{GLOBAL_TAU}')
        row=dict(messages=msg(prefix+dump(new)+suffix))
        h=fingerprint(row)
        requests[h]=dict(id=h,**row)
        derived.append(dict(id=c['id'],view=c['view'],prompt=h,inverse=inverse))
    jsonl(DEST/'audit/fixed_length_cases.jsonl',derived)
    jsonl(DEST/'audit/fixed_length_inputs.jsonl',requests.values())
    metrics=[];missing=[];predrows=[]
    for model in BASELINES:
        cache=existing(model)
        extra=DEST/'results'/f'{model}__Length-64.jsonl'
        if extra.exists():cache.update(indexed(rows(extra)))
        needed={c['prompt'] for c in derived if c['prompt'] not in cache}
        if emit_inputs:
            jsonl(DEST/'inputs'/f'{model}__Length-64.jsonl',(requests[h] for h in sorted(needed)))
        for view in VIEWS:
            cc=[c for c in derived if c['view']==view]
            scores=[]
            for c in cc:
                if c['prompt'] not in cache:continue
                p=cache[c['prompt']]
                result=outcome(dict(p,parsed_pred=remap(p['parsed_pred'],c['inverse'])),keys[view,c['id']])
                scores.append(result)
                predrows.append(dict(model=model,view=view,id=c['id'],tool_correct=result[0],strict_correct=result[1],
                    name_hijack=result[2] if view in ('Name-Swapped','Target-Swapped') else None,
                    invalid=result[3],prompt=c['prompt']))
            complete=len(scores)==len(cc)
            a,b,h,j=map(sum,zip(*scores)) if scores else (0,0,0,0)
            metrics.append(dict(model=model,mode='Global-Length-64',view=view,n=len(cc),
                scored=len(scores),missing=len(cc)-len(scores),correct=a if complete else None,
                tool_acc=a/len(cc) if complete else None,strict_acc=b/len(cc) if complete else None,
                invalid=j if complete else None,H_dual=None,length_tau=GLOBAL_TAU))
        missing.append(dict(model=model,missing_unique_prompts=len(needed)))
    for m in BASELINES:
        cells={r['view']:r for r in metrics if r['model']==m}
        a,b=[cells[v]['tool_acc'] for v in ('Coarse-Desc','Name-Swapped')]
        h=None if a is None or b is None else hm(a,b)
        for r in cells.values():r['H_dual']=h
    save_table(DEST/'audit/global_length_metrics.csv',metrics)
    jsonl(DEST/'audit/global_length_predictions.jsonl',predrows)
    write(DEST/'audit/global_length_policy.json',dict(tau=GLOBAL_TAU,unit='Python len of function description in Unicode codepoints',
        action='mask candidate iff len(description or empty) > 64',
        selection='Fixed central value of the existing {32,64,128} grid before this run scored it. Same threshold across all models and all seven views; no view-specific optimization.',
        reuse='Exact rendered-message SHA-256 cache only. Missing cells require supplement; never substitute a partial-denominator accuracy.',missing=missing))
    print('fixed-global length cache:',missing,flush=True)


def main():
    (DEST/'audit').mkdir(exist_ok=True,parents=True)
    audit_gate();exclusions();fixed_length()
    print('CPU evidence audits written to',DEST/'audit',flush=True)


if __name__=='__main__':main()
