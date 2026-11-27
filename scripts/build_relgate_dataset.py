"""Freeze metadata-only weak-supervision data from checksum-pinned xlam60k.

No evaluation or held-out path is opened by this module. Validation groups are
original tool-name families. A swapped donor name may occur across partitions;
the original evidence family, not each spelling token, defines the split.
"""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from copy import deepcopy

from routeb_stage1 import permutation, cleaned, eligibility, STOP, words
from relgate_cpu_audit import DEST, R, PROMPT_SHA, write, jsonl

RAW_SHA='4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6'
PLACEHOLDERS=['API endpoint.','Utility function.','Service interface.','Callable operation.',
              'Application API.','General utility.','Endpoint interface.','Function entry.']
NOUNS=set('account address airport album animal appointment article artist asset author balance bank battery bill book booking budget building business calendar camera car card category channel city client cloud code collection color comment company contact content country course currency customer database date device directory document domain driver email employee event exchange expense file flight folder food forecast form game group health history hotel house image index invoice ip item job language latitude library location longitude machine market media member message metadata metric model money movie music network note number object order organization package page parameter password path payment person phone photo place plan playlist point policy portfolio price product profile project property query quote rate rating receipt recipe record region report repository reservation resource result review role route row schedule schema school search security sensor server service session setting shipment shop site song source station status stock storage store street student subscription system table tag task team temperature tenant ticket time token topic transaction trip type url user value vehicle version video wallet weather website window workflow year zone'.split())


def canonical(x):
    return json.dumps({k:x.get(k) for k in ('name','description','parameters')},ensure_ascii=False,sort_keys=True,separators=(',',':'))


def key(x):return hashlib.sha256(canonical(x).encode()).hexdigest()


def family(t):return t['name'].casefold()


def unspecified(t,kind):
    if kind=='empty':return ''
    if kind=='nouns':
        tokens=words(t.get('description'))+words(t['name'])
        nouns=list(dict.fromkeys(w for w in tokens if w in NOUNS and w not in STOP))[:3]
        return ' '.join(nouns) if nouns else ''
    return PLACEHOLDERS[int(kind.split('-')[1])]


def build():
    target=DEST/'data'
    if target.exists():raise FileExistsError('Frozen data already exists; do not overwrite it')
    raw=DEST/'source/xlam_function_calling_60k.json'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==RAW_SHA
    source=json.loads(raw.read_bytes());assert len(source)==60000
    assignment=R.parent/'routeB_stage1_v1/train/assignment.jsonl'
    assigned=[json.loads(s) for s in assignment.read_bytes().splitlines()]
    eligible_ids={r['id'] for r in assigned if r['eligible']}
    assert len(eligible_ids)==10697 and all(s.startswith('xlam60k:') for s in eligible_ids)
    originals={};proposals={}
    for index,row in enumerate(source):
        tools=json.loads(row['tools']);answers=json.loads(row['answers']);sid=f'xlam60k:{index}'
        for slot,t in enumerate(tools):
            k=key(t)
            originals.setdefault(k,dict(triple={x:t.get(x) for x in ('name','description','parameters')},
                origin_triple=k,source_id=sid,source_slot=slot,family=family(t)))
        if sid not in eligible_ids:continue
        assert eligibility(tools,answers)
        names=[t['name'] for t in tools]
        perm,*_=permutation(tools,names.index(answers[0]['name']),sid)
        for slot,t in enumerate(tools):
            z={x:t.get(x) for x in ('name','description','parameters')}
            z.update(name=names[perm[slot]],description=cleaned(t.get('description'),t['name']))
            assert z['name']!=t['name'] and z['parameters']==t['parameters']
            if key(z) in originals:continue
            proposals.setdefault(key(z),dict(triple=z,origin_triple=key(t),source_id=sid,source_slot=slot,
                donor_slot=perm[slot],original_name=t['name'],family=family(t),variant='hard_swap'))
    # Collision filtering is repeated after the full original pool is known.
    proposals={k:v for k,v in proposals.items() if k not in originals}
    groups={}
    for k,r in originals.items():groups.setdefault(r['family'],[]).append(k)
    order=sorted(groups);random.Random(42).shuffle(order)
    val_groups=set();nval=0
    # Preserve exactly 200 original full-evidence triples for internal validation.
    for g in order:
        if nval+len(groups[g])<=200:
            val_groups.add(g);nval+=len(groups[g])
        if nval==200:break
    assert nval==200
    pools={s:{l:[] for l in ('SUPPORTED','CONTRADICTED','UNVERIFIED')} for s in ('train','validation')}
    for r in originals.values():
        split='validation' if r['family'] in val_groups else 'train'
        pools[split]['SUPPORTED'].append(dict(r,label='SUPPORTED',variant='original'))
    for r in proposals.values():
        split='validation' if r['family'] in val_groups else 'train'
        pools[split]['CONTRADICTED'].append(dict(r,label='CONTRADICTED'))
    for split in pools:
        originals_in_split=pools[split]['SUPPORTED']
        seen=set();u=[]
        kinds=['empty','nouns']+[f'placeholder-{i}' for i in range(len(PLACEHOLDERS))]
        for r in originals_in_split:
            # Permute transformation order per tool; no uniform class-specific sentence.
            order=kinds.copy();random.Random('relgate/u/42/'+r['origin_triple']).shuffle(order)
            for kind in order:
                z=dict(r['triple'],description=unspecified(r['triple'],kind))
                k=key(z)
                if k not in seen and k not in originals and k not in proposals:
                    seen.add(k);u.append(dict(r,triple=z,label='UNVERIFIED',variant=kind))
        pools[split]['UNVERIFIED']=u
    selected={}
    targets={'train':dict(SUPPORTED=4000,CONTRADICTED=4000,UNVERIFIED=4000),
             'validation':dict(SUPPORTED=200,CONTRADICTED=400,UNVERIFIED=400)}
    for split,classes in pools.items():
        selected[split]=[]
        for label,pool in classes.items():
            pool=sorted(pool,key=lambda r:key(r['triple']))
            random.Random('relgate/42/'+split+'/'+label).shuffle(pool)
            n=targets[split][label]
            assert len(pool)>=n,(split,label,len(pool),n)
            selected[split].extend(pool[:n])
        random.Random('relgate/order/42/'+split).shuffle(selected[split])
    template=(R/'external_gate_prompt_v1.txt').read_bytes()
    assert hashlib.sha256(template).hexdigest()==PROMPT_SHA
    data={}
    for split,records in selected.items():
        data[split]=[]
        for r in records:
            t=r['triple']
            text=template.decode().format(name=t['name'],description=t['description'] or '',
                  parameters=json.dumps(t['parameters'],ensure_ascii=False,sort_keys=True))
            data[split].append(dict(instruction=text,input='',output=r['label']))
    trainkeys={key(r['triple']) for r in selected['train']}
    valkeys={key(r['triple']) for r in selected['validation']}
    assert not trainkeys&valkeys
    assert not {r['family'] for r in selected['train']}&{r['family'] for r in selected['validation']}
    assert all(r['triple']['parameters']==originals[r['origin_triple']]['triple']['parameters'] for rs in selected.values() for r in rs)
    assert all(not re.search(r'(?<!\w)'+re.escape(r['original_name'])+r'(?!\w)',r['triple']['description'] or '',re.I)
               for rs in selected.values() for r in rs if r['label']=='CONTRADICTED')
    target.mkdir(parents=True)
    for split in selected:
        jsonl(target/f'{split}_3way.jsonl',data[split])
        jsonl(target/f'{split}_2way.jsonl',(d for d,r in zip(data[split],selected[split],strict=True) if r['label']!='UNVERIFIED'))
        jsonl(target/f'{split}_provenance.jsonl',(dict(r,triple_sha256=key(r['triple'])) for r in selected[split]))
    (target/'external_gate_prompt_v1.txt').write_bytes(template)
    write(target/'dataset_info.json',{f'{s}_{n}way':{'file_name':f'{s}_{n}way.jsonl'} for s in selected for n in (2,3)})
    manifest=dict(seed=42,raw_path=str(raw),raw_sha256=RAW_SHA,source='Salesforce/xlam-function-calling-60k, full 60000-row original file',
        source_rows=60000,original_unique_triples=len(originals),original_name_families=len(groups),
        contradiction_source='Only the frozen 10697 eligible xlam60k request IDs; same deterministic hard-swap permutation',
        eligible_assignment_sha256=hashlib.sha256(assignment.read_bytes()).hexdigest(),
        pool_sizes={s:{l:len(p) for l,p in c.items()} for s,c in pools.items()},
        class_counts={s:dict(Counter(r['label'] for r in rs)) for s,rs in selected.items()},
        original_family_disjoint=True,exact_metadata_disjoint=True,validation_balanced=False,
        validation_reason='Only 4211 deduplicated originals exist. Reserve 200 original-evidence triples by whole name family, with 400 swapped and 400 degraded examples; retain 4000 distinct SUPPORTED training examples. Report per-class validation metrics.',
        donor_name_tokens_may_overlap=True,
        unverified_methods=['empty description','up to three domain-noun lexicon tokens','eight heterogeneous minimal placeholders'],
        weak_label_limitations='Alignment/swap/degradation are synthetic relation proxies. A swap need not be semantically contradictory, and parameters alone may support some degraded cases. No human semantic certification was performed.',
        template_sha256=PROMPT_SHA,eval_inputs_opened_by_builder=[],heldout_inputs_opened=[],
        train_source_split='xlam60k internal split; evaluation-source OOD is a provenance statement, not proof of zero semantic overlap')
    write(target/'manifest.json',manifest)
    files=sorted(p for p in target.iterdir() if p.is_file())
    (target/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':build()
