"""Construct and audit a new, independently frozen UNVERIFIED training arm."""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from build_relgate_dataset import RAW_SHA, NOUNS, canonical, words, family
from routeb_stage1 import permutation, cleaned, eligibility

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'research_outputs/relgate_v1'
OUT = ROOT / 'research_outputs/relgate_v2_final/data'
EVAL = ROOT / 'research_outputs/routeB_step2b_v1/inputs/self_gate.jsonl'

SHORT = ['', 'API endpoint.', 'Utility function.', 'Service interface.', 'Callable operation.']
IO = [
    'Receives input values and returns the corresponding result in a structured response.',
    'Accepts a set of inputs and produces a response containing the resulting output.',
    'Takes input data from a caller and returns output in the expected response format.',
    'The interface accepts supplied input values and provides a result to the caller.',
    'Input values are accepted by this interface and output is returned as a response.',
    'Provides an interface for receiving input data and returning the resulting output.',
    'Returns an output response after receiving the input values supplied by a caller.',
    'Exposes an endpoint that takes input data and responds with a structured result.',
    'A callable interface receives input and returns an output response to the caller.',
    'Given input values, this endpoint provides a response containing the output.',
    'Accepts input through an interface and responds with the corresponding result.',
    'The caller supplies inputs and receives an output through this interface.',
    'This endpoint receives values as input and supplies the result in its response.',
    'Input information enters through the interface; output is returned to the caller.',
    'An interface for input values that provides an output in a structured response.',
    'Caller input is accepted and the corresponding output is returned in a response.',
    'The interface receives input from the calling application and returns a structured response containing the resulting output.',
    'The endpoint accepts input data provided by the caller and responds with the corresponding output in a structured response format.',
    'Input values supplied by an application are received through this interface, and the resulting output is returned in a response.',
    'This callable interface accepts the input supplied by the caller and provides the corresponding output in its response format.',
    'The calling application provides input through the endpoint and receives a structured response containing the resulting output.',
    'This service interface receives the caller input and supplies a response with the corresponding output in a structured format.',
    'The caller sends input values to the interface and receives the corresponding output as part of a structured response.',
    'Input data is accepted by the endpoint and the resulting output is made available to the caller in a structured response.',
    'The application supplies input to this callable interface, which returns a structured response containing the output result.',
    'Through this endpoint, an application can provide input values and receive the corresponding output in a structured response.',
    'The interface accepts the caller input and supplies the resulting output to the application in a structured response format.',
    'Input values enter through the interface, and the caller receives the corresponding output within a structured response.',
    'A calling application supplies input values and receives output results through the interface in a structured response.',
    'The endpoint receives input from the application and returns a structured response containing the corresponding output.',
    'This interface accepts supplied input values from the calling application and provides a structured output response.',
    'The caller provides input information through this endpoint and receives the result as a structured output response.',
]
CONTEXT = [
    'This interface concerns {domain} information within the associated service context.',
    'The endpoint is associated with {domain} data in the surrounding service environment.',
    'An interface in the {domain} domain, with context supplied by the calling application.',
    'Service context: {domain}. The interface provides a response to an application.',
    'A service interface related to {domain} resources and their associated information.',
    'The service offers an interface in the context of {domain} information and resources.',
    'This endpoint belongs to a service context involving {domain} data and resources.',
    'The interface relates to {domain} within the scope of the associated application.',
]
BROAD = [
    'Handles information associated with {domain} in the relevant service context.',
    'Processes information related to {domain} within the associated application.',
    'Interacts with {domain} information in the context of the service interface.',
    'Operates on information associated with {domain} in the service context.',
    'Supports operations for {domain} information within the associated service.',
    'Handles {domain} resources and related information within an application.',
    'Processes {domain} resources in the surrounding service environment.',
    'Interacts with resources associated with {domain} in an application context.',
]
VAL = [
    'Within this service, {domain} information is the subject of a general interface.',
    'A general-purpose interface is available for information concerning {domain}.',
    'For the application, the relevant subject of this endpoint is {domain} data.',
    'The application context encompasses resources connected with {domain}.',
    'Information in the area of {domain} is addressed by this service interface.',
    'The endpoint concerns the broader subject of {domain} and related resources.',
    'In this application, an interface is associated with the topic of {domain}.',
    'The surrounding application context relates to {domain} and its information.',
]


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triple_hash(t):
    return hashlib.sha256(canonical(t).encode()).hexdigest()


def forbidden(t):
    desc = t.get('description') or ''
    return 'using the provided parameters' in desc.lower() or bool(re.search(r'manage\s+.+\s+using', desc, re.I))


def replacement_pools(val_families):
    raw = json.loads((OLD / 'source/xlam_function_calling_60k.json').read_bytes())
    assigned = read(ROOT / 'research_outputs/routeB_stage1_v1/train/assignment.jsonl')
    eligible = {r['id'] for r in assigned if r['eligible']}
    assert len(raw) == 60000 and len(eligible) == 10697
    originals, swaps = {}, {}
    for idx, row in enumerate(raw):
        tools = json.loads(row['tools']); answers = json.loads(row['answers']); sid = f'xlam60k:{idx}'
        for slot, t in enumerate(tools):
            triple = {k:t.get(k) for k in ('name', 'description', 'parameters')}
            key = triple_hash(triple)
            originals.setdefault(key, dict(triple=triple, origin_triple=key, source_id=sid,
                source_slot=slot, family=family(t), label='SUPPORTED', variant='original', triple_sha256=key))
        if sid not in eligible: continue
        assert eligibility(tools, answers)
        names = [t['name'] for t in tools]
        perm, *_ = permutation(tools, names.index(answers[0]['name']), sid)
        for slot, t in enumerate(tools):
            triple = {k:t.get(k) for k in ('name', 'description', 'parameters')}
            triple.update(name=names[perm[slot]], description=cleaned(t.get('description'), t['name']))
            key = triple_hash(triple)
            swaps.setdefault(key, dict(triple=triple, origin_triple=triple_hash(t), source_id=sid,
                source_slot=slot, donor_slot=perm[slot], original_name=t['name'], family=family(t),
                label='CONTRADICTED', variant='hard_swap', triple_sha256=key))
    pools = {(s, l): [] for s in ('train','validation') for l in ('SUPPORTED','CONTRADICTED')}
    for label, data in [('SUPPORTED', originals), ('CONTRADICTED', swaps)]:
        for key, r in data.items():
            if forbidden(r['triple']) or (label == 'CONTRADICTED' and key in originals): continue
            split = 'validation' if r['family'] in val_families else 'train'
            pools[(split,label)].append(r)
    return pools


def main():
    assert not OUT.exists(), f'refusing to overwrite {OUT}'
    assert not set(VAL) & set(SHORT + IO + CONTEXT + BROAD), 'validation template overlap'
    assert digest(OLD / 'source/xlam_function_calling_60k.json') == RAW_SHA
    for line in (OLD / 'data/SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert digest(OLD / 'data' / name) == expected, name
    template = (OLD / 'data/external_gate_prompt_v1.txt').read_text(encoding='utf-8')
    evals = read(EVAL)
    evalkeys = {triple_hash(t) for t in evals}
    evaldescs = {t.get('description') or '' for t in evals if t.get('description')}
    tokenizer = AutoTokenizer.from_pretrained(str(OLD / 'RelGate-3Way/adapter'), local_files_only=True)
    old_rows = {split:read(OLD / f'data/{split}_provenance.jsonl') for split in ('train','validation')}
    val_families = {r['family'] for r in old_rows['validation']}
    pools = replacement_pools(val_families)
    records, stats, replacements = {}, {}, []
    for split in ('train', 'validation'):
        rows = old_rows[split]
        assert len(rows) == (12000 if split == 'train' else 1000)
        existing = {r['triple_sha256'] for r in rows}
        for i, row in enumerate(rows):
            if row['label'] == 'UNVERIFIED' or not forbidden(row['triple']): continue
            pool = sorted(pools[(split,row['label'])], key=lambda r:r['triple_sha256'])
            candidate = next((x for x in pool if x['triple_sha256'] not in existing and x['triple_sha256'] not in evalkeys), None)
            assert candidate is not None, ('no eligible replacement',split,row['label'])
            existing.add(candidate['triple_sha256'])
            replacements.append({'split':split, 'label':row['label'], 'before':row, 'after':candidate})
            rows[i] = candidate
        original = [r for r in rows if r['label'] == 'UNVERIFIED']
        rng = random.Random('relgate-v2/'+split+'/42')
        rng.shuffle(original)
        counts = [600, 1200, 1200, 1000] if split == 'train' else [0, 0, 0, len(original)]
        unverified = []
        used_unverified = set()
        for i, r in enumerate(original):
            if split == 'validation':
                category, templates = 'phrasing_ood', VAL
            elif i < sum(counts[:1]): category, templates = 'short', SHORT
            elif i < sum(counts[:2]): category, templates = 'io', IO
            elif i < sum(counts[:3]): category, templates = 'context', CONTEXT
            else: category, templates = 'broad', BROAD
            nouns = [w for w in words(r['triple']['name']) if w in NOUNS]
            domain = nouns[0] if nouns else 'service'
            t = dict(r['triple'])
            options = list(range(len(templates)))
            random.Random(f'{split}/{i}/42').shuffle(options)
            template_id = None
            for option in options:
                t['description'] = templates[option].format(domain=domain)
                if triple_hash(t) not in used_unverified:
                    template_id = option
                    break
            assert template_id is not None, ('cannot make unique sample',split,i,category,t['name'])
            used_unverified.add(triple_hash(t))
            x = dict(r, triple=t, variant=f'{category}:{template_id}', triple_sha256=triple_hash(t))
            unverified.append(x)
        rows = [r for r in rows if r['label'] != 'UNVERIFIED'] + unverified
        rng.shuffle(rows)
        assert Counter(r['label'] for r in rows) == (Counter(SUPPORTED=4000, CONTRADICTED=4000, UNVERIFIED=4000) if split == 'train' else Counter(SUPPORTED=200, CONTRADICTED=400, UNVERIFIED=400))
        assert not any(r['triple_sha256'] in evalkeys for r in rows), 'evaluation triple overlap'
        assert len({r['triple_sha256'] for r in rows}) == len(rows), 'duplicate training or validation triple'
        assert not any((r['triple']['description'] or '') in evaldescs for r in unverified), 'evaluation description overlap'
        lengths, token_lengths = {}, []
        for r in rows:
            t = r['triple']; desc = t['description'] or ''
            if r['label'] == 'UNVERIFIED' and split == 'train':
                category = r['variant'].split(':')[0]
                low, high = {'short': (0, 25), 'io': (45, 130), 'context': (35, 140), 'broad': (35, 110)}[category]
                assert low <= len(desc) <= high, (category, len(desc), desc)
            assert 'using the provided parameters' not in desc.lower()
            assert not re.search(r'manage\s+.+\s+using', desc, re.I)
            text = template.format(name=t['name'], description=desc, parameters=json.dumps(t['parameters'], ensure_ascii=False, sort_keys=True))
            r['instruction'] = text
            ids = tokenizer.apply_chat_template([{'role':'user','content':text},{'role':'assistant','content':r['label']}], tokenize=True)
            assert len(ids) < 2048, (split, len(ids), t['name'])
            token_lengths.append(len(ids)); lengths.setdefault(r['label'], []).append(len(desc))
        records[split] = rows
        stats[split] = {'counts':dict(Counter(r['label'] for r in rows)), 'unverified_variants':dict(Counter(r['variant'].split(':')[0] for r in unverified)), 'description_lengths':{k:np.percentile(v,[10,50,90,100]).tolist() for k,v in lengths.items()}, 'max_tokens':max(token_lengths), 'unique_triples':len({r['triple_sha256'] for r in rows})}
    assert len(replacements) == 21, len(replacements)
    assert not {r['family'] for r in records['train']} & {r['family'] for r in records['validation']}
    assert not {r['triple_sha256'] for r in records['train']} & {r['triple_sha256'] for r in records['validation']}
    OUT.mkdir(parents=True)
    for split, rows in records.items():
        (OUT / f'{split}_3way.jsonl').write_text(''.join(json.dumps({'instruction':r['instruction'],'input':'','output':r['label']},ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        (OUT / f'{split}_provenance.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='instruction'},ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    (OUT / 'replacements.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in replacements),encoding='utf-8')
    (OUT / 'audit.json').write_text(json.dumps({'source_sha256':RAW_SHA,'source_v1_manifest_sha256':digest(OLD/'data/manifest.json'),'eval_input_sha256':digest(EVAL),'stats':stats,'label_note':'Weak synthetic labels; no semantic verification. Validation splits original evidence families, donor names may overlap.'},indent=2),encoding='utf-8')
    (OUT / 'SHA256SUMS.txt').write_text(''.join(digest(f)+'  '+f.name+'\n' for f in sorted(OUT.iterdir())),encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__ == '__main__': main()
