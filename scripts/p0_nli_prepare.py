"""Adapt only frozen exploration prompts and actions; never access confirmation data."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_routeB_swap_probe import split_prompt
from p0_nli_compare import VIEWS, canon, key, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_outputs/p0_nli'
STEP_B = ROOT / 'research_outputs/step_b_plugplay'
REVISION = 'bab4bc7178836f731dcfd18c06ca9def0a137712'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def freeze_model(folder):
    files = {}
    folder = Path(folder).resolve()
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        metadata = folder / '.cache/huggingface/download' / (path.name+'.metadata')
        commit, etag, _ = metadata.read_text().splitlines()
        assert commit == REVISION, path.name
        h = hashlib.sha256()
        git = hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4*1024*1024), b''):
                h.update(block)
                git.update(block)
        assert (h.hexdigest() if len(etag) == 64 else git.hexdigest()) == etag, path.name
        files[path.name] = dict(sha256=h.hexdigest(), huggingface_etag=etag, bytes=path.stat().st_size)
    assert {'model.safetensors', 'config.json', 'tokenizer.json'} <= files.keys()
    provenance = dict(repository='cross-encoder/nli-deberta-v3-large', revision=REVISION,
                      local_snapshot=str(folder), files=files, all_download_metadata_commits_match=True,
                      all_download_etags_verified=True)
    (OUT / 'model_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('MODEL_SNAPSHOT_REVISION_AND_ETAGS_VERIFIED', REVISION, len(files), flush=True)
    return provenance


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def load(relative, expected=None):
        path = ROOT / relative
        actual = sha(path)
        if expected:
            assert actual == expected, relative
        inputs[relative] = actual
        return read(path)

    manifest = {}
    for line in (STEP_B / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        manifest[name] = digest
    pool = {r['id']: r for r in load('research_outputs/step_b_plugplay/unique_prompts.jsonl', manifest['unique_prompts.jsonl'])}
    schedule = load('research_outputs/step_b_plugplay/Qwen2.5-7B-Instruct.schedule.jsonl', manifest['Qwen2.5-7B-Instruct.schedule.jsonl'])
    cells = {(r['view'], r['id'], r['policy']): r for r in schedule if r['view'] in VIEWS}
    clean = {sid: split_prompt(pool[c['prompt']])[1]
             for (view, sid, policy), c in cells.items() if view == 'Clean' and policy == 'Keep-Name'}
    labels = {(r['view'], r['id']): r
              for r in load('research_outputs/routeB_stage1_v1/eval_keys.jsonl')
              if r['mode'] == 'Keep-Name' and r['id'] in clean and r['view'] in VIEWS}
    probes = {(r['view'], r['id']): r for r in load('research_outputs/step_a_v1/probes.jsonl',
              'b7fa9b653252ae3db211837016fde002ca07c05acf747d10022c0bb485ac1ed6')}
    frozen = {}
    for relative, expected in json.loads((STEP_B / 'b1_validation.json').read_bytes())['sources_sha256'].items():
        relative = relative.replace('\\', '/')
        for r in load(relative, expected):
            frozen[r['view'], r['id'], r['policy']] = r
    rows = []
    action_checks = 0
    for (view, sid, policy), cell in cells.items():
        if policy != 'Keep-Name':
            continue
        tools = split_prompt(pool[cell['prompt']])[1]
        action = cells[view, sid, 'RelGate-0.50']
        old = frozen[view, sid, 'RelGate-0.50']
        for field in ('prompt', 'inverse', 'masked_count', 'candidate_count'):
            assert old[field] == action[field], (view, sid, field)
        masked = split_prompt(pool[action['prompt']])[1]
        assert len(tools) == len(masked) == len(clean[sid]) == action['candidate_count']
        flags = [t['name'] != clean[sid][i]['name'] for i, t in enumerate(tools)]
        if (view, sid) in probes:
            label = probes[view, sid]
            assert tools == label['tools']
            g = label['g']
            if view == 'Verb-Flip-Target':
                assert flags == [i == g for i in range(len(tools))]
            else:
                assert not any(flags)
        else:
            label = labels[view, sid]
            g = label['g']
            assert flags == label['flags']
        masks = []
        for i, (t, m) in enumerate(zip(tools, masked)):
            selected = t['name'] != m['name']
            assert {k: v for k, v in t.items() if k != 'name'} == {k: v for k, v in m.items() if k != 'name'}
            assert action['inverse'].get(m['name'], m['name']) == t['name']
            if selected:
                assert action['inverse'][m['name']] == t['name']
            assert t['parameters'] == clean[sid][i]['parameters']
            rows.append(dict(request_id=sid, view=view, candidate_index=i,
                             name=t['name'], description=t['description'], parameters=t['parameters'],
                             changed=flags[i], is_target=i == g, relgate_mask=selected))
            masks.append(selected)
            action_checks += 1
        assert sum(masks) == action['masked_count'] == len(action['inverse'])
    counts = validate(rows)
    assert counts == VIEWS
    path = OUT / 'exploration_candidates.jsonl'
    payload = ''.join(canon(r)+'\n' for r in rows).encode('utf-8')
    if path.exists():
        assert path.read_bytes() == payload, 'Refusing to replace different frozen input'
    else:
        path.write_bytes(payload)
    audit = dict(model='cross-encoder/nli-deberta-v3-large', revision=REVISION,
                 view_counts=counts, request_view_cells=sum(counts.values()),
                 candidate_occurrences=len(rows), unique_triples=len({key(r) for r in rows}),
                 per_view_candidates=dict(Counter(r['view'] for r in rows)),
                 exact_frozen_action_checks=action_checks,
                 input_sha256=sha(path), source_sha256=inputs,
                 changed='name differs from frozen Clean at same candidate slot; construction proxy, not human semantic labels',
                 is_target='frozen exploration gold candidate slot; statistics only',
                 no_confirmation_data_access=True, no_new_gate_scoring=True,
                 no_template_or_threshold_changes=True)
    (OUT / 'adaptation_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
