"""Freeze P0-A2 caller prompts from P0-A actions and the Step B Qwen cache."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_routeB_swap_probe import split_prompt
from routeb_stage1 import dump, msg
from routeb_step2a import fingerprint


ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / 'research_outputs/step_b_plugplay'
P0 = ROOT / 'research_outputs/p0_nli'
OUT = ROOT / 'research_outputs/p0_a2'
MODEL = 'Qwen2.5-7B-Instruct'
VIEWS = {'Clean': 197, 'Coarse-Desc': 197, 'Name-Swapped': 197,
         'Target-Swapped': 182, 'Distractor-Swapped': 182,
         'Verbose-Coarse': 197, 'Verb-Flip-Target': 72}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, records):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in records), encoding='utf-8')


def verify_manifest(folder):
    for line in (folder / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(folder / name) == digest, name


def main():
    verify_manifest(P0)
    verify_manifest(STEP)
    report = json.loads((STEP / 'raw' / MODEL / 'full.report.json').read_bytes())
    assert report['backend'] == 'vllm' and report['dtype'] == 'bfloat16'
    assert report['generation'] == {'do_sample': False, 'temperature': 0.0, 'max_new_tokens': 256}
    assert report['max_num_seqs'] == 16 and report['count'] == 4125
    assert report['input_sha256'] == sha(STEP / 'unique_prompts.jsonl')
    assert report['schedule_sha256'] == sha(STEP / f'{MODEL}.schedule.jsonl')
    raw_path = STEP / 'raw' / MODEL / 'predictions.jsonl'
    assert report['output_sha256'] == sha(raw_path)
    old_predictions = read(raw_path)
    assert len(old_predictions) == len({r['id'] for r in old_predictions}) == 4125
    cache = {r['id']: r for r in old_predictions}
    old_prompts = {r['id']: r for r in read(STEP / 'unique_prompts.jsonl')}
    assert set(cache) == set(old_prompts)
    old_cells = {(r['view'], r['id'], r['policy']): r for r in read(STEP / f'{MODEL}.schedule.jsonl')}
    actions = read(P0 / 'request_actions.jsonl')
    assert len(actions) == 1224 * 3
    assert Counter(a['policy'] for a in actions) == {'RelGate-0.50': 1224, 'NLI-0.50': 1224, 'NLI-Matched': 1224}
    assert all(Counter(a['view'] for a in actions if a['policy'] == p) == VIEWS for p in ('RelGate-0.50', 'NLI-0.50', 'NLI-Matched'))
    prompts, schedule = {}, []
    parity = 0
    for action in actions:
        view, sid, policy = action['view'], action['request_id'], action['policy']
        original = old_cells[view, sid, 'Keep-Name']
        before, tools, after = split_prompt(old_prompts[original['prompt']])
        names = [t['name'] for t in tools]
        assert len(names) == action['candidate_count'] == len(action['visible_names'])
        assert [action['inverse'].get(n, n) for n in action['visible_names']] == names
        assert len(action['inverse']) == action['masked_count']
        replaced = [dict(t, name=name) for t, name in zip(tools, action['visible_names'], strict=True)]
        messages = msg(before + dump(replaced) + after) if action['masked_count'] else msg(before + dump(tools) + after)
        prompt_id = fingerprint({'messages': messages})
        prior = prompts.setdefault(prompt_id, {'id': prompt_id, 'messages': messages})
        assert prior['messages'] == messages
        cell = {'id': sid, 'view': view, 'policy': policy, 'prompt': prompt_id,
                'inverse': action['inverse'], 'masked_count': action['masked_count'],
                'candidate_count': action['candidate_count']}
        if policy == 'RelGate-0.50':
            frozen = old_cells[view, sid, policy]
            assert cell == frozen, (view, sid)
            parity += 1
        schedule.append(cell)
    assert parity == 1224
    new_ids = set(prompts) - set(cache)
    assert len(new_ids) <= 1201, f'New prompts exceed approved budget: {len(new_ids)}'
    prior_policies = ('Keep-Name', 'Always-Mask', 'Length-64', 'RelGate-0.50')
    combined = [r for r in old_cells.values() if r['view'] in VIEWS and r['policy'] in prior_policies] + [
        r for r in schedule if r['policy'] != 'RelGate-0.50']
    assert len(combined) == 1224 * 6
    assert len({(r['view'], r['id'], r['policy']) for r in combined}) == len(combined)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'unique_prompts.jsonl', prompts.values())
    write(OUT / 'new_prompts.jsonl', (prompts[k] for k in sorted(new_ids)))
    write(OUT / 'schedule.jsonl', combined)
    metadata = {'model': MODEL, 'historical_backend': report['backend'],
                'historical_dtype': report['dtype'], 'historical_generation': report['generation'],
                'historical_max_num_seqs': report['max_num_seqs'],
                'historical_chat_template_sha256': report['chat_template_sha256'],
                'historical_model_path': report['model_path'],
                'historical_resolved_commit': report['resolved_commit'],
                'historical_report_sha256': sha(STEP / 'raw' / MODEL / 'full.report.json'),
                'historical_predictions_sha256': sha(raw_path),
                'historical_prompt_pool_sha256': sha(STEP / 'unique_prompts.jsonl'),
                'p0_actions_sha256': sha(P0 / 'request_actions.jsonl'),
                'relgate_exact_parity_cells': parity,
                'total_policy_cells': len(combined), 'p0_cells': len(schedule),
                'p0_unique_prompts': len(prompts), 'cache_hits': len(prompts) - len(new_ids),
                'new_unique_prompts': len(new_ids), 'per_policy_new_unique': {
                    p: len({r['prompt'] for r in schedule if r['policy'] == p} - set(cache))
                    for p in ('NLI-0.50', 'NLI-Matched')},
                'no_confirmation_set_access': True}
    (OUT / 'preflight.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for name in ('unique_prompts.jsonl', 'new_prompts.jsonl', 'schedule.jsonl', 'preflight.json'):
        assert (OUT / name).exists()
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{sha(OUT / name)}  {name}\n' for name in
        ('unique_prompts.jsonl', 'new_prompts.jsonl', 'schedule.jsonl', 'preflight.json')), encoding='ascii')
    verify_manifest(OUT)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
