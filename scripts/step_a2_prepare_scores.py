import hashlib
import json
from pathlib import Path

from routeb_stage1 import rows
from routeb_validation import tid

root = Path(__file__).resolve().parents[1]
out = root / 'research_outputs/step_a2_incremental'
out.mkdir(parents=True, exist_ok=True)
probe = root / 'research_outputs/step_a_v1/probes.jsonl'
assert hashlib.sha256(probe.read_bytes()).hexdigest() == 'b7fa9b653252ae3db211837016fde002ca07c05acf747d10022c0bb485ac1ed6'
template = (root / 'research_outputs/relgate_v1/data/external_gate_prompt_v1.txt').read_text(encoding='utf-8').strip()
existing = {r['id'] for r in rows(root / 'research_outputs/relgate_v2_final/scores/3way.jsonl')}
labels = {r['id'] for r in rows(root / 'research_outputs/relgate_v2_final/greedy_labels.jsonl')}
assert existing == labels
triples = {}
for row in rows(probe):
    for tool in row['tools']:
        key, fields = tid(tool)
        triples[key] = fields
records = []
for key, fields in sorted(triples.items()):
    if key not in existing:
        instruction = template.format(name=fields['name'], description=fields['description'] or '', parameters=json.dumps(fields['parameters'], ensure_ascii=False, sort_keys=True))
        records.append({'id': key, 'instruction': instruction})
(out / 'score_inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
print(json.dumps({'triples': len(triples), 'reused': len(triples)-len(records), 'new': len(records)}))
