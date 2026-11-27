import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'research_outputs/relgate_v2_final/score_inputs.jsonl'
assert not out.exists()
def read(p):
    return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
existing = read(root/'research_outputs/relgate_pareto/score_inputs/3way.jsonl')
evaluation = [r for r in existing if not r['id'].startswith('validation:')]
validation = read(root/'research_outputs/relgate_v2_final/data/validation_3way.jsonl')
assert len(evaluation) == 3733 and len(validation) == 1000
combined = evaluation + [{'id':f'validation:{i:04d}', 'instruction':r['instruction'], 'label':r['output']} for i,r in enumerate(validation)]
assert len({r['id'] for r in combined}) == 4733
content = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined).encode()
out.write_bytes(content)
print(hashlib.sha256(content).hexdigest(), out)
