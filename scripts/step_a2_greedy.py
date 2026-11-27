import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

root = Path('/root/autodl-tmp/Hammer')
run = root / 'research_outputs/step_a2_incremental'
base = root / 'models/Qwen2-1.5B-Instruct'
gate = root / 'research_outputs/relgate_v2_final/RelGate-3Way-v2/adapter'
rows = [json.loads(x) for x in (run/'score_inputs.jsonl').read_text().splitlines()]
tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
prompts = [tok.apply_chat_template([{'role':'user','content':r['instruction']}], tokenize=True, add_generation_prompt=True) for r in rows]
model = LLM(model=str(base), dtype='bfloat16', seed=42, enforce_eager=True, max_model_len=max(map(len,prompts))+32, gpu_memory_utilization=.8, max_num_seqs=16, enable_lora=True, max_lora_rank=32)
params = SamplingParams(temperature=0, max_tokens=16, stop_token_ids=[tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')])
out = run/'greedy_labels.jsonl'
done = {x['id'] for x in (json.loads(s) for s in out.read_text().splitlines())} if out.exists() else set()
for i in range(0,len(rows),128):
    batch = [(r,p) for r,p in zip(rows[i:i+128],prompts[i:i+128]) if r['id'] not in done]
    if not batch: continue
    outputs = model.generate([{'prompt_token_ids':p} for _,p in batch], params, lora_request=LoRARequest('relgatev2',1,str(gate)), use_tqdm=False)
    with out.open('a',encoding='utf8') as f:
        for (row,_),result in zip(batch,outputs,strict=True):
            f.write(json.dumps({'id':row['id'],'raw':result.outputs[0].text.strip()},ensure_ascii=False)+'\n')
    print('greedy', min(i+128,len(rows)), '/',len(rows),flush=True)
assert len(out.read_text().splitlines()) == len(rows)
(run/'greedy.done.json').write_text(json.dumps({'count':len(rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
