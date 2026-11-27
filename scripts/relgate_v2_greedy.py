import json
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

root = Path('/root/autodl-tmp/Hammer')
run = root / 'research_outputs/relgate_v2_final'
base = root / 'models/Qwen2-1.5B-Instruct'
rows = [json.loads(line) for line in (run / 'score_inputs.jsonl').read_text().splitlines()]
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['instruction']}], tokenize=True, add_generation_prompt=True) for r in rows]
model = LLM(model=str(base), dtype='bfloat16', seed=42, enforce_eager=True, max_model_len=max(map(len, prompts))+32, gpu_memory_utilization=.8, max_num_seqs=16, enable_lora=True, max_lora_rank=32)
params = SamplingParams(temperature=0, max_tokens=16, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')])
out = run / 'greedy_labels.jsonl'
assert not out.exists()
for i in range(0, len(rows), 128):
    outputs = model.generate([{'prompt_token_ids': p} for p in prompts[i:i+128]], params, lora_request=LoRARequest('relgatev2',1,str(run/'RelGate-3Way-v2'/'adapter')), use_tqdm=False)
    with out.open('a', encoding='utf8') as f:
        for row, result in zip(rows[i:i+128], outputs, strict=True):
            f.write(json.dumps({'id':row['id'],'raw':result.outputs[0].text.strip()}, ensure_ascii=False)+'\n')
    print('greedy', min(i+128,len(rows)), '/', len(rows), flush=True)
assert sum(1 for _ in out.open(encoding='utf8')) == len(rows)
(run/'greedy.done').write_text('complete\n')
