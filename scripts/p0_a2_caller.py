"""Run only frozen P0-A2 cache misses under the exact Step B Qwen configuration."""
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from step_b_caller_runner import clean_json, gpu_monitor, STOP, sha, rows


ROOT = Path('/root/autodl-tmp/Hammer')
OUT = ROOT / 'research_outputs/p0_a2'
MODEL = ROOT / 'models/Qwen2.5-7B-Instruct'


def main():
    from vllm import LLM, SamplingParams
    for line in (OUT / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(OUT / name) == digest, name
    historical = json.loads((OUT / 'step_b_full.report.json').read_bytes())
    preflight = json.loads((OUT / 'preflight.json').read_bytes())
    assert historical['input_sha256'] == preflight['historical_prompt_pool_sha256']
    assert historical['output_sha256'] == preflight['historical_predictions_sha256']
    assert historical['backend'] == 'vllm' and historical['dtype'] == 'bfloat16'
    assert historical['generation'] == {'do_sample': False, 'temperature': 0.0, 'max_new_tokens': 256}
    assert historical['max_model_len'] == 3880 and historical['max_num_seqs'] == 16
    assert sha(OUT / 'step_b_full.report.json') == preflight['historical_report_sha256']
    provenance = json.loads((OUT / 'step_b_model_provenance.json').read_bytes())['Qwen2.5-7B-Instruct']
    assert provenance['caller_chat_template_sha256'] == historical['chat_template_sha256']
    prompts = rows(OUT / 'new_prompts.jsonl')
    assert len(prompts) == preflight['new_unique_prompts'] == 896
    assert len({p['id'] for p in prompts}) == len(prompts)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == historical['chat_template_sha256']
    tokenized = {r['id']: tokenizer.apply_chat_template(r['messages'], tokenize=True,
                 add_generation_prompt=True) for r in prompts}
    max_input = max(map(len, tokenized.values()))
    assert max_input + 256 <= 3880, max_input
    if '--preflight-only' in sys.argv:
        print('P0_A2_TOKEN_PREFLIGHT_OK', len(prompts), max_input,
              hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(), flush=True)
        return
    assert all((MODEL / x['path']).is_file() for x in provenance['weight_files'])
    dest = OUT / 'predictions.new.jsonl'
    old = rows(dest) if dest.exists() else []
    done = {r['id']: r for r in old}
    assert len(done) == len(old) and set(done) <= set(tokenized)
    todo = [r['id'] for r in prompts if r['id'] not in done]
    monitor_result = {}
    monitor = threading.Thread(target=lambda: monitor_result.setdefault('mb', gpu_monitor()), daemon=True)
    monitor.start()
    started = time.perf_counter()
    llm = LLM(model=str(MODEL), dtype='bfloat16', seed=42, enforce_eager=True,
              max_model_len=3880, gpu_memory_utilization=.88, max_num_seqs=16)
    terminators = {tokenizer.eos_token_id}
    for special in ('<|im_end|>', '<|eot_id|>'):
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            terminators.add(tid)
    params = SamplingParams(temperature=0.0, max_tokens=256,
                            stop_token_ids=sorted(x for x in terminators if x is not None))
    try:
        for offset in range(0, len(todo), 128):
            batch = todo[offset:offset + 128]
            at = time.perf_counter()
            outputs = llm.generate([{'prompt_token_ids': tokenized[pid]} for pid in batch], params, use_tqdm=False)
            seconds = time.perf_counter() - at
            with dest.open('a', encoding='utf-8') as stream:
                for pid, output in zip(batch, outputs, strict=True):
                    result = output.outputs[0]
                    raw = result.text
                    parsed, valid = clean_json(raw)
                    record = {'id': pid, 'raw': raw, 'parsed_pred': parsed, 'parse_ok': valid,
                              'latency_seconds': seconds / len(batch), 'batch_seconds': seconds,
                              'batch_count': len(batch), 'stage': 'p0_a2',
                              'prompt_tokens': len(tokenized[pid]),
                              'output_tokens': len(result.token_ids), 'finish_reason': result.finish_reason}
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    done[pid] = record
                stream.flush()
                os.fsync(stream.fileno())
            print('P0_A2_PROGRESS', len(done), '/', len(tokenized), flush=True)
        assert set(done) == set(tokenized)
        elapsed = time.perf_counter() - started
        report = {'model': 'Qwen2.5-7B-Instruct', 'count': len(done),
                  'new_generated_this_run': len(todo), 'output_sha256': sha(dest),
                  'input_sha256': sha(OUT / 'new_prompts.jsonl'),
                  'historical_report_sha256': sha(OUT / 'step_b_full.report.json'),
                  'chat_template_sha256': historical['chat_template_sha256'],
                  'generation': historical['generation'], 'backend': 'vllm', 'dtype': 'bfloat16',
                  'max_num_seqs': 16, 'max_model_len': 3880, 'queued_batch': 128,
                  'max_input_tokens': max_input, 'elapsed_seconds': elapsed,
                  'json_parse_rate': sum(r['parse_ok'] for r in done.values()) / len(done),
                  'peak_torch_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    finally:
        STOP.set()
        monitor.join(timeout=5)
    report['peak_gpu_memory_gib_nvidia_smi'] = monitor_result.get('mb', 0) / 1024
    (OUT / 'inference.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('P0_A2_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
