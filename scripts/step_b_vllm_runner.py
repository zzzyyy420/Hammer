"""Native-chat BF16 caller inference with a mandatory smoke gate."""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from step_b_caller_runner import OUT, INPUTS, SMOKE_KEYS, GENERATION, rows, sha, clean_json, call_name, gpu_monitor, STOP


def main():
    started_process = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--repository-record", required=True)
    a = p.parse_args()
    from vllm import LLM, SamplingParams

    # Every policy and seed resolves to this one frozen prompt pool.
    for line in (OUT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(OUT / name) == digest, name
    prompts = rows(INPUTS)
    assert len({r["id"] for r in prompts}) == len(prompts)
    schedule = rows(OUT / (a.model_id + ".schedule.jsonl"))
    smoke = [r for r in schedule if r["view"] == "Clean" and r["policy"] == "Keep-Name"][:20]
    assert len(smoke) == 20
    smoke_keys = {r["id"]: r for r in rows(SMOKE_KEYS)}
    assert set(smoke_keys) == {r["id"] for r in smoke}
    repo_path = Path(a.repository_record)
    repo = json.loads(repo_path.read_text())
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tokenized = {r["id"]: tok.apply_chat_template(r["messages"], tokenize=True,
                   add_generation_prompt=True) for r in prompts}
    length = max(map(len, tokenized.values())) + 256
    folder = OUT / "raw" / a.model_id
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "predictions.jsonl"
    old = rows(dest) if dest.exists() else []
    done = {r["id"]: r for r in old}
    assert len(done) == len(old) and set(done) <= set(tokenized)
    monitor_result = {}
    monitor = threading.Thread(target=lambda: monitor_result.setdefault("mb", gpu_monitor()), daemon=True)
    monitor.start()
    llm = LLM(model=a.model, dtype="bfloat16", seed=42, enforce_eager=True,
              max_model_len=length, gpu_memory_utilization=.88, max_num_seqs=16)
    terminators = {tok.eos_token_id}
    for special in ("<|im_end|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(special)
        if tid is not None and tid >= 0 and tid != tok.unk_token_id:
            terminators.add(tid)
    params = SamplingParams(temperature=0.0, max_tokens=256,
                            stop_token_ids=sorted(x for x in terminators if x is not None))
    common = dict(model_id=a.model_id, model_path=a.model,
                  repository_record_sha256=sha(repo_path),
                  resolved_commit=repo["Data"]["LatestCommitter"],
                  input_sha256=sha(INPUTS), schedule_sha256=sha(OUT / (a.model_id + ".schedule.jsonl")),
                  generation=GENERATION, backend="vllm", dtype="bfloat16", max_num_seqs=16,
                  max_model_len=length, chat_template_sha256=__import__("hashlib").sha256(tok.chat_template.encode()).hexdigest())

    def generate(ids, stage):
        todo = [pid for pid in dict.fromkeys(ids) if pid not in done]
        begin = time.perf_counter()
        for start in range(0, len(todo), 128):
            batch = todo[start:start + 128]
            at = time.perf_counter()
            output = llm.generate([{"prompt_token_ids": tokenized[pid]} for pid in batch], params, use_tqdm=False)
            seconds = time.perf_counter() - at
            with dest.open("a", encoding="utf-8") as f:
                for pid, result in zip(batch, output, strict=True):
                    raw = result.outputs[0].text
                    parsed, valid = clean_json(raw)
                    record = dict(id=pid, raw=raw, parsed_pred=parsed, parse_ok=valid,
                                  latency_seconds=seconds / len(batch), batch_seconds=seconds,
                                  batch_count=len(batch), stage=stage,
                                  finish_reason=result.outputs[0].finish_reason,
                                  prompt_tokens=len(tokenized[pid]), output_tokens=len(result.outputs[0].token_ids))
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done[pid] = record
                f.flush()
                os.fsync(f.fileno())
            print(stage, len(done), "/", len(prompts), flush=True)
        return time.perf_counter() - begin

    try:
        generate([r["prompt"] for r in smoke], "smoke")
        smoke_pred = [done[r["prompt"]] for r in smoke]
        parse_rate = sum(x["parse_ok"] for x in smoke_pred) / 20
        correct = 0
        for cell, pred in zip(smoke, smoke_pred):
            calls = pred["parsed_pred"]
            if pred["parse_ok"] and isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
                correct += call_name(calls[0]) == call_name(smoke_keys[cell["id"]]["ground_truth"][0])
        report = dict(common, count=20, json_parse_rate=parse_rate, tool_acc=correct / 20,
                      gate_pass=parse_rate >= .95, raw_samples=smoke_pred[:5])
        (folder / "smoke.report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("SMOKE", json.dumps(report), flush=True)
        if not report["gate_pass"]:
            raise SystemExit("SMOKE GATE FAILED")
        full_seconds = generate(list(tokenized), "full")
        assert set(done) == set(tokenized)
        records = list(done.values())
        generation_seconds = sum(r["latency_seconds"] for r in records)
        report = dict(common, count=len(records), output_sha256=sha(dest),
                      elapsed_seconds=time.perf_counter() - started_process,
                      inference_seconds_including_smoke=generation_seconds,
                      full_stage_seconds_this_process=full_seconds,
                      average_latency_ms_per_request=generation_seconds / len(records) * 1000,
                      latency_definition="amortized wall time per unique prompt; queued batches of 128, max_num_seqs=16",
                      json_parse_rate=sum(x["parse_ok"] for x in records) / len(records))
    finally:
        STOP.set()
        monitor.join(timeout=5)
    report["peak_gpu_memory_gib_nvidia_smi"] = monitor_result.get("mb", 0) / 1024
    report["peak_torch_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (folder / "full.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FULL_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
