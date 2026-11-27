"""C2 native-chat vLLM inference with frozen parsing and a smoke gate."""
import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from step_b_caller_runner import clean_json, call_name, gpu_monitor, STOP, sha, rows, GENERATION


ROOT = Path("/root/autodl-tmp/Hammer")
OUT = ROOT / "research_outputs/step_c_confirm/c2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    started = time.perf_counter()
    for manifest in ("SHA256SUMS.txt", "SCHEDULE_SHA256SUMS.txt"):
        for line in (OUT / manifest).read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            assert sha(OUT / name) == digest, name
    protocol = json.loads((OUT / "protocol.json").read_bytes())
    if args.adapter:
        relative = (Path(args.adapter) / "adapter_model.safetensors").relative_to(ROOT).as_posix()
        assert sha(ROOT / relative) == protocol["inputs_sha256"][relative]
    promptpath = OUT / "unique_prompts.jsonl"
    prompts = rows(promptpath)
    assert len({r["id"] for r in prompts}) == len(prompts)
    smoke = rows(OUT / "private/smoke_gold_clean_keep20.jsonl")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = {r["id"]: tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True) for r in prompts}
    max_length = max(map(len, tokens.values())) + 256
    folder = OUT / "raw" / args.model_id
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "predictions.jsonl"
    old = rows(dest) if dest.exists() else []
    done = {r["id"]: r for r in old}
    assert len(done) == len(old) and set(done) <= set(tokens)
    llm = LLM(model=args.model, dtype="bfloat16", seed=42, enforce_eager=True,
              max_model_len=max_length, gpu_memory_utilization=.88, max_num_seqs=16,
              enable_lora=bool(args.adapter), **({"max_lora_rank": 32} if args.adapter else {}))
    lora = LoRARequest(args.model_id, 1, args.adapter) if args.adapter else None
    stop = {tok.eos_token_id}
    for special in ("<|im_end|>", "<|eot_id|>"):
        value = tok.convert_tokens_to_ids(special)
        if value is not None and value >= 0 and value != tok.unk_token_id:
            stop.add(value)
    sampling = SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=sorted(x for x in stop if x is not None))
    monitor_value = {}
    monitor = threading.Thread(target=lambda: monitor_value.setdefault("mb", gpu_monitor()), daemon=True)
    monitor.start()

    def generate(ids, stage):
        todo = [k for k in dict.fromkeys(ids) if k not in done]
        begin = time.perf_counter()
        for offset in range(0, len(todo), 128):
            batch = todo[offset:offset + 128]
            at = time.perf_counter()
            outputs = llm.generate([{"prompt_token_ids": tokens[k]} for k in batch], sampling, lora_request=lora, use_tqdm=False)
            seconds = time.perf_counter() - at
            with dest.open("a", encoding="utf-8") as stream:
                for key, output in zip(batch, outputs, strict=True):
                    result = output.outputs[0]
                    raw = result.text
                    parsed, valid = clean_json(raw)
                    record = {"id": key, "raw": raw, "parsed_pred": parsed, "parse_ok": valid,
                              "latency_seconds": seconds / len(batch), "batch_seconds": seconds,
                              "batch_count": len(batch), "stage": stage, "prompt_tokens": len(tokens[key]),
                              "output_tokens": len(result.token_ids), "finish_reason": result.finish_reason}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done[key] = record
                stream.flush()
                os.fsync(stream.fileno())
            print(args.model_id, stage, len(done), "/", len(tokens), flush=True)
        return time.perf_counter() - begin

    common = {"model_id": args.model_id, "base": args.model, "adapter": args.adapter,
              "adapter_sha256": sha(Path(args.adapter) / "adapter_model.safetensors") if args.adapter else None,
              "generation": GENERATION, "protocol_sha256": sha(OUT / "protocol.json"),
              "input_sha256": sha(promptpath), "schedule_sha256": sha(OUT / "schedule.jsonl"),
              "backend": "vllm", "dtype": "bfloat16", "max_num_seqs": 16, "queued_batch": 128,
              "max_model_len": max_length, "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest()}
    try:
        generate([r["prompt"] for r in smoke], "smoke")
        smoke_preds = [done[r["prompt"]] for r in smoke]
        parse_rate = sum(r["parse_ok"] for r in smoke_preds) / 20
        correct = 0
        for cell, pred in zip(smoke, smoke_preds):
            calls = pred["parsed_pred"]
            if pred["parse_ok"] and isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
                correct += call_name(calls[0]) == call_name(cell["ground_truth"][0])
        smoke_report = dict(common, count=20, json_parse_rate=parse_rate, tool_acc=correct / 20, gate_pass=parse_rate >= .95)
        (folder / "smoke.report.json").write_text(json.dumps(smoke_report, indent=2) + "\n")
        print("SMOKE", json.dumps(smoke_report), flush=True)
        if parse_rate < .95:
            raise SystemExit("SMOKE GATE FAILED; stop without prompt tuning")
        full_seconds = generate(tokens, "full")
        assert set(done) == set(tokens)
        seconds = sum(r["latency_seconds"] for r in done.values())
        report = dict(common, count=len(done), output_sha256=sha(dest),
                      elapsed_seconds=time.perf_counter() - started, full_stage_seconds_this_process=full_seconds,
                      inference_seconds_including_smoke=seconds, average_amortized_ms_per_prompt=seconds * 1000 / len(done),
                      latency_definition="batched amortized throughput; not batch=1 online latency",
                      json_parse_rate=sum(r["parse_ok"] for r in done.values()) / len(done))
    finally:
        STOP.set()
        monitor.join(timeout=5)
    report["peak_gpu_memory_gib_nvidia_smi"] = monitor_value.get("mb", 0) / 1024
    (folder / "full.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FULL_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
