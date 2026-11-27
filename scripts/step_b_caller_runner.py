"""Run a frozen Step B caller schedule with native chat formatting."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/root/autodl-tmp/Hammer")
OUT = ROOT / "research_outputs/step_b_plugplay"
INPUTS = OUT / "unique_prompts.jsonl"
SMOKE_KEYS = OUT / "smoke_gold_clean_keep20.jsonl"
GENERATION = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256}
STOP = threading.Event()


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(raw):
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if match:
        text = match.group(1)
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, TypeError):
        return None, False


def call_name(call):
    return call.get("name", call.get("tool"))


def gpu_monitor():
    peak = 0
    while not STOP.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=5)
            peak = max(peak, max(int(x.strip()) for x in result.stdout.splitlines() if x.strip()))
        except Exception:
            pass
        STOP.wait(0.5)
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--resolved-commit")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()

    model_dir = Path(a.model)
    schedule_name = a.model_id + ".schedule.jsonl"
    schedule = rows(OUT / schedule_name)
    prompts = {r["id"]: r for r in rows(INPUTS)}
    assert len(prompts) == len(rows(INPUTS)), "duplicate prompt IDs in frozen input"
    assert all(r["prompt"] in prompts for r in schedule), "schedule references missing prompt"
    smoke_rows = [r for r in schedule
                  if r["view"] == "Clean" and r["policy"] == "Keep-Name"][:20]
    assert len(smoke_rows) == 20, f"expected 20 Clean Keep-Name rows, got {len(smoke_rows)}"
    smoke_ids = [r["prompt"] for r in smoke_rows]
    smoke_keys = {}
    if a.mode == "smoke":
        smoke_keys = {(r["id"], r["view"], r["mode"]): r for r in rows(SMOKE_KEYS)}
        assert len(smoke_keys) == 20, f"expected 20 isolated smoke gold keys, got {len(smoke_keys)}"
        assert all(r["view"] == "Clean" and r["mode"] == "Keep-Name" for r in smoke_keys.values())
        assert all((r["id"], "Clean", "Keep-Name") in smoke_keys for r in smoke_rows), (
            "smoke requests do not align with permitted Clean/Keep-Name gold keys")

    dest = OUT / "raw" / a.model_id / "predictions.jsonl"
    report_path = OUT / "raw" / a.model_id / f"{a.mode}.report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    available = smoke_ids if a.mode == "smoke" else list(prompts)
    done = {}
    if dest.exists():
        existing = rows(dest)
        done = {r["id"]: r for r in existing}
        assert len(done) == len(existing), "duplicate IDs in resumable output"
        assert set(done) <= set(prompts), "output contains IDs absent from frozen inputs"
    if a.mode == "smoke":
        assert not done, "smoke output already exists; refusing to overwrite"
    todo = [pid for pid in available if pid not in done]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=a.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, revision=a.revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True)
    model.eval().to("cuda")
    config_commit = getattr(model.config, "_commit_hash", None)
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    try:
        resolved_commit = config_commit or tokenizer_commit
    except Exception:
        resolved_commit = config_commit
    resolved_commit = resolved_commit or a.resolved_commit
    assert resolved_commit, "provide the resolved model commit recorded by the downloader"

    encoded = []
    for pid in todo:
        messages = prompts[pid]["messages"]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(formatted, dict):
            formatted = formatted["input_ids"]
        encoded.append((pid, formatted.reshape(-1)))
    if encoded:
        longest = max(len(ids) for _, ids in encoded)
        assert longest + GENERATION["max_new_tokens"] <= a.max_model_len, (
            f"prompt+generation length {longest + GENERATION['max_new_tokens']} "
            f"exceeds max model length {a.max_model_len}")

    stop_ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
            stop_ids.add(tid)
    terminators = sorted(x for x in stop_ids if x is not None)
    monitor_result = {}
    monitor = threading.Thread(target=lambda: monitor_result.setdefault("mb", gpu_monitor()), daemon=True)
    torch.cuda.reset_peak_memory_stats()
    monitor.start()
    start_all = time.perf_counter()
    request_times = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(encoded), a.batch_size):
                batch = encoded[offset:offset + a.batch_size]
                ids = [x[1].to("cuda") for x in batch]
                width = max(map(len, ids))
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long, device="cuda")
                attention_mask = torch.zeros_like(input_ids)
                for i, seq in enumerate(ids):
                    input_ids[i, width - len(seq):] = seq
                    attention_mask[i, width - len(seq):] = 1
                started = time.perf_counter()
                generated = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    do_sample=False, temperature=0.0, max_new_tokens=256,
                    eos_token_id=terminators, pad_token_id=pad_id,
                    use_cache=True)
                torch.cuda.synchronize()
                batch_elapsed = time.perf_counter() - started
                for i, (pid, seq) in enumerate(batch):
                    completion = generated[i, width:]
                    raw = tokenizer.decode(completion, skip_special_tokens=True)
                    parsed, ok = clean_json(raw)
                    latency = batch_elapsed / len(batch)
                    request_times.append(latency)
                    record = {"id": pid, "raw": raw, "parsed_pred": parsed, "parse_ok": ok,
                              "latency_seconds": latency}
                    with dest.open("a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                print(f"{a.model_id} {a.mode}: {min(offset + len(batch), len(encoded))}/{len(encoded)}", flush=True)
    finally:
        elapsed = time.perf_counter() - start_all
        STOP.set()
        monitor.join(timeout=5)
    final_rows = rows(dest) if dest.exists() else []
    if a.mode == "smoke":
        smoke = {r["id"]: r for r in final_rows if r["id"] in set(smoke_ids)}
        request_predictions = [smoke[r["prompt"]] for r in smoke_rows if r["prompt"] in smoke]
        parse_rate = (sum(r["parse_ok"] for r in request_predictions) / len(smoke_rows)
                      if request_predictions else 0.0)
        tool_correct = 0
        for request in smoke_rows:
            pred = smoke.get(request["prompt"])
            if pred and pred["parse_ok"] and isinstance(pred["parsed_pred"], list) and len(pred["parsed_pred"]) == 1:
                key = smoke_keys[(request["id"], "Clean", "Keep-Name")]
                if call_name(pred["parsed_pred"][0]) == call_name(key["ground_truth"][0]):
                    tool_correct += 1
        tool_acc = tool_correct / len(smoke_rows)
        report = {"mode": "smoke", "count": len(smoke_rows),
                  "unique_prompts_generated": len(smoke), "json_parse_rate": parse_rate,
                  "tool_acc": tool_acc, "tool_correct": tool_correct,
                  "elapsed_seconds": elapsed,
                  "gate_pass": len(request_predictions) == len(smoke_rows) and parse_rate >= 0.95,
                  "raw_samples": list(smoke.values())[:5]}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not report["gate_pass"]:
            raise SystemExit("SMOKE GATE FAILED: stopping this model before full inference")
    else:
        assert len(final_rows) == len(prompts), (len(final_rows), len(prompts))
        assert len({r["id"] for r in final_rows}) == len(final_rows)
        report = {
            "mode": "full", "model_id": a.model_id, "model_path": str(model_dir),
            "model_revision": a.revision, "resolved_commit": resolved_commit,
            "input_sha256": sha(INPUTS), "schedule_sha256": sha(OUT / schedule_name),
            "output_sha256": sha(dest), "count": len(final_rows),
            "generated_this_run": len(todo), "elapsed_seconds": elapsed,
            "cumulative_output_count": len(final_rows),
            "average_latency_ms_per_request": (sum(request_times) / len(request_times) * 1000
                                                if request_times else None),
            "peak_gpu_memory_gib_nvidia_smi": monitor_result.get("mb", 0) / 1024,
            "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "generation": GENERATION, "batch_size": a.batch_size,
            "json_parse_rate": sum(r["parse_ok"] for r in final_rows) / len(final_rows),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
