"""Measured online batch=1 timing on the frozen 50-request exploration workload."""
import argparse
import copy
import hashlib
import inspect
import json
import threading
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from step_b_caller_runner import ROOT, STOP, gpu_monitor, sha


OUT = ROOT / "research_outputs/step_c_confirm/c2"
BEGIN, END = "[BEGIN OF AVAILABLE TOOLS]\n", "\n[END OF AVAILABLE TOOLS]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-id", required=True)
    a = p.parse_args()
    path = OUT / "benchmark_inputs.json"
    for line in (OUT / "BENCHMARK_SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(OUT / name) == digest
    data = json.loads(path.read_bytes())
    requests = data["requests"]
    assert len(requests) == 50 and data["batch_size"] == 1
    base = ROOT / "models/Qwen2-1.5B-Instruct"
    adapter = ROOT / "research_outputs/relgate_v2_final/RelGate-3Way-v2/adapter"
    assert sha(adapter / "adapter_model.safetensors") == "8403404d9518324bf16cb457dc25b795fb134309687a1c74313863215fa02259"
    gate_tok = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side="left")
    gate_tok.pad_token_id = gate_tok.eos_token_id
    gate_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", local_files_only=True)
    gate_model = PeftModel.from_pretrained(gate_model, adapter).eval().to("cuda")
    signature = inspect.signature(gate_model.get_base_model().forward).parameters
    keep = "logits_to_keep" if "logits_to_keep" in signature else "num_logits_to_keep" if "num_logits_to_keep" in signature else None
    assert keep, "bounded suffix logits are required for the candidate mini-batch"
    answers = [gate_tok.encode(s, add_special_tokens=False) for s in ("SUPPORTED", "UNVERIFIED", "CONTRADICTED")]
    max_answer = max(map(len, answers))
    instructions = {c["id"]: c["instruction"] for r in requests for c in r["candidates"]}
    parsed = {}
    for r in requests:
        prefix, block = r["messages"][0]["content"].split(BEGIN, 1)
        block, suffix = block.split(END, 1)
        parsed[r["id"]] = (prefix + BEGIN, json.loads(block), END + suffix)

    def gate(row, cache):
        _, tools, _ = parsed[row["id"]]
        keys = []
        for t in tools:
            triple = {k: t.get(k) for k in ("name", "description", "parameters")}
            key = hashlib.sha256(json.dumps(triple, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            keys.append(key)
        assert keys == row["candidate_ids"]
        missing = list(dict.fromkeys(k for k in keys if k not in cache))
        if missing:
            sequences = []
            for key in missing:
                ids = gate_tok.apply_chat_template([{"role": "user", "content": instructions[key]}], tokenize=True, add_generation_prompt=True)
                sequences.extend(ids + answer[:-1] for answer in answers)
            batch = gate_tok.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to("cuda")
            positions = batch["attention_mask"].long().cumsum(-1) - 1
            positions.masked_fill_(batch["attention_mask"] == 0, 0)
            logits = gate_model(**batch, position_ids=positions, use_cache=False, **{keep: max_answer}).logits.float().log_softmax(-1)
            values = []
            for i in range(len(missing)):
                likelihoods = []
                for j, answer in enumerate(answers):
                    scores = logits[i * 3 + j, -len(answer):]
                    likelihoods.append(sum(scores[k, token] for k, token in enumerate(answer)))
                values.append(torch.stack(likelihoods))
            probabilities = torch.stack(values).double().softmax(-1)[:, 2].cpu().tolist()
            cache.update(zip(missing, probabilities))
        mask = [cache[k] >= .50 for k in keys]
        # Timing never replaces the hash-locked experiment actions.
        discrepancies = sum(x != y for x, y in zip(mask, row["expected_mask"], strict=True))
        return missing, discrepancies

    records = []
    def timed_stage(stage, function):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for row in requests:
            torch.cuda.synchronize()
            at = time.perf_counter()
            extra = function(row)
            torch.cuda.synchronize()
            records.append(dict(stage=stage, id=row["id"], ms=(time.perf_counter() - at) * 1000,
                                candidate_count=row["candidate_count"], **extra))
        subset = [r for r in records if r["stage"] == stage]
        report = dict(stage=stage, requests=50, elapsed_seconds=time.perf_counter() - start,
                      mean_ms_per_request=sum(r["ms"] for r in subset) / 50,
                      peak_torch_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      candidate_cache_misses=sum(r.get("misses", 0) for r in subset),
                      batched_vs_reference_mask_discrepancies=sum(r.get("mask_discrepancies", 0) for r in subset))
        print("BENCH_STAGE", json.dumps(report), flush=True)
        return report

    def gate_call(row, cache):
        missing, errors = gate(row, cache)
        return {"misses": len(missing), "mask_discrepancies": errors}

    folder = OUT / "microbench" / a.model_id
    folder.mkdir(parents=True, exist_ok=True)
    monitor_value = {}
    monitor = threading.Thread(target=lambda: monitor_value.setdefault("mb", gpu_monitor()), daemon=True)
    monitor.start()
    reports = []
    with torch.inference_mode():
        gate(requests[0], {})
        reports.append(timed_stage("gate_only_cold_each_request", lambda r: gate_call(r, {})))
        cache = {}
        reports.append(timed_stage("gate_only_stream_cache_empty_at_start", lambda r: gate_call(r, cache)))
        gate_model.to("cpu")
        torch.cuda.empty_cache()
        caller_tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        caller = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", local_files_only=True).eval().to("cuda")
        stops = {caller_tok.eos_token_id}
        for special in ("<|im_end|>", "<|eot_id|>"):
            token = caller_tok.convert_tokens_to_ids(special)
            if token is not None and token != caller_tok.unk_token_id:
                stops.add(token)

        def call(messages):
            text = caller_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = caller_tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            output = caller.generate(**batch, do_sample=False, temperature=0.0, max_new_tokens=256,
                                     eos_token_id=sorted(x for x in stops if x is not None), pad_token_id=caller_tok.eos_token_id)
            raw = caller_tok.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            return {"output_tokens": output.shape[1] - batch["input_ids"].shape[1],
                    "raw_sha256": hashlib.sha256(raw.encode()).hexdigest()}

        def combined(row, cache):
            result = gate_call(row, cache)
            prefix, tools, suffix = parsed[row["id"]]
            modified = copy.deepcopy(tools)
            for t, name in zip(modified, row["visible_names"], strict=True):
                t["name"] = name
            messages = [{"role": "user", "content": prefix + json.dumps(modified, ensure_ascii=False) + suffix}]
            assert messages == row["gate_messages"]
            result.update(call(messages))
            return result

        call(requests[0]["gate_messages"])
        reports.append(timed_stage("caller_only_same_frozen_gate_prompts", lambda r: call(r["gate_messages"])))
        torch.cuda.empty_cache()
        gate_model.to("cuda")
        gate(requests[0], {})
        reports.append(timed_stage("combined_cold_each_request", lambda r: combined(r, {})))
        cache = {}
        reports.append(timed_stage("combined_stream_cache_empty_at_start", lambda r: combined(r, cache)))
    STOP.set()
    monitor.join(timeout=5)
    report = {"model_id": a.model_id, "input_sha256": sha(path), "requests": 50,
              "backend": "Transformers BF16 SDPA for both caller and gate; caller alone during caller-only stage, co-resident during combined stages",
              "batch_size": 1, "cross_request_batching": False,
              "gate_forward": "one forward per uncached request mini-batch, expanded over all three full labels; no EOS; no greedy generation",
              "mask_policy": "frozen Step B actions used for identical caller load; numerical batch mask differences counted separately",
              "cache": "candidate metadata SHA256; stream starts empty; no cross-request batching",
              "stages": reports, "peak_device_gib_monitor_all_stages": monitor_value.get("mb", 0) / 1024,
              "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256},
              "timing": "synchronized measured wall time includes tokenization, scoring/cache lookup, prompt assembly and generation where applicable; excludes model loading and one warmup",
              "difference_from_full_run": "full evaluation uses vLLM batched throughput; these online results use identical Transformers batch=1 for both model families"}
    (folder / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    report["requests_sha256"] = sha(folder / "requests.jsonl")
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BENCH_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
