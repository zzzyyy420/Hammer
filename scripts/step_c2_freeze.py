"""Lock the authorized seven-view C2 scope before any final GPU outputs."""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs/step_c_confirm"
DEST = OUT / "c2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def locked_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, f"Refusing to change frozen artifact: {path.name}"
    else:
        path.write_bytes(content)


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def main():
    manifest = OUT / "SHA256SUMS.txt"
    checked = 0
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        assert sha(OUT / relative) == digest, relative
        checked += 1
    expected = {
        "heldout": {"Clean": 334, "Coarse-Desc": 334, "Name-Swapped": 334,
                    "Verbose-Coarse": 334, "Target-Swapped": 309,
                    "Distractor-Swapped": 309, "Verb-Flip-Target": 156},
        "bfcl": {"BFCL-Original": 1253, "BFCL-Name-Swapped": 1253, "BFCL-Verb-Flip": 177},
    }
    candidates, instructions, sources, inputs = {}, {}, {}, {}
    for dataset in ("heldout", "bfcl"):
        folder = OUT / "c15" / dataset
        views = folder / ("private/seven_views.jsonl" if dataset == "heldout" else "views.jsonl")
        assert Counter(r["view"] for r in read(views)) == expected[dataset]
        for name in ("unique_candidate_triples.jsonl", "score_inputs.jsonl", "static_unique_prompts.jsonl", "static_schedule.jsonl"):
            path = folder / name
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
        sources[views.relative_to(ROOT).as_posix()] = sha(views)
        rows = read(folder / "unique_candidate_triples.jsonl")
        scores = read(folder / "score_inputs.jsonl")
        assert {r["id"] for r in rows} == {r["id"] for r in scores}
        assert len({r["id"] for r in rows}) == len(rows)
        for row in rows:
            assert row["id"] not in candidates or candidates[row["id"]] == row
            candidates[row["id"]] = row
        for row in scores:
            assert row["id"] not in instructions or instructions[row["id"]] == row
            instructions[row["id"]] = row
        inputs[dataset] = {"view_counts": expected[dataset], "candidate_triples": len(rows),
                           "static_unique_prompts": len(read(folder / "static_unique_prompts.jsonl"))}
    assert len(candidates) == len(instructions) <= 7577
    for name, records in (("unique_candidate_triples.jsonl", candidates), ("score_inputs.jsonl", instructions)):
        content = "".join(json.dumps(records[k], ensure_ascii=False, sort_keys=True) + "\n" for k in sorted(records)).encode()
        locked_write(DEST / name, content)
    artifacts = [
        "research_outputs/relgate_v2_final/RelGate-3Way-v2/adapter/adapter_model.safetensors",
        "research_outputs/relgate_v2_final/RelGate-3Way-v2/adapter/adapter_config.json",
        "research_outputs/phase1_gpu_v1/Pure-NameMask/adapter_model.safetensors",
        "research_outputs/phase1_gpu_v1/Pure-NameMask/adapter_config.json",
        "research_outputs/routeB_step2b_v1/B2/attempt1/adapter/adapter_model.safetensors",
        "research_outputs/routeB_step2b_v1/B2/attempt1/adapter/adapter_config.json",
        "research_outputs/relgate_v1/data/external_gate_prompt_v1.txt",
        "research_outputs/step_c_confirm/microbench_50_clean.json",
        "research_outputs/step_b_plugplay/model_provenance.json",
        "scripts/step_b_caller_runner.py", "scripts/pareto_policies.py",
        "scripts/step_c_bfcl_adapter.py", "scripts/build_pure_name_pair.py",
        "upstream/Hammer/train/data_processing.py",
    ]
    for name in artifacts:
        sources[name] = sha(ROOT / name)
    assert sources[artifacts[0]] == "8403404d9518324bf16cb457dc25b795fb134309687a1c74313863215fa02259"
    protocol = {
        "status": "C2 authorized by uploaded C1.5 approval; frozen before final predictions",
        "supersedes": "phase_c2_budget.json preparation status and obsolete Desc-Degraded blocker",
        "threshold": 0.50, "gate": "RelGate-3Way-v2",
        "training_or_tuning_allowed": False,
        "heldout_desc_degraded": "N/A: original independent annotation list unavailable; do not reconstruct",
        "datasets": inputs,
        "callers_in_order": ["Hammer-Repro 1.5B", "B2 1.5B", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"],
        "policies": ["Keep-Name", "Always-Mask", "Length-64", "RelGate-0.50"],
        "matched_seeds": [42, 123, 2026], "matched_views": ["Target-Swapped", "Distractor-Swapped"],
        "matched_scope": "heldout only; per-request equal masked count from actual gate output",
        "gate_score": "full three-label sequence log likelihood without EOS; softmax; CONTRADICTED >= 0.50",
        "labels": ["SUPPORTED", "UNVERIFIED", "CONTRADICTED"],
        "decoding": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256},
        "format": "frozen Hammer instruction, tools and JSON wrapper; native caller chat template with generation prompt",
        "parser": "exact clean_json from hash-pinned Step B runner",
        "bfcl_commit": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "retain_official_schema_gold_anomalies": ["live_multiple_862-181-3", "live_multiple_964-207-0"],
        "smoke": {"count": 20, "view": "Clean", "policy": "Keep-Name", "minimum_json_parse_rate": 0.95,
                  "on_failure": "stop that model and return raw outputs; no prompt tuning"},
        "bootstrap": {"replicates": 20000, "confidence": 0.95, "unit": "joint paired original request ID",
                      "seed": 2026},
        "macro_comparisons": "exploration/heldout comparisons restricted to identical shared view sets",
        "microbenchmark": {"requests": 50, "batch": 1, "cross_request_batching": False,
                           "gate": "candidate mini-batch inside one request; pure scoring; cold and streaming hot cache",
                           "report": "caller only, gate only and measured combined end-to-end; distinguish batch throughput"},
        "unique_candidates_exact": len(candidates),
        "logical_units_per_caller": 21026, "logical_units_all_callers": 84104,
        "unique_prompt_upper_per_caller": {"heldout": 8146, "bfcl": 7297},
        "dedup_counts": "exact dynamic/matched counts only after frozen gate scores",
        "finish": "return and SHA/count verify each model before clearing weights; shutdown immediately after all GPU artifacts verified, before local analysis",
        "inputs_sha256": sources, "prior_manifest_sha256": sha(manifest), "prior_files_verified": checked,
        "score_inputs_sha256": sha(DEST / "score_inputs.jsonl"),
    }
    locked_write(DEST / "protocol.json", encode(protocol))
    locked_write(DEST / "SHA256SUMS.txt", "".join(f"{sha(p)}  {p.name}\n" for p in sorted(DEST.iterdir())
                                                if p.is_file() and p.name != "SHA256SUMS.txt").encode())
    for line in (DEST / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(DEST / name) == digest
    print(json.dumps({"prior_files_verified": checked, "unique_candidates_exact": len(candidates),
                      "protocol_sha256": sha(DEST / "protocol.json"), "datasets": inputs,
                      "gpu_used_by_local_freeze": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
