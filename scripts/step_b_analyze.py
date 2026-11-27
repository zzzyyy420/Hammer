"""Score the frozen Step B outputs and run request-paired exploratory CIs."""
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pareto_experiment import predictions
from routeb_stage1 import remap, rows
from score_routeB_probe import outcome


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs/step_b_plugplay"
BASE = ROOT / "research_outputs/step_a2_local_report"
V2 = ROOT / "research_outputs/relgate_v2_final/predictions"
V1 = ROOT / "research_outputs/relgate_v1/report/predictions"
STAGE2 = ROOT / "research_outputs/routeB_step2b_v1/report/predictions"
STEP = ROOT / "research_outputs/step_a2_incremental"
VIEWS = ("Clean", "Coarse-Desc", "Name-Swapped", "Target-Swapped",
         "Distractor-Swapped", "Desc-Degraded", "Verb-Flip-Target", "Verbose-Coarse")
COUNTS = dict(zip(VIEWS, (197, 197, 197, 182, 182, 165, 72, 197)))
POLICIES = ("Keep-Name", "Always-Mask", "Length-64", "RelGate-0.50")
NEW = ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
OLD = {"Pure-NameMask": "Hammer-Repro 1.5B", "B2": "B2 1.5B"}
SEEDS = (42, 123, 2026)
BOOTSTRAPS = 20000


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, data):
    assert data, path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def keys():
    frozen = rows(OUT / "Qwen2.5-7B-Instruct.schedule.jsonl")
    allowed = {(r["id"], r["view"]) for r in frozen if r["policy"] == "Keep-Name"}
    records = rows(ROOT / "research_outputs/routeB_stage1_v1/eval_keys.jsonl")
    result = {(r["id"], r["view"]): r for r in records
              if r["mode"] == "Keep-Name" and r["view"] in VIEWS[:6] and (r["id"], r["view"]) in allowed}
    assert Counter(v for _, v in result) == {v: COUNTS[v] for v in VIEWS[:6]}
    path = ROOT / "research_outputs/step_a_v1/probes.jsonl"
    assert sha(path) == "b7fa9b653252ae3db211837016fde002ca07c05acf747d10022c0bb485ac1ed6"
    probes = rows(path)
    assert Counter(p["view"] for p in probes) == {v: COUNTS[v] for v in VIEWS[6:]}
    for p in probes:
        key = copy.deepcopy(result[p["id"], "Clean"])
        key["view"] = p["view"]
        if p["view"] == "Verb-Flip-Target":
            key["ground_truth"][0]["name"] = p["tools"][p["g"]]["name"]
        result[p["id"], p["view"]] = key
    return result


def score_new(model, gold):
    folder = OUT / "raw" / model
    report = json.loads((folder / "full.report.json").read_text(encoding="utf-8"))
    raw_path = folder / "predictions.jsonl"
    assert report["count"] == 4125 and sha(raw_path) == report["output_sha256"]
    raw_rows = rows(raw_path)
    raw = {r["id"]: r for r in raw_rows}
    assert len(raw) == len(raw_rows) == 4125
    prompts = {r["id"] for r in rows(OUT / "unique_prompts.jsonl")}
    assert set(raw) == prompts
    schedule = rows(OUT / f"{model}.schedule.jsonl") + rows(OUT / "matched_random.schedule.jsonl")
    assert len(schedule) == 6648
    result = []
    for case in schedule:
        pred = raw[case["prompt"]]
        recovered = dict(pred, parsed_pred=remap(pred["parsed_pred"], case["inverse"]))
        tool, strict, hijack, invalid = outcome(recovered, gold[case["id"], case["view"]])
        result.append(dict(model=model, id=case["id"], view=case["view"], policy=case["policy"],
                           prompt=case["prompt"], tool_correct=tool, strict_correct=strict,
                           invalid=invalid, name_hijack=hijack if case["view"] in ("Name-Swapped", "Target-Swapped") else None,
                           masked_count=case["masked_count"], candidate_count=case["candidate_count"]))
    expected = {v: COUNTS[v] for v in VIEWS}
    for policy in POLICIES:
        assert Counter(r["view"] for r in result if r["policy"] == policy) == expected
    for seed in SEEDS:
        assert Counter(r["view"] for r in result if r["policy"] == f"Matched-0.50-seed{seed}") == {
            "Target-Swapped": 182, "Distractor-Swapped": 182}
    return result


def old_standard(model, policy):
    if model == "B2":
        return rows(V2 / f"B2__{policy}.jsonl")
    stem = "Global-Length-64" if policy == "Length-64" else policy
    records = rows(V1 / f"Pure-NameMask__{stem}.jsonl") if (V1 / f"Pure-NameMask__{stem}.jsonl").exists() else []
    if policy in ("Keep-Name", "Always-Mask"):
        extras = rows(STAGE2 / f"Pure-NameMask__{policy}.jsonl")
        merged = {(r["id"], r["view"]): r for r in records}
        merged.update({(r["id"], r["view"]): r for r in extras})
        records = list(merged.values())
    if policy == "RelGate-0.50":
        records = []
    return records


def score_old(model, gold):
    result = []
    incremental = {(r["id"], r["view"], r["policy"]): r for r in rows(STEP / f"{model}__step_a2.cases.jsonl")}
    cached = predictions(model)
    for policy in POLICIES:
        indexed = {(r["id"], r["view"]): r for r in old_standard(model, policy)}
        for (sid, view, chosen), case in incremental.items():
            if chosen != policy or (sid, view) in indexed:
                continue
            if view not in VIEWS and not (model == "Pure-NameMask" and policy == "RelGate-0.50"):
                continue
            pred = cached[case["prompt"]]
            recovered = dict(pred, parsed_pred=remap(pred["parsed_pred"], case["inverse"]))
            tool, strict, hijack, invalid = outcome(recovered, gold[sid, view])
            indexed[sid, view] = dict(id=sid, view=view, tool_correct=tool, strict_correct=strict,
                                       name_hijack=hijack, invalid=invalid)
        for (sid, view), r in indexed.items():
            if view not in VIEWS:
                continue
            result.append(dict(model=OLD[model], id=sid, view=view, policy=policy,
                               tool_correct=int(r["tool_correct"]), strict_correct=int(r["strict_correct"]),
                               invalid=int(r["invalid"]), name_hijack=r.get("name_hijack")
                               if view in ("Name-Swapped", "Target-Swapped") else None))
    for policy in POLICIES:
        assert Counter(r["view"] for r in result if r["policy"] == policy) == COUNTS
    return result


def harmonic(values):
    arr = np.stack(values, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.all(arr > 0, axis=0), len(values) / np.sum(1 / arr, axis=0), 0.0)
    return float(result) if np.ndim(result) == 0 else result


def aggregates(rates):
    return {"H_dual": harmonic([rates["Coarse-Desc"], rates["Name-Swapped"]]),
            "three_view_HM": harmonic([rates[v] for v in VIEWS[:3]]),
            "mixed_mean": np.mean([rates[v] for v in ("Target-Swapped", "Distractor-Swapped")], axis=0),
            "mixed_HM": harmonic([rates[v] for v in ("Target-Swapped", "Distractor-Swapped")]),
            "six_view_macro": np.mean([rates[v] for v in VIEWS[:6]], axis=0)}


def metrics(records):
    detail, summary = [], []
    groups = {}
    for r in records:
        groups.setdefault((r["model"], r["policy"], r["view"]), []).append(r)
    for model in dict.fromkeys(r["model"] for r in records):
        for policy in POLICIES:
            by_view = {}
            for view in VIEWS:
                seq = groups[model, policy, view]
                assert len(seq) == COUNTS[view], (model, policy, view, len(seq))
                n = len(seq)
                tool = sum(r["tool_correct"] for r in seq) / n
                strict = sum(r["strict_correct"] for r in seq) / n
                invalid = sum(r["invalid"] for r in seq) / n
                nhr = sum(r["name_hijack"] for r in seq) / n if view in ("Name-Swapped", "Target-Swapped") else None
                by_view[view] = tool
                detail.append(dict(model=model, policy=policy, view=view, n=n, tool_acc=tool,
                                   strict_match=strict, invalid_rate=invalid, NHR_abs=nhr))
            summary.append(dict(model=model, policy=policy, **aggregates(by_view)))
    return detail, summary


def baseline_parity(detail):
    with (BASE / "six_view_metrics.csv").open(encoding="utf-8", newline="") as f:
        standard = list(csv.DictReader(f))
    with (BASE / "probe_end_to_end.csv").open(encoding="utf-8", newline="") as f:
        probes = list(csv.DictReader(f))
    expected = {(r["model"], r["policy"], r["view"]): r for r in standard + probes}
    for r in detail:
        if r["model"] not in OLD.values():
            continue
        name = "B2" if r["model"] == "B2 1.5B" else "Hammer-Repro (66.38% Func+Param Mask)"
        p = expected[name, r["policy"], r["view"]]
        assert int(p["n"]) == r["n"]
        assert abs(float(p["tool_acc"]) - r["tool_acc"]) < 1e-12, r
        assert abs(float(p["strict_match"]) - r["strict_match"]) < 1e-12, r


def bootstrap(records):
    ids = sorted({r["id"] for r in records if r["view"] == "Clean"})
    assert len(ids) == 197
    position = {sid: i for i, sid in enumerate(ids)}
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(ids), size=(BOOTSTRAPS, len(ids)))
    weights = np.zeros((BOOTSTRAPS, len(ids)), dtype=np.uint16)
    np.add.at(weights, (np.arange(BOOTSTRAPS)[:, None], draws), 1)
    keys = list(POLICIES) + [f"Matched-0.50-seed{x}" for x in SEEDS]
    fields = ("tool_correct", "strict_correct")
    numer = np.zeros((len(ids), len(VIEWS), len(keys), len(fields)), dtype=np.float32)
    present = np.zeros((len(ids), len(VIEWS), len(keys)), dtype=np.float32)
    for r in records:
        i, j, k = position[r["id"]], VIEWS.index(r["view"]), keys.index(r["policy"])
        assert present[i, j, k] == 0, (r["id"], r["view"], r["policy"])
        present[i, j, k] = 1
        numer[i, j, k] = [r[field] for field in fields]
    sums = weights @ numer.reshape(len(ids), -1)
    sums = sums.reshape(BOOTSTRAPS, len(VIEWS), len(keys), len(fields))
    denoms = (weights @ present.reshape(len(ids), -1)).reshape(BOOTSTRAPS, len(VIEWS), len(keys))
    rates = np.divide(sums, denoms[..., None], out=np.zeros_like(sums), where=denoms[..., None] != 0)
    point = np.divide(numer.sum(0), present.sum(0)[..., None],
                      out=np.zeros_like(numer.sum(0)), where=present.sum(0)[..., None] != 0)

    def values(arr, field, policy):
        ix = fields.index(field)
        pi = keys.index(policy)
        views = {v: arr[..., j, pi, ix] for j, v in enumerate(VIEWS)}
        combined = aggregates(views)
        return {**views, **combined,
                "mixed_pool": (arr[..., VIEWS.index("Target-Swapped"), pi, ix] * 182
                               + arr[..., VIEWS.index("Distractor-Swapped"), pi, ix] * 182) / 364}

    contrasts = []
    for field in fields:
        for baseline in ("Keep-Name", "Length-64"):
            base_point, gate_point = values(point, field, baseline), values(point, field, "RelGate-0.50")
            base_boot, gate_boot = values(rates, field, baseline), values(rates, field, "RelGate-0.50")
            for metric in list(VIEWS) + ["H_dual", "three_view_HM", "mixed_mean", "mixed_HM", "six_view_macro", "mixed_pool"]:
                delta = gate_boot[metric] - base_boot[metric]
                lo, hi = np.quantile(delta, [.025, .975])
                n_requests = COUNTS[metric] if metric in COUNTS else (182 if metric.startswith("mixed") else 197)
                contrasts.append(dict(model=records[0]["model"], field=field, comparison=f"RelGate-0.50 minus {baseline}",
                                      metric=metric, n_requests=n_requests,
                                      delta_pp=100 * (gate_point[metric] - base_point[metric]),
                                      ci_low_pp=100 * lo, ci_high_pp=100 * hi, bootstrap_draws=BOOTSTRAPS))
        for seed in (*SEEDS, "3-seed mean"):
            ref = [f"Matched-0.50-seed{x}" for x in SEEDS] if seed == "3-seed mean" else [f"Matched-0.50-seed{seed}"]
            rgp = values(point, field, "RelGate-0.50")["mixed_pool"]
            rgb = values(rates, field, "RelGate-0.50")["mixed_pool"]
            rp = np.mean([values(point, field, p)["mixed_pool"] for p in ref], axis=0)
            rb = np.mean([values(rates, field, p)["mixed_pool"] for p in ref], axis=0)
            lo, hi = np.quantile(rgb-rb, [.025, .975])
            contrasts.append(dict(model=records[0]["model"], field=field,
                                  comparison=f"RelGate-0.50 minus Matched-0.50 {seed}", metric="mixed_pool",
                                  n_requests=182, delta_pp=100*(rgp-rp), ci_low_pp=100*lo,
                                  ci_high_pp=100*hi, bootstrap_draws=BOOTSTRAPS))
    return contrasts


def efficiency():
    gate = json.loads((OUT / "microbench.report.json").read_text())
    assert gate["requests"] == 100 and gate["unique_candidates"] == 294
    table = [dict(component="RelGate-3Way-v2 score + greedy", model="Qwen2-1.5B-Instruct",
                  workload="100 requests; 294 unique / 770 candidate occurrences; cache empty at start",
                  total_seconds=gate["elapsed_seconds"], average_ms_per_request=gate["actual_combined_cached_pipeline_ms_per_request"],
                  average_ms_per_candidate=gate["score_ms_per_candidate"] + gate["greedy_ms_per_candidate"],
                  peak_gpu_gib=gate["peak_gpu_memory_gib_nvidia_smi"],
                  timing_definition="measured cached pipeline; batch size 1")]
    for model in NEW:
        caller = json.loads((OUT / "raw" / model / "full.report.json").read_text())
        assert caller["count"] == 4125
        table.append(dict(component="general caller inference", model=model,
                          workload="4125 unique prompts, including 20 smoke prompts",
                          total_seconds=caller["elapsed_seconds"],
                          average_ms_per_request=caller["average_latency_ms_per_request"],
                          average_ms_per_candidate="", peak_gpu_gib=caller["peak_gpu_memory_gib_nvidia_smi"],
                          timing_definition=caller["latency_definition"]))
    write_csv(OUT / "efficiency.csv", table)
    notes = dict(gate_requests=100, candidate_occurrences=gate["candidate_occurrences"],
                 unique_candidates=gate["unique_candidates"],
                 candidate_cache_hit_fraction=1-gate["unique_candidates"]/gate["candidate_occurrences"],
                 score_ms_per_candidate=gate["score_ms_per_candidate"],
                 greedy_ms_per_candidate=gate["greedy_ms_per_candidate"],
                 actual_cached_score_ms_per_request=gate["actual_cached_score_ms_per_request"],
                 actual_cached_greedy_ms_per_request=gate["actual_cached_greedy_ms_per_request"],
                 actual_cached_full_ms_per_request=gate["actual_combined_cached_pipeline_ms_per_request"],
                 uncached_full_ms_per_request_derived=(gate["uncached_score_ms_per_request_from_measured_candidates"]
                                                       +gate["uncached_greedy_ms_per_request_from_measured_candidates"]),
                 comparison_note="Gate timing is batch-1 on a 100-request sample. Caller timing is batched, amortized over 4125 unique prompts; ratios are workload dependent.")
    (OUT / "efficiency_notes.json").write_text(json.dumps(notes, indent=2) + "\n")


def provenance():
    result = {}
    for model, prefix, repository in ((NEW[0], "qwen", "Qwen/Qwen2.5-7B-Instruct"),
                                      (NEW[1], "llama", "LLM-Research/Meta-Llama-3.1-8B-Instruct")):
        path = OUT / "logs" / f"{prefix}_repository.json"
        meta = json.loads(path.read_text())
        files = [f for f in meta["Data"]["Files"] if f["Type"] == "blob"]
        weights = [f for f in files if f["Path"].endswith(".safetensors")]
        assert weights
        result[model] = dict(repository=repository, repository_metadata_sha256=sha(path),
                             weight_files=[dict(path=f["Path"], revision=f["Revision"], sha256=f["Sha256"])
                                           for f in weights],
                             tokenizer_revisions=sorted({f["Revision"] for f in files if "tokenizer" in f["Path"]}),
                             caller_chat_template_sha256=json.loads((OUT / "raw" / model / "full.report.json").read_text())["chat_template_sha256"])
    (OUT / "model_provenance.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    gold = keys()
    new = [r for model in NEW for r in score_new(model, gold)]
    old = [r for model in OLD for r in score_old(model, gold)]
    detail, summary = metrics(old + new)
    baseline_parity(detail)
    write_csv(OUT / "scored_requests.csv", new)
    write_csv(OUT / "eight_view_metrics.csv", detail)
    write_csv(OUT / "aggregate_metrics.csv", summary)
    ci = [r for model in NEW for r in bootstrap([x for x in new if x["model"] == model])]
    write_csv(OUT / "paired_bootstrap_20000.csv", ci)
    efficiency()
    provenance()
    protocol = dict(bootstraps=BOOTSTRAPS, seed=42, unit="original request ID",
                    resampling="same 197-ID draw shared by views, policies and seeds within each model",
                    ci="pointwise percentile 2.5/97.5%; exploratory, no multiplicity adjustment",
                    matched_random="mean of three fixed seeds per request, then paired contrast",
                    old_models="recomputed from archived scored predictions and probe raw predictions; parity checked against prior tables",
                    standard_denominators={v: COUNTS[v] for v in VIEWS})
    (OUT / "analysis_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("Scored", len(new), "new cells; baseline parity and 20,000 paired draws passed")


if __name__ == "__main__":
    main()
