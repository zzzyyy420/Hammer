"""Post-shutdown CPU scoring and joint paired confirmation statistics."""
import copy
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from step_c2_freeze import ROOT, OUT, DEST, read, sha, encode
from step_c_bfcl_adapter import frozen_json_parser, official_checker, score_raw
from paired_did_eval import call_parts
from step_c1_prepare import BFCL, REVISION


MODELS = ("Hammer-Repro-1.5B", "B2-1.5B", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
POLICIES = ("Keep-Name", "Always-Mask", "Length-64", "RelGate-0.50")
SEEDS = (42, 123, 2026)
VIEWS = ("Clean", "Coarse-Desc", "Name-Swapped", "Target-Swapped", "Distractor-Swapped", "Verb-Flip-Target", "Verbose-Coarse")
STANDARD = VIEWS[:5]
BFVIEWS = ("BFCL-Original", "BFCL-Name-Swapped", "BFCL-Verb-Flip")
REPS = 20000


def csvwrite(name, records):
    assert records
    with (DEST / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def hm(values):
    arr = np.stack(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.all(arr > 0, axis=0), len(values) / np.sum(1 / arr, axis=0), 0.)


def aggregates(values):
    return dict(H_dual=hm([values[v] for v in ("Coarse-Desc", "Name-Swapped")]),
                three_view_HM=hm([values[v] for v in VIEWS[:3]]),
                mixed_pool=np.mean([values[v] for v in VIEWS[3:5]], axis=0),
                mixed_HM=hm([values[v] for v in VIEWS[3:5]]),
                shared_five_standard_macro=np.mean([values[v] for v in STANDARD], axis=0),
                seven_view_macro=np.mean([values[v] for v in VIEWS], axis=0))


def score_all():
    assert json.loads((DEST / "shutdown.json").read_bytes()), "GPU shutdown must precede analysis"
    protocol = json.loads((DEST / "protocol.json").read_bytes())
    for name, expected in protocol["inputs_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert subprocess.check_output(["git", "-C", str(BFCL), "rev-parse", "HEAD"], text=True).strip() == REVISION
    parser, check = frozen_json_parser(ROOT), official_checker(BFCL)
    heldout = {r["id"]: r["record"] for r in read(OUT / "c15/heldout/private/restored_334.jsonl")}
    hv = {(r["id"], r["view"]): r for r in read(OUT / "c15/heldout/private/seven_views.jsonl")}
    bf = {}
    for split in ("multiple", "live_multiple"):
        answers = {r["id"]: r["ground_truth"] for r in read(OUT / f"c15/bfcl/{split}/official_answers.jsonl")}
        for r in read(BFCL / f"bfcl_eval/data/BFCL_v4_{split}.json"):
            bf[r["id"]] = {"schema": r["function"], "gold": answers[r["id"]], "split": split}
    schedule = read(DEST / "schedule.jsonl")
    prompts = {r["id"] for r in read(DEST / "unique_prompts.jsonl")}
    ledger = json.loads((DEST / "return_verifications.json").read_bytes())
    scored, failures = [], []
    for model in MODELS:
        for name, digest in ledger[model]["files"].items():
            assert sha(DEST / name) == digest
        predictions = read(DEST / f"raw/{model}/predictions.jsonl")
        raw = {r["id"]: r for r in predictions}
        assert len(predictions) == len(raw) == len(prompts) and set(raw) == prompts
        for case in schedule:
            pred = raw[case["prompt"]]
            row = {k: case[k] for k in ("dataset", "split", "id", "view", "policy", "prompt", "masked_count", "candidate_count")}
            row.update(model=model, invalid=int(not pred["parse_ok"]), tool_correct=0, strict_correct=0,
                       name_hijack=None, ast_error="")
            calls = pred["parsed_pred"]
            single = pred["parse_ok"] and isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict)
            if case["dataset"] == "heldout":
                gold_name, gold_args = call_parts(heldout[case["id"]]["answers"][0])
                if single:
                    visible_name, arguments = call_parts(calls[0])
                    view_name = case["visible_to_view"].get(visible_name) if isinstance(visible_name, str) else None
                    restored = case["view_to_original"].get(view_name)
                    row["tool_correct"] = int(restored == gold_name)
                    canonical = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    row["strict_correct"] = int(row["tool_correct"] and isinstance(arguments, dict) and canonical(arguments) == canonical(gold_args))
                    if case["view"] in ("Name-Swapped", "Target-Swapped"):
                        row["name_hijack"] = int(not row["tool_correct"] and view_name == gold_name)
                if row["name_hijack"] is None and case["view"] in ("Name-Swapped", "Target-Swapped"):
                    row["name_hijack"] = 0
            else:
                item = bf[case["id"]]
                result = score_raw(pred["raw"], parser, check, item["schema"], item["gold"],
                                   case["visible_to_view"], case["view_to_original"], item["split"])
                row["strict_correct"] = int(result["valid"])
                row["ast_error"] = result.get("error_type", "")
                if single:
                    visible_name, _ = call_parts(calls[0])
                    view_name = case["visible_to_view"].get(visible_name) if isinstance(visible_name, str) else None
                    restored = case["view_to_original"].get(view_name)
                    gold_name = next(iter(item["gold"][0]))
                    row["tool_correct"] = int(restored == gold_name)
                    if case["view"] == "BFCL-Name-Swapped":
                        row["name_hijack"] = int(not row["tool_correct"] and view_name == gold_name)
            scored.append(row)
        print("SCORED", model, len(schedule), flush=True)
        for dataset in ("heldout", "bfcl"):
            choices = [r for r in scored if r["model"] == model and r["dataset"] == dataset and r["policy"] == "RelGate-0.50" and not r["strict_correct"]]
            counts = Counter()
            for row in choices:
                category = "parse_failure" if row["invalid"] else "wrong_tool" if not row["tool_correct"] else "argument_error"
                if counts[category] >= 2:
                    continue
                counts[category] += 1
                entry = dict(row, category=category, raw=raw[row["prompt"]]["raw"])
                if dataset == "heldout":
                    entry.update(query=hv[row["id"], row["view"]]["query"], gold=heldout[row["id"]]["answers"])
                else:
                    entry.update(gold=bf[row["id"]]["gold"])
                failures.append(entry)
    csvwrite("scored_requests.csv", scored)
    (DEST / "failure_examples.jsonl").write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in failures).encode())
    return scored


def tables(records):
    detail, summary, bfdetail = [], [], []
    for model in MODELS:
        for policy in POLICIES:
            values = {}
            for view in VIEWS:
                seq = [r for r in records if r["model"] == model and r["dataset"] == "heldout" and r["policy"] == policy and r["view"] == view]
                n = len(seq)
                values[view] = sum(r["tool_correct"] for r in seq) / n
                detail.append(dict(model=model, policy=policy, view=view, n=n, tool_acc=values[view],
                                   strict_match=sum(r["strict_correct"] for r in seq) / n,
                                   invalid_rate=sum(r["invalid"] for r in seq) / n,
                                   NHR_abs=sum(r["name_hijack"] for r in seq) / n if view in ("Name-Swapped", "Target-Swapped") else ""))
            summary.append(dict(model=model, policy=policy, **{k: float(v) for k, v in aggregates(values).items()},
                                Desc_Degraded="N/A: original independent annotation list unavailable"))
            for split in ("multiple", "live_multiple", "pooled"):
                for view in BFVIEWS:
                    seq = [r for r in records if r["model"] == model and r["dataset"] == "bfcl" and r["policy"] == policy
                           and r["view"] == view and (split == "pooled" or r["split"] == split)]
                    n = len(seq)
                    bfdetail.append(dict(model=model, policy=policy, split=split, view=view, n=n,
                                         AST_accuracy=sum(r["strict_correct"] for r in seq) / n,
                                         tool_acc=sum(r["tool_correct"] for r in seq) / n,
                                         invalid_rate=sum(r["invalid"] for r in seq) / n))
    csvwrite("heldout_seven_view_metrics.csv", detail)
    csvwrite("heldout_aggregate_metrics.csv", summary)
    csvwrite("bfcl_official_AST_metrics.csv", bfdetail)
    return detail, summary, bfdetail


def bootstrap(records, model, dataset, split="pooled"):
    seq = [r for r in records if r["model"] == model and r["dataset"] == dataset and (split == "pooled" or r["split"] == split)]
    ids = sorted({r["id"] for r in seq})
    views = VIEWS if dataset == "heldout" else BFVIEWS
    policies = list(POLICIES) + ([f"Matched-0.50-seed{s}" for s in SEEDS] if dataset == "heldout" else [])
    fields = ("tool_correct", "strict_correct")
    index = {sid: i for i, sid in enumerate(ids)}
    numerator = np.zeros((len(ids), len(views), len(policies), 2), dtype=np.float64)
    present = np.zeros(numerator.shape[:-1], dtype=np.float64)
    for row in seq:
        i, j, k = index[row["id"]], views.index(row["view"]), policies.index(row["policy"])
        assert present[i, j, k] == 0
        present[i, j, k] = 1
        numerator[i, j, k] = [row[f] for f in fields]
    totals = numerator.sum(0)
    denominator = present.sum(0)
    point = np.divide(totals, denominator[..., None], out=np.zeros_like(totals), where=denominator[..., None] > 0)
    rng = np.random.default_rng(2026)
    rates = []
    for offset in range(0, REPS, 500):
        draws = rng.integers(0, len(ids), size=(min(500, REPS - offset), len(ids)))
        weights = np.zeros((len(draws), len(ids)), dtype=np.float64)
        np.add.at(weights, (np.arange(len(draws))[:, None], draws), 1)
        num = (weights @ numerator.reshape(len(ids), -1)).reshape(len(draws), *numerator.shape[1:])
        den = (weights @ present.reshape(len(ids), -1)).reshape(len(draws), *present.shape[1:])
        rates.append(np.divide(num, den[..., None], out=np.zeros_like(num), where=den[..., None] > 0))
    rates = np.concatenate(rates)

    def values(arr, policy, field):
        p, f = policies.index(policy), fields.index(field)
        result = {view: arr[..., j, p, f] for j, view in enumerate(views)}
        if dataset == "heldout":
            result.update(aggregates(result))
        return result

    result = []
    for field in fields:
        gp, gb = values(point, "RelGate-0.50", field), values(rates, "RelGate-0.50", field)
        for baseline in ("Keep-Name", "Length-64"):
            bp, bb = values(point, baseline, field), values(rates, baseline, field)
            for metric in gp:
                delta = gb[metric] - bb[metric]
                low, high = np.quantile(delta, [.025, .975])
                result.append(dict(model=model, dataset=dataset, split=split, field=field, metric=metric,
                                   comparison="RelGate-0.50 minus " + baseline, delta_pp=float(100 * (gp[metric] - bp[metric])),
                                   ci_low_pp=float(100 * low), ci_high_pp=float(100 * high), replicates=REPS,
                                   resampled_request_ids=len(ids), seed=2026))
        if dataset == "heldout":
            for seed in (*SEEDS, "3-seed mean"):
                ps = [f"Matched-0.50-seed{s}" for s in SEEDS] if seed == "3-seed mean" else [f"Matched-0.50-seed{seed}"]
                bp = np.mean([values(point, p, field)["mixed_pool"] for p in ps], axis=0)
                bb = np.mean([values(rates, p, field)["mixed_pool"] for p in ps], axis=0)
                low, high = np.quantile(gb["mixed_pool"] - bb, [.025, .975])
                result.append(dict(model=model, dataset=dataset, split=split, field=field, metric="mixed_pool",
                                   comparison=f"RelGate-0.50 minus Matched-0.50 {seed}", delta_pp=float(100 * (gp["mixed_pool"] - bp)),
                                   ci_low_pp=float(100 * low), ci_high_pp=float(100 * high), replicates=REPS,
                                   resampled_request_ids=len(ids), seed=2026))
    print("BOOTSTRAP", model, dataset, split, flush=True)
    return result


def plot(detail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    lookup = {(r["model"], r["policy"], r["view"]): r["tool_acc"] * 100 for r in detail}
    colors = {"Keep-Name": "#59636F", "Always-Mask": "#B67528", "Length-64": "#1C8279", "RelGate-0.50": "#BD4660"}
    markers = dict(zip(POLICIES, ("o", "s", "^", "D")))
    planes = (("Coarse-Desc", "Name-Swapped"), ("Distractor-Swapped", "Target-Swapped"))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.3), dpi=180)
    frontier = []
    for col, model in enumerate(MODELS):
        for row, (xv, yv) in enumerate(planes):
            ax = axes[row, col]
            points = {p: (lookup[model, p, xv], lookup[model, p, yv]) for p in POLICIES}
            non = {p: (x, y) for p, (x, y) in points.items() if not any(u >= x and v >= y and (u > x or v > y) for u, v in points.values())}
            for p, (x, y) in points.items():
                frontier.append(dict(model=model, x_view=xv, y_view=yv, policy=p, x=x, y=y, nondominated=p in non))
                ax.scatter(x, y, c=colors[p], marker=markers[p], s=68, zorder=3)
            ordered = sorted(non.values())
            if len(ordered) > 1:
                ax.plot(*zip(*ordered), color="#999999", lw=1, zorder=1)
            ax.set(xlabel=xv + " (%)", ylabel=yv + " (%)")
            if row == 0:
                ax.set_title(model.replace("-Instruct", ""), fontsize=10)
            ax.margins(.2)
            ax.grid(color="#e4e6e8", lw=.6)
            ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], marker=markers[p], color=colors[p], linestyle="none", label=p) for p in POLICIES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Frozen confirmation: original 334-request holdout", fontsize=14)
    fig.tight_layout(rect=(0, .06, 1, .95))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(DEST / f"dual_pareto.{ext}", bbox_inches="tight")
    plt.close(fig)
    csvwrite("empirical_nondominated_points.csv", frontier)


def efficiency():
    table = []
    for model in MODELS:
        report = json.loads((DEST / f"raw/{model}/full.report.json").read_bytes())
        table.append(dict(model=model, stage="full_evaluation_batched_throughput", n=report["count"],
                          mean_ms=report["average_amortized_ms_per_prompt"], peak_torch_gib="",
                          peak_device_gib=report["peak_gpu_memory_gib_nvidia_smi"], cache_misses="",
                          device_peak_scope="full caller evaluation",
                          backend="vLLM BF16; max_num_seqs=16, queued128; amortized, not online latency"))
    for model in MODELS[2:]:
        report = json.loads((DEST / f"microbench/{model}/report.json").read_bytes())
        for row in report["stages"]:
            table.append(dict(model=model, stage=row["stage"], n=50, mean_ms=row["mean_ms_per_request"],
                              peak_torch_gib=row["peak_torch_allocated_gib"], peak_device_gib=report["peak_device_gib_monitor_all_stages"],
                              cache_misses=row["candidate_cache_misses"],
                              device_peak_scope="whole benchmark across stages; not a stage-specific device peak",
                              backend=report["backend"]))
    csvwrite("efficiency.csv", table)


def cross_stage(detail):
    oldpath = ROOT / "research_outputs/step_b_plugplay/eight_view_metrics.csv"
    with oldpath.open(encoding="utf-8", newline="") as f:
        old = list(csv.DictReader(f))
    names = {"Hammer-Repro 1.5B": MODELS[0], "B2 1.5B": MODELS[1], MODELS[2]: MODELS[2], MODELS[3]: MODELS[3]}
    output = []
    for stage, records in (("exploration197", old), ("independent_holdout334", detail)):
        for model in MODELS:
            for policy in POLICIES:
                values = {r["view"]: float(r["tool_acc"]) for r in records
                          if names.get(r["model"], r["model"]) == model and r["policy"] == policy}
                output.append(dict(stage=stage, model=model, policy=policy,
                                   shared_five_standard_macro=float(np.mean([values[v] for v in STANDARD])),
                                   shared_seven_view_macro=float(np.mean([values[v] for v in VIEWS])),
                                   standard_views=";".join(STANDARD), seven_views=";".join(VIEWS)))
    csvwrite("cross_stage_shared_view_metrics.csv", output)


def report(detail, summary, bfdetail, contrasts):
    lines = ["# Phase C2 Final Confirmation", "", "Original334 heldout; seven frozen views; Desc-Degraded N/A. RelGate-3Way-v2, tau=0.50. No training or tuning.", "",
             "## Heldout Tool Accuracy (%)", "", "| Caller / Policy | Clean | Coarse | Name Swap | Target | Distractor | Verb Flip | Verbose |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for policy in POLICIES:
            rates = {r["view"]: r["tool_acc"] for r in detail if r["model"] == model and r["policy"] == policy}
            lines.append("| " + model + " / " + policy + " | " + " | ".join(f"{rates[v]*100:.2f}" for v in VIEWS) + " |")
    lines.extend(["", "## Equal-Dose Selectivity", "", "Request-ID paired bootstrap, 20,000 draws, seed2026; three fixed random seeds averaged within request. Percentage points.", "",
                  "| Caller | RelGate minus matched random | 95% CI |", "|---|---:|---:|"])
    for r in contrasts:
        if r["dataset"] == "heldout" and r["field"] == "tool_correct" and r["comparison"].endswith("3-seed mean"):
            lines.append(f"| {r['model']} | {r['delta_pp']:+.2f} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] |")
    lines.extend(["", "## Clean Change", "", "CI spanning zero is uncertainty and is not evidence of no performance loss.", "",
                  "| Caller | RelGate minus Keep | 95% CI |", "|---|---:|---:|"])
    for r in contrasts:
        if r["dataset"] == "heldout" and r["metric"] == "Clean" and r["field"] == "tool_correct" and r["comparison"].endswith("Keep-Name"):
            lines.append(f"| {r['model']} | {r['delta_pp']:+.2f} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] |")
    lines.extend(["", "## BFCL Official AST Accuracy (%)", "", "Both official schema/gold anomalies remain included. Full per-split results and paired intervals are in CSV.", "",
                  "| Caller / Policy | Original | Name Swapped | Target Verb Flip |", "|---|---:|---:|---:|"])
    for model in MODELS:
        for policy in POLICIES:
            rates = {r["view"]: r["AST_accuracy"] for r in bfdetail if r["model"] == model and r["policy"] == policy and r["split"] == "pooled"}
            lines.append("| " + model + " / " + policy + " | " + " | ".join(f"{rates[v]*100:.2f}" for v in BFVIEWS) + " |")
    lines.extend(["", "## Strong Baseline and Generalization", "",
                  "Heldout Tool Accuracy changes versus Length-64, with request-ID paired 95% intervals. Percentage points.", "",
                  "| Caller | View / aggregate | Change | 95% CI |", "|---|---|---:|---:|"])
    for r in contrasts:
        if (r["dataset"] == "heldout" and r["field"] == "tool_correct"
                and r["comparison"].endswith("Length-64")
                and r["metric"] in ("mixed_pool", "shared_five_standard_macro", "seven_view_macro", "Verbose-Coarse", "Verb-Flip-Target")):
            lines.append(f"| {r['model']} | {r['metric']} | {r['delta_pp']:+.2f} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] |")
    lines.extend(["", "## BFCL Paired Comparisons", "",
                  "Pooled multiple/live_multiple official AST accuracy; all registered view denominators retained. Percentage points.", "",
                  "| Caller | View | Comparison | Change | 95% CI |", "|---|---|---|---:|---:|"])
    for r in contrasts:
        if r["dataset"] == "bfcl" and r["split"] == "pooled" and r["field"] == "strict_correct":
            lines.append(f"| {r['model']} | {r['metric']} | {r['comparison']} | {r['delta_pp']:+.2f} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] |")
    lines.extend(["", "## Empirical Pareto Result", "",
                  "RelGate is nondominated on both registered planes for Qwen2.5-7B. Length-64 dominates RelGate on both planes for Hammer-Repro, B2, and Llama. These are observed point estimates, not confidence statements.", "",
                  "## Same-Load Online Timing", "",
                  "50 fixed exploration Clean requests, batch=1 request, no cross-request batching; 381 candidate occurrences / 159 unique candidates. Each uncached request scores its candidates in one full-label forward pass. Stream caches start empty; these are not fully prepopulated cache timings.", "",
                  "| Caller | Caller only ms/req | Gate cold ms/req | Gate stream ms/req | Combined cold ms/req | Combined stream ms/req | Caller / combined Torch peak GiB |", "|---|---:|---:|---:|---:|---:|---:|"])
    for model in MODELS[2:]:
        bench = json.loads((DEST / f"microbench/{model}/report.json").read_bytes())
        stages = {r["stage"]: r for r in bench["stages"]}
        stage_order = ("caller_only_same_frozen_gate_prompts", "gate_only_cold_each_request", "gate_only_stream_cache_empty_at_start", "combined_cold_each_request", "combined_stream_cache_empty_at_start")
        timing = " | ".join(f"{stages[s]['mean_ms_per_request']:.1f}" for s in stage_order)
        lines.append(f"| {model} | {timing} | {stages[stage_order[0]]['peak_torch_allocated_gib']:.2f} / {stages[stage_order[3]]['peak_torch_allocated_gib']:.2f} |")
        assert all(r["batched_vs_reference_mask_discrepancies"] == 0 for r in bench["stages"])
    lines.extend(["", "Gate-only peak Torch allocation: 4.56 GiB. All measured mini-batch mask discrepancies: zero. Measured stages totaled about 215 seconds (Qwen) and 227 seconds (Llama), exceeding the original approximate 30-second budget. Loading and warmup are excluded. Whole-benchmark device peaks and all per-request measurements are retained separately; device peaks are not stage-specific.", "",
                  "## Completion and Artifacts", "",
                  "All four callers returned 11,945 unique predictions each; 84,104 logical evaluation units were scored. Raw output IDs, counts and SHA-256 were checked. Both online benchmarks returned 250 timing records each. The remote shutdown command was sent after return verification and before CPU analysis; billing status requires confirmation in the AutoDL console.", "",
                  "Tables: heldout_seven_view_metrics.csv, heldout_aggregate_metrics.csv, bfcl_official_AST_metrics.csv, paired_bootstrap_20000.csv, efficiency.csv. Figures: dual_pareto.png/pdf/svg. Manifest: FINAL_SHA256SUMS.txt."])
    lines.extend(["", "## Limitations", "", "- BFCL multiple/live_multiple only; not a full BFCL leaderboard evaluation.",
                  "- Seven-view confirmation excludes unavailable independent Desc-Degraded annotations; cross-stage standard macros use the five shared standard views.",
                  "- Pointwise percentile intervals have no multiplicity correction; joint original-ID resampling preserves within-request dependence across views and strategies.",
                  "- Timing uses Transformers online batch1 and vLLM batched evaluation; throughput is not online latency. Full online timings and numerical mini-batch mask discrepancies are retained.",
                  "- Fixed three-seed random controls quantify the registered realization, not uncertainty over every possible random mask.",
                  "- Failure examples are deterministic descriptive samples and do not estimate failure prevalence."])
    (DEST / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    records = score_all()
    detail, summary, bfdetail = tables(records)
    contrasts = []
    for model in MODELS:
        contrasts.extend(bootstrap(records, model, "heldout"))
        for split in ("multiple", "live_multiple", "pooled"):
            contrasts.extend(bootstrap(records, model, "bfcl", split))
    csvwrite("paired_bootstrap_20000.csv", contrasts)
    plot(detail)
    efficiency()
    cross_stage(detail)
    report(detail, summary, bfdetail, contrasts)
    (DEST / "analysis_protocol.json").write_bytes(encode({"replicates": REPS, "seed": 2026, "unit": "original request ID; joint across all views/policies/seeds within model and dataset/split", "interval": "pointwise 95% percentile; no multiplicity correction", "gate": "frozen tau0.50", "heldout_views": VIEWS, "shared_standard_macro_views": STANDARD, "Desc-Degraded": "N/A", "BFCL": REVISION}))
    code = DEST / "code"
    code.mkdir(exist_ok=True)
    for name in ("step_c2_freeze.py", "step_c2_score.py", "step_c2_schedule.py", "step_c2_caller.py",
                 "step_c2_bench_prepare.py", "step_c2_microbench.py", "step_c2_control.py",
                 "step_c2_analyze.py", "step_c_bfcl_adapter.py", "step_b_caller_runner.py"):
        shutil.copyfile(ROOT / "scripts" / name, code / name)
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and p.name != "FINAL_SHA256SUMS.txt")
    manifest = DEST / "FINAL_SHA256SUMS.txt"
    manifest.write_text("".join(f"{sha(p)}  {p.relative_to(DEST).as_posix()}\n" for p in files))
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(DEST / name) == digest
    print("FINAL_ANALYSIS_COMPLETE", len(records), "logical units;", len(files), "SHA256 verified files", flush=True)


if __name__ == "__main__":
    main()
