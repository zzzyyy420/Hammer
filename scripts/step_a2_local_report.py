"""Build the frozen Phase A2 local CPU report from verified predictions."""
import copy
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pareto_experiment import predictions
from relgate_cpu_audit import hm
from routeb_stage1 import rows, remap, split_prompt
from routeb_step2a import outcome
from routeb_validation import R, gold, tid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs/step_a2_local_report"
STEP = ROOT / "research_outputs/step_a2_incremental"
PROBES = ROOT / "research_outputs/step_a_v1/probes.jsonl"
V2 = ROOT / "research_outputs/relgate_v2_final"
V1 = ROOT / "research_outputs/relgate_v1/report/predictions"
BASE = ROOT / "research_outputs/routeB_step2b_v1/report/predictions"
VIEWS = ["Clean", "Coarse-Desc", "Name-Swapped", "Target-Swapped", "Distractor-Swapped", "Desc-Degraded"]
EXPECTED = dict(zip(VIEWS, (197, 197, 197, 182, 182, 165)))
POLICIES = ["Keep-Name", "Always-Mask", "Length-64", "Oracle", "RelGate-0.50", "RelGate-default", "RelGate-0.30"]
DISPLAY = {"Pure-NameMask": "Hammer-Repro (66.38% Func+Param Mask)", "B2": "B2"}


def check(path, meta, count):
    info = json.loads(meta.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == info["sha256"] and len(rows(path)) == count == info["count"], path


def table(name, data):
    assert data, name
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def by_key(path):
    result = {}
    for row in rows(path):
        key = row["id"], row["view"]
        assert key not in result, (path, key)
        result[key] = row
    return result


def score_case(case, raw, key):
    p = raw[case["prompt"]]
    result = outcome(dict(p, parsed_pred=remap(p["parsed_pred"], case["inverse"])), key)
    return dict(id=case["id"], view=case["view"], policy=case["policy"],
                tool_correct=result[0], strict_correct=result[1],
                masked_count=case["masked_count"], candidate_count=case["candidate_count"],
                target_masked=int(case["target_masked"]) if case["target_masked"] is not None else None)


def rate(seq, field):
    return sum(r[field] for r in seq) / len(seq) if seq else None


def metric(seq):
    n = len(seq)
    if not n:
        return dict(n=0, tool_acc=None, strict_match=None, candidate_mask_rate=None, target_mask_rate=None)
    target = [x for x in seq if x.get("target_masked") is not None]
    return dict(n=n, tool_acc=rate(seq, "tool_correct"), strict_match=rate(seq, "strict_correct"),
                candidate_mask_rate=sum(x["masked_count"] for x in seq) / sum(x["candidate_count"] for x in seq),
                target_mask_rate=rate(target, "target_masked"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(PROBES.read_bytes()).hexdigest() == "b7fa9b653252ae3db211837016fde002ca07c05acf747d10022c0bb485ac1ed6"
    check(STEP / "scores/3way.jsonl", STEP / "scores/3way.done.json", 666)
    check(STEP / "greedy_labels.jsonl", STEP / "greedy.done.json", 666)
    for model, n in (("B2", 3702), ("Pure-NameMask", 913)):
        parent = ROOT / "research_outputs/relgate_pareto/results"
        check(parent / f"{model}__step_a2.jsonl", parent / f"{model}__step_a2.done.json", n)

    probes = {(r["id"], r["view"]): r for r in rows(PROBES)}
    assert Counter(v for _, v in probes) == {"Verbose-Coarse": 197, "Verb-Flip-Target": 72}
    base_keys = gold()
    keys = dict(base_keys)
    for (sid, view), probe in probes.items():
        k = copy.deepcopy(base_keys["Clean", sid])
        k["view"] = view
        if view == "Verb-Flip-Target":
            k["ground_truth"][0]["name"] = probe["tools"][probe["g"]]["name"]
        keys[view, sid] = k

    scores = {r["id"]: r for r in rows(V2 / "scores/3way.jsonl")}
    new_scores = {r["id"]: r for r in rows(STEP / "scores/3way.jsonl")}
    assert not set(scores) & set(new_scores)
    scores.update(new_scores)
    labels = {r["id"]: r["raw"].strip() for r in rows(V2 / "greedy_labels.jsonl")}
    new_labels = {r["id"]: r["raw"].strip() for r in rows(STEP / "greedy_labels.jsonl")}
    assert not set(labels) & set(new_labels)
    labels.update(new_labels)

    computed, all_cases = {}, {}
    for model in ("B2", "Pure-NameMask"):
        cases = rows(STEP / f"{model}__step_a2.cases.jsonl")
        all_cases[model] = cases
        raw = predictions(model)
        assert all(c["prompt"] in raw for c in cases), model
        recs = {}
        for c in cases:
            key = c["id"], c["view"], c["policy"]
            assert key not in recs
            recs[key] = score_case(c, raw, keys[c["view"], c["id"]])
        computed[model] = recs

    # Previous reports already contain fully scored standard-view predictions.
    standard = {}
    for model in ("B2", "Pure-NameMask"):
        standard[model] = {}
        for policy in POLICIES:
            filename = f"{model}__{'Global-Length-64' if policy == 'Length-64' and model == 'Pure-NameMask' else policy}.jsonl"
            path = (V2 / "predictions" / filename) if model == "B2" else (V1 / filename)
            if not path.exists() and model == "Pure-NameMask" and policy in ("Keep-Name", "Always-Mask"):
                path = BASE / filename
            if path.exists():
                standard[model][policy] = by_key(path)
        if model == "Pure-NameMask":
            for policy in ("Keep-Name", "Always-Mask"):
                extra = by_key(BASE / f"{model}__{policy}.jsonl")
                standard[model][policy].update(extra)
            for policy in ("RelGate-0.50", "RelGate-default", "RelGate-0.30"):
                standard[model][policy] = {
                    (sid, view): row for (sid, view, p), row in computed[model].items()
                    if p == policy and view in VIEWS
                }
        for policy, data in standard[model].items():
            for (sid, view), row in data.items():
                assert row["id"] == sid and row["view"] == view

    source = {r["id"]: r for r in rows(R / "inputs/keep.jsonl")}
    source_cases = {(r["id"], r["view"]): r for r in rows(R / "cases.jsonl")}
    tools = {(sid, view): split_prompt(source[c["prompt"]])[1] for (sid, view), c in source_cases.items() if view in VIEWS}

    def actions(model, policy, sid, view):
        t = tools[sid, view]
        k = keys[view, sid]
        if policy == "Keep-Name": sel = [False] * len(t)
        elif policy == "Always-Mask": sel = [True] * len(t)
        elif policy == "Length-64": sel = [len(x.get("description") or "") > 64 for x in t]
        elif policy == "Oracle": sel = k["flags"]
        elif policy == "RelGate-default": sel = [labels[tid(x)[0]] == "CONTRADICTED" for x in t]
        else: sel = [scores[tid(x)[0]]["q"]["CONTRADICTED"] >= float(policy.split("-")[1]) for x in t]
        return sum(sel), len(sel), int(sel[k["g"]])

    detail = []
    full = []
    for model in ("Pure-NameMask", "B2"):
        for policy in POLICIES:
            records = standard[model].get(policy, {})
            by_view = {}
            for view in VIEWS:
                rr = [records[sid, v] for sid, v in source_cases if v == view and (sid, v) in records]
                if rr:
                    assert len(rr) == EXPECTED[view], (model, policy, view, len(rr))
                mask = [actions(model, policy, r["id"], view) for r in rr]
                m = dict(n=len(rr), tool_acc=rate(rr, "tool_correct"), strict_match=rate(rr, "strict_correct"),
                         candidate_mask_rate=sum(a for a, _, _ in mask) / sum(b for _, b, _ in mask) if mask else None,
                         target_mask_rate=sum(c for _, _, c in mask) / len(mask) if mask else None)
                by_view[view] = m
                detail.append(dict(model=DISPLAY[model], policy=policy, view=view, **m))
            a = by_view["Coarse-Desc"]["tool_acc"]
            b = by_view["Name-Swapped"]["tool_acc"]
            mixed = [by_view[v]["tool_acc"] for v in ("Target-Swapped", "Distractor-Swapped")]
            three = [by_view[v]["tool_acc"] for v in ("Clean", "Coarse-Desc", "Name-Swapped")]
            six = [by_view[v]["tool_acc"] for v in VIEWS]
            three_strict = [by_view[v]["strict_match"] for v in ("Clean", "Coarse-Desc", "Name-Swapped")]
            six_strict = [by_view[v]["strict_match"] for v in VIEWS]
            full.append(dict(model=DISPLAY[model], policy=policy, H_dual=hm(a, b) if None not in (a, b) else None,
                             three_view_HM=3/sum(1/x for x in three) if all(x is not None and x > 0 for x in three) else None,
                             mixed_mean=np.mean(mixed) if None not in mixed else None,
                             mixed_HM=hm(*mixed) if None not in mixed else None,
                             six_view_macro=np.mean(six) if None not in six else None,
                             three_view_strict_macro=np.mean(three_strict) if None not in three_strict else None,
                             six_view_strict_macro=np.mean(six_strict) if None not in six_strict else None,
                             complete_six=all(x is not None for x in six)))
    table("six_view_metrics.csv", detail)
    table("six_view_aggregate.csv", full)

    probe_detail = []
    for model in ("B2", "Pure-NameMask"):
        for policy in ("Keep-Name", "Always-Mask", "Length-64", "RelGate-0.50", "RelGate-default", "RelGate-0.30"):
            for view in ("Verbose-Coarse", "Verb-Flip-Target"):
                seq = [r for (sid, v, p), r in computed[model].items() if v == view and p == policy]
                assert len(seq) == (197 if view == "Verbose-Coarse" else 72)
                probe_detail.append(dict(model=DISPLAY[model], policy=policy, view=view, **metric(seq)))
    table("probe_end_to_end.csv", probe_detail)

    flip = [p for p in probes.values() if p["view"] == "Verb-Flip-Target"]
    verbose = [p for p in probes.values() if p["view"] == "Verbose-Coarse"]
    candidate = []
    def collect(view, p):
        for slot, t in enumerate(p["tools"]):
            key = tid(t)[0]
            candidate.append(dict(view=view, id=p["id"], slot=slot, name=t["name"], description_length=len(t.get("description") or ""),
                                  flipped=int(view == "Verb-Flip-Target" and slot == p["g"]), target=int(slot == p["g"]),
                                  score=scores[key]["q"]["CONTRADICTED"], greedy_label=labels[key],
                                  ell_S=scores[key]["ell"]["SUPPORTED"], ell_U=scores[key]["ell"]["UNVERIFIED"],
                                  ell_C=scores[key]["ell"]["CONTRADICTED"]))
    for p in flip: collect("Verb-Flip-Target", p)
    for p in verbose: collect("Verbose-Coarse", p)
    table("probe_candidate_scores.csv", candidate)
    flip_rows = [x for x in candidate if x["view"] == "Verb-Flip-Target"]
    y = [x["flipped"] for x in flip_rows]
    s = [x["score"] for x in flip_rows]
    gate_diag = []
    for policy in ("RelGate-default", "RelGate-0.30", "RelGate-0.50", "RelGate-0.70"):
        select = lambda x: x["greedy_label"] == "CONTRADICTED" if policy == "RelGate-default" else x["score"] >= float(policy.split("-")[1])
        hit = [x for x in flip_rows if x["flipped"]]
        distract = [x for x in flip_rows if not x["flipped"]]
        vc = [x for x in candidate if x["view"] == "Verbose-Coarse"]
        gate_diag.append(dict(policy=policy, flip_target_recall=np.mean([select(x) for x in hit]),
                              flip_distractor_fpr=np.mean([select(x) for x in distract]),
                              verbose_coarse_fpr=np.mean([select(x) for x in vc]),
                              flip_pool_auroc=roc_auc_score(y, s), flip_pool_pr_auc=average_precision_score(y, s)))
    table("probe_gate_diagnostics.csv", gate_diag)

    pairs = []
    for p in flip:
        old, new = p["original_name"], p["tools"][p["g"]]["name"]
        match = re.match(r"^([a-z]+)(?=(_|[A-Z]|$))", old)
        assert match, old
        old_verb = match.group(1)
        suffix = old[len(old_verb):]
        assert new.endswith(suffix), (old, new)
        new_verb = new[:-len(suffix)] if suffix else new
        pairs.append(dict(id=p["id"], original=old, flipped=new, old_verb=old_verb, new_verb=new_verb,
                          review_note="待人工审阅" if old_verb in {"list", "post", "open", "close", "send", "receive"} else ""))
    table("verb_flip_audit.csv", pairs)
    counts = Counter((x["old_verb"], x["new_verb"], x["review_note"]) for x in pairs)
    table("verb_flip_pair_counts.csv", [dict(old_verb=a, new_verb=b, review_note=c, n=n) for (a,b,c),n in sorted(counts.items())])

    intersection = {p["id"] for p in flip} & {sid for sid, view in source_cases if view == "Target-Swapped"}
    assert len(intersection) <= 72
    common = []
    for model in ("B2", "Pure-NameMask"):
        for policy in POLICIES:
            for view in ("Clean", "Target-Swapped", "Verb-Flip-Target"):
                if view == "Verb-Flip-Target":
                    seq = [r for (sid,v,p),r in computed[model].items() if v == view and p == policy and sid in intersection]
                    all_72 = [r for (sid,v,p),r in computed[model].items() if v == view and p == policy]
                else:
                    seq = []
                    for (sid,v),r in standard[model].get(policy, {}).items():
                        if v == view and sid in intersection:
                            masked, total, target = actions(model, policy, sid, view)
                            seq.append(dict(r, masked_count=masked, candidate_count=total, target_masked=target))
                    all_72 = []
                common.append(dict(model=DISPLAY[model], policy=policy, view=view, n_inter=len(intersection),
                                   **metric(seq), full_flip_n=len(all_72), full_flip_tool_acc=rate(all_72, "tool_correct"),
                                   full_flip_strict=rate(all_72, "strict_correct")))
    table("verb_flip_common_denominator.csv", common)

    occ = rows(R / "report/gate_candidate_occurrences.jsonl")
    strata = defaultdict(list)
    for o in occ:
        if o["view"] not in ("Clean", "Coarse-Desc", "Name-Swapped"): continue
        t = tools[o["id"], o["view"]][o["slot"]]
        length = len(t.get("description") or "")
        if o["view"] == "Coarse-Desc": label = "Short-Coarse"
        elif o["view"] == "Clean": label = "Short-Specific Match" if length <= 64 else "Verbose-Specific Match"
        else: label = "Short-Specific Conflict" if length <= 64 else "Verbose-Specific Conflict"
        strata[label].append(dict(score=scores[o["triple"]]["q"]["CONTRADICTED"], length=length,
                                  greedy=labels[o["triple"]]))
    strata["Verbose-Coarse"] = [dict(score=x["score"], length=x["description_length"], greedy=x["greedy_label"])
                                 for x in candidate if x["view"] == "Verbose-Coarse"]
    expected_strata = {"Short-Coarse":1414, "Short-Specific Match":447, "Short-Specific Conflict":447,
                       "Verbose-Coarse":1414, "Verbose-Specific Match":967, "Verbose-Specific Conflict":967}
    assert {k:len(v) for k,v in strata.items()} == expected_strata
    length_rows = []
    for label, seq in strata.items():
        arr = [x["score"] for x in seq]
        length_rows.append(dict(slice=label, n=len(seq), length64_mask_rate=np.mean([x["length"] > 64 for x in seq]),
                                relgate_050_mask_rate=np.mean([x["score"] >= .5 for x in seq]),
                                relgate_default_mask_rate=np.mean([x["greedy"] == "CONTRADICTED" for x in seq]),
                                relgate_030_mask_rate=np.mean([x["score"] >= .3 for x in seq]),
                                score_p10=np.quantile(arr,.1), score_p25=np.quantile(arr,.25),
                                score_median=np.median(arr), score_p75=np.quantile(arr,.75), score_p90=np.quantile(arr,.9)))
    table("length_stratified_semantics.csv", length_rows)

    random_rows = []
    for policy in ("Random-0.25", "Random-0.50", "Random-0.75"):
        for view in ("Clean", "Coarse-Desc", "Name-Swapped"):
            vals = [metric([r for (sid,v,p),r in computed["B2"].items() if v == view and p == f"{policy}-seed{seed}"])
                    for seed in (42,123,2026)]
            assert all(x["n"] == 197 for x in vals)
            random_rows.append(dict(policy=policy, view=view, n=197,
                                    tool_mean=np.mean([x["tool_acc"] for x in vals]), tool_sd=np.std([x["tool_acc"] for x in vals],ddof=1),
                                    strict_mean=np.mean([x["strict_match"] for x in vals]), strict_sd=np.std([x["strict_match"] for x in vals],ddof=1),
                                    actual_mask_mean=np.mean([x["candidate_mask_rate"] for x in vals])))
    table("b2_random_mask_curve.csv", random_rows)

    a = standard["B2"]["RelGate-0.50"]
    b = standard["B2"]["Length-64"]
    coarse_ids = {sid for sid,view in source_cases if view == "Coarse-Desc"}
    false_ids = {o["id"] for o in occ if o["view"] == "Coarse-Desc" and not o["swapped"]
                 and o["slot"] == keys[o["view"],o["id"]]["g"] and scores[o["triple"]]["q"]["CONTRADICTED"] >= .5}
    assert len(coarse_ids) == 197 and len(false_ids) == 13
    decomposition = []
    for group, ids in (("all_197",coarse_ids),("false_mask_13",false_ids),("other_184",coarse_ids-false_ids)):
        for field in ("tool_correct", "strict_correct"):
            c = Counter((int(b[sid,"Coarse-Desc"][field]),int(a[sid,"Coarse-Desc"][field])) for sid in ids)
            decomposition.append(dict(group=group, metric=field, n=len(ids), both_correct=c[1,1], both_wrong=c[0,0],
                                      length_only=c[1,0], relgate_only=c[0,1], relgate_net=c[0,1]-c[1,0]))
    table("coarse_paired_decomposition.csv", decomposition)
    false_detail = []
    for sid in sorted(false_ids):
        t = tools[sid,"Coarse-Desc"][keys["Coarse-Desc",sid]["g"]]
        pars = t.get("parameters") or {}
        has_param_desc = any((v.get("description") or "").strip() for v in pars.values() if isinstance(v,dict))
        false_detail.append(dict(id=sid, name=t["name"], has_parameter_description=has_param_desc,
                                 length_tool_correct=b[sid,"Coarse-Desc"]["tool_correct"], relgate_tool_correct=a[sid,"Coarse-Desc"]["tool_correct"],
                                 length_strict=b[sid,"Coarse-Desc"]["strict_correct"], relgate_strict=a[sid,"Coarse-Desc"]["strict_correct"]))
    table("coarse_false_mask_13.csv", false_detail)

    def point(model, policy, view):
        return next(r["tool_acc"] for r in detail if r["model"] == DISPLAY[model] and r["policy"] == policy and r["view"] == view)
    fig, axes = plt.subplots(1,2,figsize=(12,5.4))
    fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.26,wspace=.25)
    colors = {"B2":"#147d72", "Pure-NameMask":"#a74c3c"}
    for ax, (xv,yv,title) in zip(axes, (("Coarse-Desc","Name-Swapped","Main plane"),
                                             ("Distractor-Swapped","Target-Swapped","Local swap plane"))):
        for model in ("B2", "Pure-NameMask"):
            for policy, marker in (("Keep-Name","o"),("Always-Mask","s"),("Length-64","^"),("RelGate-0.50","D"),
                                   ("RelGate-default","x"),("RelGate-0.30","+")):
                x,y = point(model,policy,xv),point(model,policy,yv)
                ax.scatter(100*x,100*y,c=colors[model],marker=marker,s=58)
        if title == "Main plane":
            for p in ("Random-0.25","Random-0.50","Random-0.75"):
                x = next(r["tool_mean"] for r in random_rows if r["policy"]==p and r["view"]==xv)
                y = next(r["tool_mean"] for r in random_rows if r["policy"]==p and r["view"]==yv)
                ax.scatter(100*x,100*y,c="#4d7096",marker="v",s=45)
                ax.annotate(p.replace("Random-","p="),(100*x,100*y),xytext=(4,5),textcoords="offset points",fontsize=8)
        ax.set(xlabel=xv+" Tool Acc (%)",ylabel=yv+" Tool Acc (%)",title=title)
        ax.grid(alpha=.2)
    legend = [Line2D([0],[0],color=c,marker="o",linestyle="None",label=n,markersize=7)
              for n,c in (("B2",colors["B2"]),("Hammer-Repro",colors["Pure-NameMask"]))]
    legend += [Line2D([0],[0],color="#555555",marker=m,linestyle="None",label=n,markersize=7)
               for n,m in (("Keep-Name","o"),("Always-Mask","s"),("Length-64","^"),
                           ("RelGate-0.50","D"),("RelGate-default","x"),("RelGate-0.30","+"))]
    legend.append(Line2D([0],[0],color="#4d7096",marker="v",linestyle="None",label="B2 Random-Mask",markersize=7))
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.03),ncol=5,frameon=False,fontsize=8)
    fig.savefig(OUT / "dual_pareto.png",dpi=220)
    fig.savefig(OUT / "dual_pareto.svg")
    plt.close(fig)

    pct = lambda x: "NA" if x is None else f"{100*float(x):.2f}"
    lines = ["# Phase A2 local CPU report", "", "All metrics use the frozen exploratory requests. No held-out 334 examples, GPU, or AutoDL instance were accessed.",
             "Hammer-Repro means the original Pure-NameMask (66.38% Func+Param Mask). Main work point: RelGate-0.50; default and 0.30 are auxiliary.",
             "Oracle is defined only on Target-Swapped and Distractor-Swapped; its six-view aggregates are unavailable.",
             "H_dual = HM(Coarse-Desc, Name-Swapped). Three-view HM uses Clean, Coarse-Desc, Name-Swapped.",
             "Swap status is a synthetic proxy for semantic contradiction, not a human validated conflict label.",
             "", "## 1. Six-view cross-training comparison", "",
             "Tool Acc (%); Strict Match and mask rates for every cell are in six_view_metrics.csv.", "",
             "| Model | Policy | Clean | Coarse | Name-Swapped | Target | Distractor | Degraded | H_dual | 3-view HM | Mixed mean | Mixed HM | 6-view mean |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for aggregate in full:
        model, policy = aggregate["model"], aggregate["policy"]
        rr = {r["view"]:r for r in detail if r["model"]==model and r["policy"]==policy}
        lines.append("| " + " | ".join([model, policy] + [pct(rr[v]["tool_acc"]) for v in VIEWS] +
                     [pct(aggregate[k]) for k in ("H_dual","three_view_HM","mixed_mean","mixed_HM","six_view_macro")]) + " |")
    lines += ["", "## 2. Verbose-Coarse", "",
              "Candidate false mask rates use all 1,414 unexchanged candidates; Length-64 masks all of them.", "",
              "| Gate | False mask rate (%) |", "|---|---:|"]
    for r in gate_diag:
        lines.append(f"| {r['policy']} | {pct(r['verbose_coarse_fpr'])} |")
    lines += ["", "| Model | Policy | Tool Acc (%) | Strict (%) | Candidate mask (%) | Target mask (%) |",
              "|---|---|---:|---:|---:|---:|"]
    for r in probe_detail:
        if r["view"] == "Verbose-Coarse":
            lines.append("| " + " | ".join([r["model"],r["policy"]] + [pct(r[k]) for k in ("tool_acc","strict_match","candidate_mask_rate","target_mask_rate")]) + " |")
    lines += ["", "## 3. Verb-Flip-Target", "",
              f"The request-ID intersection with Target-Swapped is {len(intersection)}/72. Full-probe metrics use all 72 requests; the common-denominator table uses this exact intersection.",
              f"Candidate pool AUROC = {gate_diag[0]['flip_pool_auroc']:.4f}; PR-AUC = {gate_diag[0]['flip_pool_pr_auc']:.4f}.",
              "Unflipped distractor false-mask rates and 72-row verb audit are in the linked CSV files. Ambiguous pairs are marked for human review.", "",
              "| Gate | Flipped target recall (%) | Unflipped distractor FPR (%) |",
              "|---|---:|---:|"]
    for r in gate_diag:
        lines.append(f"| {r['policy']} | {pct(r['flip_target_recall'])} | {pct(r['flip_distractor_fpr'])} |")
    lines += ["", "| Verb pair | Count | Review |", "|---|---:|---|"]
    for r in sorted(counts.items()):
        (old,new,note),n = r
        lines.append(f"| {old} -> {new} | {n} | {note} |")
    lines += ["", "| Model | Policy | Clean Tool/Strict | Target Tool/Strict | Flip Tool/Strict |",
              "|---|---|---:|---:|---:|"]
    for model in DISPLAY.values():
        for policy in POLICIES:
            rr = {r["view"]:r for r in common if r["model"]==model and r["policy"]==policy}
            cell = lambda v: f"{pct(rr[v]['tool_acc'])}/{pct(rr[v]['strict_match'])}"
            lines.append(f"| {model} | {policy} | {cell('Clean')} | {cell('Target-Swapped')} | {cell('Verb-Flip-Target')} |")
    lines += ["", "## 4. Length-stratified semantic robustness", "",
              "Specific match/conflict are natural description-length strata. Verbose-Coarse uses domain-free I/O boilerplate; this is not a causal isolation.", "",
              "| Slice | n | Length-64 mask (%) | RelGate-0.50 mask (%) | Default mask (%) | RelGate-0.30 mask (%) | s P10/P50/P90 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in length_rows:
        lines.append("| " + " | ".join([r["slice"],str(r["n"])] + [pct(r[k]) for k in ("length64_mask_rate","relgate_050_mask_rate","relgate_default_mask_rate","relgate_030_mask_rate")] +
                     [f"{r['score_p10']:.3f}/{r['score_median']:.3f}/{r['score_p90']:.3f}"]) + " |")
    lines += ["", "## 5. B2 global Random-Mask", "",
              "Means and sample SD across seeds 42, 123, 2026; fixed candidate-level rates per policy across all views.", "",
              "| Policy | View | Tool Acc mean +/- SD (%) | Strict mean +/- SD (%) | Actual candidate mask (%) |",
              "|---|---|---:|---:|---:|"]
    for r in random_rows:
        lines.append(f"| {r['policy']} | {r['view']} | {pct(r['tool_mean'])} +/- {pct(r['tool_sd'])} | {pct(r['strict_mean'])} +/- {pct(r['strict_sd'])} | {pct(r['actual_mask_mean'])} |")
    lines += ["", "## 6. Coarse-Desc paired outcomes and Pareto plot", "",
              "B2 RelGate-0.50 versus Length-64 on all 197 requests:", "",
              "| Group | Metric | Both correct | Both wrong | Length only | RelGate only | RelGate net |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in decomposition:
        lines.append("| " + " | ".join(str(r[k]) for k in ("group","metric","both_correct","both_wrong","length_only","relgate_only","relgate_net")) + " |")
    lines += ["", f"Of 13 falsely masked gold names, {sum(x['has_parameter_description'] for x in false_detail)} have parameter descriptions and {13-sum(x['has_parameter_description'] for x in false_detail)} do not.",
              "The two-panel figure is dual_pareto.png / dual_pareto.svg. It shows observed exploratory coordinates; connecting lines are omitted.", "",
              "## Artifacts", "", "All CSVs, plot files, and this report are listed in SHA256SUMS.txt.", ""]
    (OUT / "report.md").write_text("\n".join(lines),encoding="utf-8")
    manifest = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), intersection=len(intersection), candidate_n=len(candidate),
                          false_parameter_description=sum(x["has_parameter_description"] for x in false_detail),
                          false_total=len(false_detail), files=len(manifest)),indent=2))


if __name__ == "__main__":
    main()
