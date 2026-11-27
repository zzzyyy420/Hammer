"""Assemble frozen Step B plug-and-play prompt schedules without scoring gates."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from pareto_experiment import predictions
from pareto_policies import apply_name_mask, dose_matched_mask
from routeb_stage1 import dump, indexed, msg, rows, split_prompt
from routeb_step2a import fingerprint
from routeb_validation import R, tid

H = Path(__file__).resolve().parents[1]
OUT = H / "research_outputs/step_b_plugplay"
STEP = H / "research_outputs/step_a2_incremental"
V2 = H / "research_outputs/relgate_v2_final"
PROBES = H / "research_outputs/step_a_v1/probes.jsonl"
EXPECTED_PROBE_SHA = "b7fa9b653252ae3db211837016fde002ca07c05acf747d10022c0bb485ac1ed6"
VIEWS = {"Clean", "Coarse-Desc", "Name-Swapped", "Target-Swapped", "Distractor-Swapped", "Desc-Degraded"}


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def restore_names(value, inverse):
    if isinstance(value, dict):
        return {k: (inverse.get(v, v) if k in {"name", "tool"} and isinstance(v, str)
                    else restore_names(v, inverse)) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_names(v, inverse) for v in value]
    return value


def main():
    assert hashlib.sha256(PROBES.read_bytes()).hexdigest() == EXPECTED_PROBE_SHA
    source = indexed(rows(R / "inputs/keep.jsonl"))
    cases = [r for r in rows(R / "cases.jsonl") if r["view"] in VIEWS]
    keys = {(r["id"], r["view"]): r for r in rows(R / "cases.jsonl")}
    clean = {r["id"]: r for r in cases if r["view"] == "Clean"}
    probe = rows(PROBES)
    assert Counter(x["view"] for x in probe) == {"Verb-Flip-Target": 72, "Verbose-Coarse": 197}
    scores = indexed(rows(V2 / "scores/3way.jsonl"))
    new_scores = indexed(rows(STEP / "scores/3way.jsonl"))
    # The main score file has grown across completed incremental runs.  Validate
    # identity and disjointness instead of pinning the historical row count.
    assert scores and new_scores and not set(scores) & set(new_scores)
    scores.update(new_scores)
    with (V2 / "b2_actions.csv").open("r", encoding="utf-8-sig", newline="") as f:
        action_rows = list(csv.DictReader(f))
    assert action_rows and {"policy", "view", "n", "masked", "candidates"} <= set(action_rows[0])
    actions = {(r["policy"], r["view"]): r for r in action_rows}
    # Exact candidate actions are taken from the already-frozen per-request schedule.
    frozen_cases = {(r["id"], r["view"], r["policy"]): r for r in rows(STEP / "B2__step_a2.cases.jsonl")}
    final_cases = {(r["id"], r["view"], r["policy"]): r for r in rows(V2 / "b2_cases.jsonl")}
    matched = {(r["id"], r["view"], r["policy"]): r for r in rows(V2 / "b2_matched_cases.jsonl")}
    assert len(matched) > 0
    assert ("RelGate-0.50", "Target-Swapped") in actions
    assert ("RelGate-0.50", "Distractor-Swapped") in actions

    required_score_ids = set()
    for case in cases:
        required_score_ids.update(case["triples"])
    for row in probe:
        required_score_ids.update(tid(t)[0] for t in row["tools"])
    assert required_score_ids <= set(scores), (len(required_score_ids), len(required_score_ids - set(scores)))

    requests, schedules = {}, []
    main_prompt_ids = set()
    model_cells = {}

    def add_request(view, sid, tools, aliases, before, after, selected, policy):
        masked_tools, inverse = apply_name_mask(tools, aliases, selected)
        prompt_messages = msg(before + dump(masked_tools) + after) if any(selected) else msg(before + dump(tools) + after)
        # Preserve original system/user message structure for unmodified prompts.
        prompt_id = fingerprint({"messages": prompt_messages})
        requests.setdefault(prompt_id, {"id": prompt_id, "view": view, "messages": prompt_messages})
        return {"id": sid, "view": view, "policy": policy, "prompt": prompt_id,
                "inverse": inverse, "masked_count": sum(selected), "candidate_count": len(tools)}

    all_models = ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
    for model in all_models:
        cells = []
        for case in cases:
            before, tools, after = split_prompt(source[case["prompt"]])
            ids = case["triples"]
            choices = [("Keep-Name", [False] * len(tools)),
                       ("Always-Mask", [True] * len(tools)),
                       ("Length-64", [len(t.get("description") or "") > 64 for t in tools]),
                       ("RelGate-0.50", [scores[k]["q"]["CONTRADICTED"] >= .50 for k in ids])]
            for policy, selected in choices:
                cell = add_request(case["view"], case["id"], tools, case["aliases"],
                                   before, after, selected, policy)
                cells.append(cell)
                main_prompt_ids.add(cell["prompt"])
        for row in probe:
            clean_case = clean[row["id"]]
            before, tools, after = split_prompt({"messages": msg(row["instruction"])})
            assert tools == row["tools"]
            choices = [("Keep-Name", [False] * len(tools)),
                       ("Always-Mask", [True] * len(tools)),
                       ("Length-64", [len(t.get("description") or "") > 64 for t in tools]),
                       ("RelGate-0.50", [scores[tid(t)[0]]["q"]["CONTRADICTED"] >= .50 for t in tools])]
            for policy, selected in choices:
                cell = add_request(row["view"], row["id"], tools, clean_case["aliases"],
                                   before, after, selected, policy)
                cells.append(cell)
                main_prompt_ids.add(cell["prompt"])
        model_cells[model] = cells

    mixed_case_ids = {(r["id"], r["view"]) for r in cases if r["view"] in {"Target-Swapped", "Distractor-Swapped"}}
    matched_seeds = (42, 123, 2026)
    matched_cells = []
    matched_action_sources = Counter()
    for sid, view in sorted(mixed_case_ids):
        case = next(x for x in cases if x["id"] == sid and x["view"] == view)
        before, tools, after = split_prompt(source[case["prompt"]])
        action_key = (sid, view, "RelGate-0.50")
        if action_key in frozen_cases:
            action = frozen_cases[action_key]
            matched_action_sources["step_a2_incremental"] += 1
        else:
            action = final_cases[action_key]
            matched_action_sources["relgate_v2_final"] += 1
        k = action["masked_count"]
        for seed in matched_seeds:
            selected = dose_matched_mask(sid, len(tools), k, seed)
            matched_cells.append(add_request(view, sid, tools, case["aliases"], before, after,
                                              selected, f"Matched-0.50-seed{seed}"))

    # Validate reversible aliasing over every scheduled prompt and arbitrary JSON tool-call output.
    roundtrips = 0
    for cell in [x for group in model_cells.values() for x in group] + matched_cells:
        inverse = cell["inverse"]
        assert len(inverse) == len(set(inverse)) and len(inverse.values()) == len(set(inverse.values()))
        raw_calls = {"tool_calls": [{"function": {"name": alias}, "tool": alias,
                                     "arguments": {"nested": [{"tool": alias}], "x": 1}}
                                     for alias in inverse]}
        restored = restore_names(raw_calls, inverse)
        expected = {"tool_calls": [{"function": {"name": inverse[alias]}, "tool": inverse[alias],
                                     "arguments": {"nested": [{"tool": inverse[alias]}], "x": 1}}
                                     for alias in inverse]}
        assert restored == expected
        for alias, original in inverse.items():
            assert inverse.get(alias) == original
            roundtrips += 1

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "unique_prompts.jsonl", requests.values())
    for model, cells in model_cells.items():
        filename = model.replace("/", "_")
        write_jsonl(OUT / f"{filename}.schedule.jsonl", cells)
        schedules.append({"model": model, "cells": len(cells), "unique_prompts": len({x['prompt'] for x in cells})})
    write_jsonl(OUT / "matched_random.schedule.jsonl", matched_cells)
    with (OUT / "unique_prompt_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "view"])
        w.writeheader()
        w.writerows({"prompt": k, "view": v["view"]} for k, v in requests.items())
    summary = {"view_counts": dict(Counter(r["view"] for r in cases)), "probe_counts": dict(Counter(r["view"] for r in probe)),
               "models": schedules, "matched_cells": len(matched_cells), "matched_seeds": list(matched_seeds),
               "matched_unique_prompts": len({r["prompt"] for r in matched_cells}),
               "matched_action_sources": dict(matched_action_sources),
               "main_unique_prompts": len(main_prompt_ids), "main_cells_per_model": len(cases + probe) * 4,
               "all_unique_prompts": len(requests), "inverse_mapping_entries_tested": roundtrips,
               "tokenizer_chat_template": "model native; add_generation_prompt=True",
               "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256},
               "note": "Prompt de-duplication is exact on tool list plus query; masked aliases are the frozen Stage A aliases."}
    (OUT / "schedule.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(manifest) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
