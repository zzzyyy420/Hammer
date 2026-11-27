"""Resolve frozen C2 actions only after complete, verified gate likelihoods."""
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from build_pure_name_pair import UPSTREAM, sft, strings
from pareto_policies import apply_name_mask, dose_matched_mask
from routeb_stage1 import split_prompt
from step_c2_freeze import ROOT, OUT, DEST, read, sha, locked_write, encode


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonl(path, rows):
    locked_write(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode())


def main():
    protocol = json.loads((DEST / "protocol.json").read_bytes())
    for name, expected in protocol["inputs_sha256"].items():
        assert sha(ROOT / name) == expected, name
    scorepath = DEST / "scores/3way.jsonl"
    scores = read(scorepath)
    report = json.loads((DEST / "scores/3way.done.json").read_bytes())
    assert sha(scorepath) == report["sha256"]
    scoremap = {r["id"]: r for r in scores}
    assert len(scores) == len(scoremap) == protocol["unique_candidates_exact"]
    assert set(scoremap) == {r["id"] for r in read(DEST / "score_inputs.jsonl")}
    assert all(r["finite"] and all(0 <= v <= 1 for v in r["q"].values()) for r in scores)
    assert report["adapter_sha256"] == "8403404d9518324bf16cb457dc25b795fb134309687a1c74313863215fa02259"
    assert report["score"] == "full label likelihood"
    prompts, cells, actions, counts = {}, [], [], {}
    template = strings(UPSTREAM)
    rounds = 0
    for dataset in ("heldout", "bfcl"):
        folder = OUT / "c15" / dataset
        path = folder / ("private/seven_views.jsonl" if dataset == "heldout" else "views.jsonl")
        views = read(path)
        oldprompts = {r["id"]: r for r in read(folder / "static_unique_prompts.jsonl")}
        oldcells = {(r["id"], r["view"], r["policy"]): r for r in read(folder / "static_schedule.jsonl")}
        for row in views:
            keep = oldcells[row["id"], row["view"], "Keep-Name"]
            _, tools, _ = split_prompt({"messages": [oldprompts[keep["prompt"]]["messages"][-1]]})
            assert tools == row["tools"]
            aliases = row["aliases"]
            keys = [fingerprint({k: t.get(k) for k in ("name", "description", "parameters")}) for t in tools]
            gate = [scoremap[k]["q"]["CONTRADICTED"] >= 0.50 for k in keys]
            options = [("Keep-Name", [False] * len(tools)), ("Always-Mask", [True] * len(tools)),
                       ("Length-64", [len(t.get("description") or "") > 64 for t in tools]), ("RelGate-0.50", gate)]
            if dataset == "heldout" and row["view"] in ("Target-Swapped", "Distractor-Swapped"):
                for seed in (42, 123, 2026):
                    mask = dose_matched_mask(row["id"], len(tools), sum(gate), seed)
                    assert sum(mask) == sum(gate)
                    options.append((f"Matched-0.50-seed{seed}", mask))
            actions.append({"dataset": dataset, "id": row["id"], "view": row["view"], "candidate_ids": keys,
                            "mask": gate, "masked_count": sum(gate), "candidate_count": len(tools)})
            for policy, selected in options:
                modified, inverse = apply_name_mask(tools, aliases, selected)
                messages = deepcopy(row.get("system_messages", [])) + [{"role": "user", "content": sft(row["query"], modified, [], template)["instruction"]}]
                pid = fingerprint(messages)
                prompt = {"id": pid, "messages": messages}
                assert pid not in prompts or prompts[pid] == prompt
                prompts[pid] = prompt
                cell = {"id": row["id"], "dataset": dataset, "split": row.get("split", dataset), "view": row["view"],
                        "policy": policy, "prompt": pid, "inverse": inverse, "masked_count": sum(selected),
                        "candidate_count": len(tools), "visible_to_view": {t["name"]: v["name"] for t, v in zip(modified, tools)},
                        "view_to_original": row["view_to_original"]}
                if policy in ("Keep-Name", "Always-Mask", "Length-64"):
                    before = oldcells[row["id"], row["view"], policy]
                    assert all(cell[k] == v for k, v in before.items()), (dataset, row["id"], row["view"], policy, [k for k, v in before.items() if cell[k] != v])
                    assert prompt == oldprompts[pid]
                for t, original in zip(modified, tools):
                    assert cell["visible_to_view"][t["name"]] == original["name"]
                    assert cell["view_to_original"][original["name"]] in row["view_to_original"].values()
                    rounds += 1
                cells.append(cell)
        subset = [r for r in cells if r["dataset"] == dataset]
        counts[dataset] = {"logical": len(subset), "unique_prompts": len({r["prompt"] for r in subset}),
                           "by_policy": dict(Counter(r["policy"] for r in subset))}
    assert len(cells) == 21026
    assert counts["heldout"]["unique_prompts"] <= 8146 and counts["bfcl"]["unique_prompts"] <= 7297
    assert len(prompts) <= 15443
    smoke = [r for r in cells if r["dataset"] == "heldout" and r["view"] == "Clean" and r["policy"] == "Keep-Name"][:20]
    restored = {r["id"]: r for r in read(OUT / "c15/heldout/private/restored_334.jsonl")}
    smoke = [dict(c, ground_truth=restored[c["id"]]["record"]["answers"]) for c in smoke]
    jsonl(DEST / "unique_prompts.jsonl", prompts.values())
    jsonl(DEST / "schedule.jsonl", cells)
    jsonl(DEST / "gate_actions.jsonl", actions)
    jsonl(DEST / "private/smoke_gold_clean_keep20.jsonl", smoke)
    summary = {"datasets": counts, "all_unique_prompts": len(prompts), "logical_per_caller": len(cells),
               "logical_four_callers": len(cells) * 4, "roundtrips": rounds, "scores_sha256": sha(scorepath),
               "dedup_fraction": 1 - len(prompts) / len(cells)}
    locked_write(DEST / "schedule_summary.json", encode(summary))
    paths = [DEST / n for n in ("unique_prompts.jsonl", "schedule.jsonl", "gate_actions.jsonl",
                               "private/smoke_gold_clean_keep20.jsonl", "schedule_summary.json")]
    locked_write(DEST / "SCHEDULE_SHA256SUMS.txt", "".join(f"{sha(p)}  {p.relative_to(DEST).as_posix()}\n" for p in paths).encode())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
