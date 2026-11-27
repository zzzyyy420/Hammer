"""Validate the already assembled pool against frozen per-request actions."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from routeb_stage1 import rows, split_prompt
from routeb_step2a import fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs/step_b_plugplay"


def main():
    for line in (OUT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
    pool = {r["id"]: r for r in rows(OUT / "unique_prompts.jsonl")}
    exact = {}
    for pid, row in pool.items():
        assert fingerprint({"messages": [{"role": m["role"], "content": m["content"]}
                                         for m in row["messages"]]}) == pid
        before, tools, after = split_prompt(row)
        key = json.dumps([tools, before, after], ensure_ascii=False, sort_keys=True)
        assert key not in exact, "tool/query duplicate not removed"
        exact[key] = pid
    schedule = rows(OUT / "Qwen2.5-7B-Instruct.schedule.jsonl")
    assert schedule == rows(OUT / "Llama-3.1-8B-Instruct.schedule.jsonl")
    expected = {"Clean":197, "Coarse-Desc":197, "Name-Swapped":197, "Target-Swapped":182,
                "Distractor-Swapped":182, "Desc-Degraded":165, "Verb-Flip-Target":72, "Verbose-Coarse":197}
    assert Counter((r["policy"], r["view"]) for r in schedule) == {
        (policy, view): n for policy in ("Keep-Name", "Always-Mask", "Length-64", "RelGate-0.50") for view, n in expected.items()}
    paths = [ROOT / "research_outputs/relgate_v2_final/b2_cases.jsonl",
             ROOT / "research_outputs/step_a2_incremental/B2__step_a2.cases.jsonl",
             ROOT / "research_outputs/relgate_v2_final/b2_matched_cases.jsonl"]
    frozen = {(r["id"], r["view"], r["policy"]): r for path in paths for r in rows(path)}
    matched = rows(OUT / "matched_random.schedule.jsonl")
    verified = 0
    aliases = 0
    for cell in schedule + matched:
        old = frozen[(cell["id"], cell["view"], cell["policy"])]
        for key in ("prompt", "inverse", "masked_count", "candidate_count"):
            assert cell[key] == old[key], (cell["id"], cell["view"], cell["policy"], key)
        inverse = cell["inverse"]
        assert len(set(inverse.values())) == len(inverse)
        _, tools, _ = split_prompt(pool[cell["prompt"]])
        names = [t["name"] for t in tools]
        assert len(set(names)) == len(names)
        restored = [inverse.get(n, n) for n in names]
        assert len(set(restored)) == len(restored)
        forward = {original: alias for alias, original in inverse.items()}
        assert [forward.get(n, n) for n in restored] == names
        aliases += len(inverse)
        verified += 1
    assert len(matched) == 1092
    assert Counter((r["policy"], r["view"]) for r in matched) == {
        (f"Matched-0.50-seed{seed}", view): 182 for seed in (42,123,2026) for view in ("Target-Swapped", "Distractor-Swapped")}
    main_n = len({r["prompt"] for r in schedule})
    assert main_n <= 4438 and len(pool) <= 5530
    report = dict(passed=True, frozen_cells_verified=verified, inverse_entries_verified=aliases,
                  main_unique=main_n, total_unique=len(pool), main_cells=len(schedule), total_cells=len(schedule + matched),
                  main_dedup_rate=1-main_n/len(schedule), total_dedup_rate=1-len(pool)/len(schedule + matched),
                  sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT / "b1_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
