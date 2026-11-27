"""Bind the already frozen 50 exploration requests to frozen Step B actions."""
import json
from pathlib import Path

from step_c2_freeze import ROOT, OUT, DEST, read, sha, locked_write, encode
from routeb_stage1 import split_prompt


def main():
    protocol = json.loads((DEST / "protocol.json").read_bytes())
    source = OUT / "microbench_50_clean.json"
    assert sha(source) == protocol["inputs_sha256"][source.relative_to(ROOT).as_posix()]
    data = json.loads(source.read_bytes())
    assert data["input_complete"] and len(data["requests"]) == 50
    old = ROOT / "research_outputs/step_b_plugplay"
    prompts = {r["id"]: r for r in read(old / "unique_prompts.jsonl")}
    actions = {r["id"]: r for r in read(old / "Qwen2.5-7B-Instruct.schedule.jsonl")
               if r["view"] == "Clean" and r["policy"] == "RelGate-0.50"}
    for r in data["requests"]:
        action = actions[r["id"]]
        messages = prompts[action["prompt"]]["messages"]
        prefix, tools, suffix = split_prompt(r)
        _, modified, _ = split_prompt({"messages": messages})
        assert len(tools) == len(modified) == len(r["candidates"])
        mask, names = [], []
        for original, altered in zip(tools, modified, strict=True):
            assert {k: v for k, v in original.items() if k != "name"} == {k: v for k, v in altered.items() if k != "name"}
            names.append(altered["name"])
            mask.append(altered["name"] != original["name"])
        assert prefix + json.dumps(modified, ensure_ascii=False) + suffix == messages[0]["content"]
        r.update(gate_messages=messages, expected_mask=mask, visible_names=names)
    data.update(source_sha256=sha(source), purpose="timing only; frozen exploration actions; no final evaluation outputs")
    path = DEST / "benchmark_inputs.json"
    locked_write(path, encode(data))
    locked_write(DEST / "BENCHMARK_SHA256SUMS.txt", f"{sha(path)}  {path.name}\n".encode())
    print(json.dumps({"requests": 50, "candidate_occurrences": sum(r["candidate_count"] for r in data["requests"]),
                      "unique_candidates": len({k for r in data["requests"] for k in r["candidate_ids"]}),
                      "sha256": sha(path)}))


if __name__ == "__main__":
    main()
