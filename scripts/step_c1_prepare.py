"""Prepare Step C1 local CPU audit artifacts from pinned BFCL and frozen exploration inputs."""
import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs/step_c_confirm"
BFCL = Path(r"F:\Tooluse\_bfcl_source_audit\berkeley-function-call-leaderboard")
REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
OPPOSITES = {
    "get": ["delete", "remove"], "fetch": ["delete", "remove"],
    "list": ["delete", "remove"], "read": ["delete", "remove"],
    "query": ["delete", "remove"], "search": ["delete", "remove"],
    "create": ["delete", "cancel"], "add": ["delete", "remove"],
    "insert": ["delete", "remove"], "post": ["delete", "remove"],
    "delete": ["get", "create"], "remove": ["get", "create"],
    "cancel": ["get", "create"], "enable": ["disable", "stop"],
    "start": ["stop", "disable"], "open": ["close", "stop"],
    "disable": ["enable", "start"], "stop": ["start", "enable"],
    "close": ["open", "start"], "upload": ["download", "fetch"],
    "send": ["receive", "fetch"], "push": ["pull", "fetch"],
    "download": ["upload", "send"], "encrypt": ["decrypt"],
    "decrypt": ["encrypt"],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def quantile(xs, q):
    ys = sorted(xs)
    if not ys:
        return None
    pos = (len(ys) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def dump_jsonl(path, records):
    path.write_text("".join(canonical(x) + "\n" for x in records), encoding="utf-8")


def description_lengths(record):
    tools, parameters = [], []
    for fn in record.get("function", []):
        if isinstance(fn.get("description"), str):
            tools.append(len(fn["description"]))
        def walk(value):
            if isinstance(value, dict):
                if isinstance(value.get("description"), str):
                    parameters.append(len(value["description"]))
                for key, child in value.items():
                    if key != "description":
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(fn.get("parameters", {}))
    return tools, parameters


def summarize_lengths(lengths):
    return {"fields": len(lengths), "p10": quantile(lengths, .10), "p50": quantile(lengths, .50),
            "p90": quantile(lengths, .90), "max": max(lengths, default=0),
            "gt_64_fields": sum(x > 64 for x in lengths),
            "gt_64_fraction": sum(x > 64 for x in lengths) / len(lengths) if lengths else None}


def flip_name(name, occupied):
    match = re.match(r"^([a-z]+)(?=(_|[A-Z]|$))", name)
    if not match or match.group(1) not in OPPOSITES:
        return None
    verb = match.group(1)
    suffix = name[len(verb):]
    return next((candidate + suffix for candidate in OPPOSITES[verb]
                 if candidate + suffix not in occupied), None)


def main():
    data_dir = BFCL / "bfcl_eval/data"
    split_names = ("multiple", "live_multiple", "simple_python")
    summaries, sample_rows, triples = {}, {}, {}
    all_rows = {}
    for split in split_names:
        source = data_dir / f"BFCL_v4_{split}.json"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_rows[split] = records
        tool_lengths, parameter_lengths = [], []
        for row in records:
            tool_values, parameter_values = description_lengths(row)
            tool_lengths.extend(tool_values)
            parameter_lengths.extend(parameter_values)
        candidate_counts = [len(row.get("function", [])) for row in records]
        summaries[split] = {
            "records": len(records), "source_sha256": sha(source),
            "candidate_tool_count_distribution": dict(sorted(Counter(candidate_counts).items())),
            "tool_description_chars": summarize_lengths(tool_lengths),
            "parameter_description_chars": summarize_lengths(parameter_lengths),
        }
        if split == "simple_python":
            continue
        original = deepcopy(records)
        swapped, flipped = [], []
        hits = Counter()
        for row in records:
            sw = deepcopy(row)
            tools = sw["function"]
            names = [x["name"] for x in tools]
            mapping = {name: names[(i + 1) % len(names)] for i, name in enumerate(names)} if len(names) > 1 else {}
            for tool in tools:
                tool["name"] = mapping.get(tool["name"], tool["name"])
            inverse = {v: k for k, v in mapping.items()}
            sample_rows.setdefault(split, []).append({"original": deepcopy(row), "name_swapped": deepcopy(sw),
                                                       "name_inverse_map": inverse})
            swapped.append(sw)
            flip = deepcopy(row)
            for target in flip["function"]:
                replacement = flip_name(target["name"], set(names))
                if replacement:
                    hits[re.match(r"^([a-z]+)", target["name"]).group(1)] += 1
                    target["name"] = replacement
                    break
            else:
                replacement = None
            if replacement:
                flipped.append(flip)
            # Count unique candidate triples across every materialized BFCL candidate view.
            for view_row in (row, sw, flip):
                for tool in view_row["function"]:
                    triple = {k: tool.get(k) for k in ("name", "description", "parameters")}
                    triples[canonical(triple)] = triple
        for view, view_rows in (("original", original), ("name_swapped", swapped), ("verb_flip_candidate_pool", flipped)):
            (OUT / "bfcl" / split).mkdir(parents=True, exist_ok=True)
            dump_jsonl(OUT / "bfcl" / split / f"{view}.jsonl", view_rows)
        summaries[split]["verb_flip_candidate_pool_records_hit"] = len(flipped)
        summaries[split]["verb_flip_target_hits"] = None
        summaries[split]["verb_flip_target_hits_note"] = "BFCL raw request files omit gold target; candidate-pool verb hits are not target hits"
        summaries[split]["verb_flip_matched_verbs"] = dict(hits)
        summaries[split]["name_swap_changes_function_name_only"] = True

    # Test the official BFCL multiple AST checker on valid and invalid calls before/after reversible aliases.
    ast_test = {"status": "not_run", "reason": None}
    try:
        import sys
        import types
        sys.path.insert(0, str(BFCL))
        registry = types.ModuleType("bfcl_eval.constants.model_config")
        registry.MODEL_CONFIG_MAPPING = {}
        sys.modules["bfcl_eval.constants.model_config"] = registry
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        schema = [{"name": "lookup_item", "description": "lookup", "parameters": {
            "type": "dict", "required": ["kind"], "properties": {
                "kind": {"type": "string", "enum": ["a", "b"]},
                "limit": {"type": "integer"}}}}]
        gold = [{"lookup_item": {"kind": ["a", "b"], "limit": [1, 2]}}]
        cases = [[{"lookup_item": {"kind": "a", "limit": 1}}],
                 [{"lookup_item": {"kind": "z", "limit": 1}}],
                 [{"lookup_item": {"limit": 1}}],
                 [{"lookup_item": {"kind": "a", "limit": "1"}}]]
        checks = []
        for case in cases:
            base = ast_checker(schema, case, gold, Language.PYTHON, "multiple", "gpt_4o")
            alias_schema = deepcopy(schema)
            alias_schema[0]["name"] = "func_0"
            alias_gold = [{"func_0": gold[0]["lookup_item"]}]
            aliased_case = [{"func_0" if name == "lookup_item" else name: args for name, args in case[0].items()}]
            aliased = ast_checker(alias_schema, aliased_case, alias_gold, Language.PYTHON, "multiple", "gpt_4o")
            restored_case = [{"lookup_item" if name == "func_0" else name: args for name, args in aliased_case[0].items()}]
            restored = ast_checker(schema, restored_case, gold, Language.PYTHON, "multiple", "gpt_4o")
            checks.append(base == aliased == restored)
        ast_test = {"status": "passed" if all(checks) else "failed", "cases": len(checks),
                    "equivalent": sum(checks), "scope": "official AST checker result unchanged after deterministic func_k name round-trip",
                    "covers": ["function name", "enum", "required field", "parameter type"]}
    except Exception as exc:
        ast_test = {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}"}

    # Five paired audit examples per category. Kept only in the local review artifact.
    audit_samples = []
    for split, pairs in sample_rows.items():
        for pair in pairs[:5]:
            audit_samples.append({"split": split, **pair})

    # 50 frozen Clean requests from the previously used exploration set; IDs and prompt contents stay in the artifact.
    schedule_path = OUT.parent / "step_b_plugplay/Qwen2.5-7B-Instruct.schedule.jsonl"
    clean = [json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines()
             if line.strip() and (lambda x: x.get("view") == "Clean" and x.get("policy") == "Keep-Name")(json.loads(line))]
    clean = sorted(clean, key=lambda x: x["id"])[:50]
    score_path = ROOT / "research_outputs/relgate_v2_final/score_inputs.jsonl"
    score_rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    score_by_id = {x["id"]: x for x in score_rows}
    prompt_rows = [json.loads(line) for line in (OUT.parent / "step_b_plugplay/unique_prompts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt_by_id = {x["id"]: x for x in prompt_rows}
    frozen_prompt_rows = [json.loads(line) for line in (OUT.parent / "routeB_step2b_v1/inputs/keep.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt_by_id.update({x["id"]: x for x in frozen_prompt_rows})
    clean_inputs = []
    for row in clean:
        candidate_ids = row.get("triples", [])
        prompt = prompt_by_id.get(row.get("prompt"), {})
        messages = prompt.get("messages")
        if messages:
            sys.path.insert(0, str(ROOT / "scripts"))
            from routeb_stage1 import split_prompt
            _, tools, _ = split_prompt({"messages": messages})
        else:
            tools = []
        candidate_ids = []
        for tool in tools:
            fields = {k: tool.get(k) for k in ("name", "description", "parameters")}
            candidate_ids.append(hashlib.sha256(canonical(fields).encode("utf-8")).hexdigest())
        clean_inputs.append({"id": row["id"], "view": "Clean", "messages": prompt.get("messages"),
                             "candidate_ids": candidate_ids,
                             "candidates": [score_by_id[x] for x in candidate_ids if x in score_by_id],
                             "candidate_count": len(tools)})

    OUT.mkdir(parents=True, exist_ok=True)
    dump_jsonl(OUT / "bfcl_audit_samples.jsonl", audit_samples)
    dump_jsonl(OUT / "bfcl_unique_candidate_triples.jsonl", [triples[k] for k in sorted(triples)])
    (OUT / "bfcl_audit.json").write_text(json.dumps({
        "source": "official Gorilla BFCL v4 public repository", "repository_revision": REVISION,
        "scorer": "bfcl_eval/eval_checker/ast_eval/ast_checker.py", "splits": summaries,
        "ast_inverse_map_unit_test": ast_test,
        "name_swap": "within-request deterministic cyclic name permutation; only function name fields changed; inverse mapping retained in sample review file",
        "verb_flip": "candidate-pool-only exploratory construct; target-slot labels are unavailable in source request files, so it is not certified as Verb-Flip-Target",
        "blind_sample_content_included": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "microbench_50_clean.json").write_text(json.dumps({
        "selection": "first 50 sorted request IDs from frozen exploration Clean view",
        "batch_size": 1, "cross_request_batching": False, "requests": clean_inputs,
        "gate_mode": "pure candidate scoring; request candidates packed as one mini-batch; no greedy generation",
        "caller_mode": "single request end-to-end generation",
        "input_complete": all(x["messages"] and len(x["candidates"]) == x["candidate_count"] for x in clean_inputs),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    budget = {
        "phase": "C2 preparation only", "new_training": False, "gpu_used_in_c1": False,
        "heldout_334": {"available": False, "required_records": 334, "found_records": 0,
                        "status": "blocked: source file not found in project or F:\\Tooluse filename search; no content read"},
        "bfcl": {"revision": REVISION, "unique_candidate_triples_across_frozen_views": len(triples),
                 "exact_prompt_counts": "not computed: caller prompt construction is coupled to missing 334 and frozen gate scores"},
        "callers": ["Hammer-Repro 1.5B", "B2 1.5B", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"],
        "serial_disk_plan": "one model at a time; download, run, return outputs and verify SHA-256, delete model before next download; enforce available data disk >= model plus outputs before Phase C2",
        "runtime_estimate": "not estimable without 334 source, BFCL view freeze, exact prompt counts and per-model throughput under the agreed runtime",
        "manual_gate": "Phase C2 remains unauthorized pending human review of BFCL audit and completed compute budget",
    }
    (OUT / "phase_c2_budget.json").write_text(json.dumps(budget, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    sums = "".join(f"{sha(path)}  {path.relative_to(OUT).as_posix()}\n" for path in files)
    (OUT / "SHA256SUMS.txt").write_text(sums, encoding="ascii")
    for line in sums.splitlines():
        digest, name = line.split("  ", 1)
        assert sha(OUT / name) == digest
    print(json.dumps({"bfcl": summaries, "ast_test": ast_test, "bfcl_unique_triples": len(triples),
                      "microbench_clean_requests": len(clean_inputs), "heldout_found": 0,
                      "sha256_files": len(files), "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
