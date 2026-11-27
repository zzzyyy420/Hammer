"""Recover frozen IDs and audit official BFCL gold locally; never contact GPUs."""
import ast
import hashlib
import itertools
import json
import re
import random
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path

from build_diagnostic_views import normalized
from build_pure_name_pair import UPSTREAM, sft, strings
from prepare_identifiability_pilot import eligible_rows
from step_c1_prepare import BFCL, REVISION, canonical, flip_name, sha
from step_c_bfcl_adapter import frozen_json_parser, official_checker, restore_calls, score_raw
from routeb_stage1 import coarse, cleaned, names_mask, permutation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs/step_c_confirm"
DEST = OUT / "c15"
SOURCE = Path("F:/Tooluse/Hammer_release/source/Hammer/data/test/original")


def read(path):
    return [json.loads(x) for x in path.read_bytes().splitlines() if x.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(canonical(r) + "\n" for r in records).encode())


def hashobj(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def recover_holdout():
    frozen = ROOT / "research_outputs/phase1_freeze_v1/heldout_ids.json"
    expected = (frozen.with_suffix(".sha256")).read_text().split()[0]
    assert sha(frozen) == expected == "51520b465e0a57f9fb03412f88594f79c9f7884f43ea63c3d70ef039145a2162"
    manifest = json.loads(frozen.read_bytes())
    pilot = read(ROOT / "research_outputs/identifiability_pilot_v1/private/key.jsonl")
    restored, identity, counts = [], {}, {}
    for source, spec in manifest.items():
        path = SOURCE / (source + ".json")
        assert sha(path) == spec["source_sha256"]
        candidates = eligible_rows(path.read_bytes())
        eligible = {hashlib.sha256(f"{source}/{index}".encode()).hexdigest()[:16]: (index, row)
                    for index, row, _ in candidates}
        used = {r["id"] for r in pilot if r["source"] == source}
        ids = spec["ids"]
        assert len(ids) == len(set(ids)) and len(used) == 100
        assert set(ids) == set(eligible) - used and not set(ids) & used
        for sid in ids:
            index, row = eligible[sid]
            restored.append({"id": sid, "source": source, "row_index": index, "record": row})
        counts[source] = len(ids)
        identity[source] = {"path": str(path), "sha256": sha(path), "eligible": len(eligible),
                            "pilot": len(used), "restored": len(ids)}
    assert len(restored) == len({r["id"] for r in restored}) == 334
    jsonl(DEST / "heldout/private/restored_334.jsonl", restored)
    jsonl(DEST / "heldout/frozen_id_index.jsonl", [{k: r[k] for k in ("id", "source", "row_index")} for r in restored])
    # No question or answer text is printed or placed in the human-facing audit.
    summary = {"status": "recovered_original_frozen_population", "records": len(restored),
               "by_source": counts, "frozen_id_manifest_sha256": expected, "sources": identity,
               "pilot_overlap": 0, "resampling": False, "content_human_reviewed": False,
               "reconstruction": "original eligible-row index hash and exact frozen ID membership",
               "calibration_v3_strict": "historical eight output files contain 200 pilot rows each, not holdout",
               "backup_search": ["phase1_calibration_v3_strict.tar.gz", "phase1_metadata_views.tar.gz", "environment/server-snapshot.tar.gz"],
               "remote_contacted": False}
    write(DEST / "holdout_recovery_audit.json", summary)
    return restored, summary


def aliases_for(names):
    used, result, k = set(names), [], 0
    for _ in names:
        while f"func_{k}" in used:
            k += 1
        result.append(f"func_{k}")
        used.add(result[-1])
        k += 1
    return result


def frozen_probe_literals():
    tree = ast.parse((ROOT / "scripts/step_a_probes.py").read_text(encoding="utf-8"))
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {"templates", "opposites"}}
    from step_c1_prepare import OPPOSITES
    assert values["opposites"] == OPPOSITES
    assert all(len(t) > 70 for t in values["templates"])
    assert sha(ROOT / "research_outputs/step_a_v1/probes.jsonl") == "b7fa9b653252ae3db211837016fde002ca07c05acf747d10022c0bb485ac1ed6"
    return values["templates"]


def variants_for(tools, g, sid, templates):
    names = [t["name"] for t in tools]
    full, h, _, sim = permutation(tools, g, sid)
    descs = coarse(tools)
    swapped = deepcopy(tools)
    for i, t in enumerate(swapped):
        t["name"] = names[full[i]]
        t["description"] = cleaned(tools[i].get("description"), names[i])
    verbose = [dict(t, description=templates[i % 3]) for i, t in enumerate(tools)]
    result = {"Clean": tools, "Coarse-Desc": [dict(t, description=d) for t, d in zip(tools, descs)],
              "Name-Swapped": swapped, "Verbose-Coarse": verbose}
    if len(tools) >= 3:
        others = [i for i in range(len(tools)) if i != g]
        d1, d2 = max(itertools.combinations(others, 2), key=lambda ij: sim(*ij))
        for view, (a, b) in (("Target-Swapped", (g, h)), ("Distractor-Swapped", (d1, d2))):
            saved = {a, b}
            extras = [i for i in range(len(tools)) if i not in saved and (view == "Target-Swapped" or i != g)]
            if extras:
                saved.add(random.Random("full/42/" + sid + "/" + view).choice(extras))
            ts = deepcopy(tools)
            ts[a]["name"], ts[b]["name"] = names[b], names[a]
            for i, t in enumerate(ts):
                t["description"] = cleaned(tools[i].get("description"), names[i]) if i in saved else descs[i]
            result[view] = ts
    replacement = flip_name(names[g], set(names))
    if replacement:
        flipped = deepcopy(tools)
        flipped[g]["name"] = replacement
        result["Verb-Flip-Target"] = flipped
    return result


def holdout_views(restored):
    templates = frozen_probe_literals()
    # Verify the reused transformations against every existing exploration view
    # before processing holdout content. No exploratory outputs are regenerated.
    from build_routeB_swap_probe import split_prompt
    directory = ROOT / "research_outputs/routeB_step2b_v1"
    prompts = {r["id"]: r for r in read(directory / "inputs/keep.jsonl")}
    cases = read(directory / "cases.jsonl")
    keys = {(r["id"], r["view"]): r for r in read(ROOT / "research_outputs/routeB_stage1_v1/eval_keys.jsonl")
            if r["mode"] == "Keep-Name"}
    expected = {(r["id"], r["view"]): split_prompt(prompts[r["prompt"]])[1] for r in cases
                if r["view"] not in {"Desc-Degraded", "Name-Masked"}}
    expected.update({(r["id"], r["view"]): r["tools"] for r in read(ROOT / "research_outputs/step_a_v1/probes.jsonl")})
    checks = 0
    for c in (x for x in cases if x["view"] == "Clean"):
        tools = split_prompt(prompts[c["prompt"]])[1]
        for view, ts in variants_for(tools, keys[c["id"], "Clean"]["g"], c["id"], templates).items():
            assert ts == expected[c["id"], view]
            checks += 1
    views, envelope = [], {}
    for r in restored:
        sid, row = r["id"], r["record"]
        tools = [normalized(t, True) for t in row["tools"]]
        names = [t["name"] for t in tools]
        target = row["answers"][0].get("name", row["answers"][0].get("tool"))
        g = names.index(target)
        aliases = names_mask(tools, sid)
        for view, ts in variants_for(tools, g, sid, templates).items():
            assert not set(aliases) & {t["name"] for t in ts}
            views.append({"id": sid, "view": view, "tools": ts, "aliases": aliases, "query": row["query"],
                          "view_to_original": {t["name"]: n for t, n in zip(ts, names)}, "target_slot": g})
        for t in (normalized(t, False) for t in row["tools"]):
            triple = {k: t.get(k) for k in ("name", "description", "parameters")}
            envelope[hashobj(triple)] = triple
    jsonl(DEST / "heldout/private/seven_views.jsonl", views)
    jsonl(DEST / "heldout/desc_degraded_candidate_envelope_NOT_FINAL.jsonl", [{"id": k, **t} for k, t in sorted(envelope.items())])
    summary = assemble_static(views, "heldout")
    exact_triples = {r["id"]: {k: r.get(k) for k in ("name", "description", "parameters")}
                     for r in read(DEST / "heldout/unique_candidate_triples.jsonl")}
    summary["unique_candidate_triples_including_unresolved_desc_cohort_upper_bound"] = len(exact_triples.keys() | envelope.keys())
    summary["frozen_exploration_transform_equivalence_checks"] = checks
    summary["candidate_order"] = "source record order; original holdout freeze specifies IDs, not a shuffled candidate order"
    return summary


def representative_arguments(function, gold_arguments):
    """Materialize one official accepted value per field for scorer controls."""
    schema = function["parameters"]
    result = {}
    def dictionary(value):
        return {k: next((x for x in choices if x != ""), "")
                for k, choices in value.items() if "" not in choices}
    for key, choices in gold_arguments.items():
        if "" in choices and key not in schema["required"]:
            continue
        value = next((x for x in choices if x != ""), "")
        field = schema["properties"].get(key, {})
        if field.get("type") == "dict" and isinstance(value, dict):
            value = dictionary(value)
        elif field.get("type") == "array" and field.get("items", {}).get("type") == "dict" and isinstance(value, list):
            value = [dictionary(x) for x in value]
        result[key] = value
    return result


def bfcl_views(check, parser):
    views, audits, tests, samples, control_failures = [], {}, Counter(), [], []
    data = BFCL / "bfcl_eval/data"
    for split in ("multiple", "live_multiple"):
        request_path, answer_path = data / f"BFCL_v4_{split}.json", data / f"possible_answer/BFCL_v4_{split}.json"
        requests, answers = read(request_path), read(answer_path)
        rc, ac = Counter(x["id"] for x in requests), Counter(x["id"] for x in answers)
        assert all(x == 1 for x in rc.values()) and all(x == 1 for x in ac.values())
        assert rc.keys() == ac.keys()
        answers = {r["id"]: r["ground_truth"] for r in answers}
        hits, counts, field_issues = Counter(), Counter(), []
        for row in requests:
            sid, functions, gold = row["id"], row["function"], answers[row["id"]]
            names = [t["name"] for t in functions]
            assert len(names) >= 2 and len(names) == len(set(names))
            assert len(gold) == 1 and len(gold[0]) == 1
            target = next(iter(gold[0]))
            assert target in names
            g = names.index(target)
            target_schema = functions[g]["parameters"]
            gold_keys = set(gold[0][target])
            missing = sorted(set(target_schema["required"]) - gold_keys)
            extra = sorted(gold_keys - set(target_schema["properties"]))
            if missing or extra:
                field_issues.append({"id": sid, "required_keys_missing_gold": missing,
                                     "gold_keys_absent_schema": extra, "excluded_from_evaluation": False})
            original_map = {n: n for n in names}
            swapped = deepcopy(functions)
            for i, tool in enumerate(swapped):
                tool["name"] = names[(i + 1) % len(names)]
            swap_map = {t["name"]: n for t, n in zip(swapped, names)}
            variants = [("BFCL-Original", functions, original_map), ("BFCL-Name-Swapped", swapped, swap_map)]
            replacement = flip_name(target, set(names))
            if replacement:
                flipped = deepcopy(functions)
                flipped[g]["name"] = replacement
                flip_map = {t["name"]: n for t, n in zip(flipped, names)}
                variants.append(("BFCL-Verb-Flip", flipped, flip_map))
                hits[re.match(r"^([a-z]+)", target).group(1)] += 1
            # A representative gold call tests mapping, not caller performance.
            args = representative_arguments(functions[g], gold[0][target])
            baseline = check(functions, [{target: args}], gold, split)
            tests["gold_representative_valid"] += bool(baseline["valid"])
            tests["gold_representative_total"] += 1
            if not baseline["valid"]:
                tests["representative_failure:" + baseline.get("error_type", "unknown")] += 1
                control_failures.append({"id": sid, "split": split, "official_schema": functions[g],
                                         "official_gold": gold, "synthetic_control_arguments": args,
                                         "official_result": baseline, "excluded_from_evaluation": False})
            occupied = names + ([replacement] if replacement else [])
            aliases = aliases_for(occupied)[:len(names)]
            for view, tools, inverse in variants:
                vn = [t["name"] for t in tools]
                # Exhaustive masks for small pools; all single slots plus alternating
                # and full masks cover larger pools without an exponential audit.
                if len(tools) <= 6:
                    masks = list(itertools.product((False, True), repeat=len(tools)))
                else:
                    masks = [tuple(False for _ in tools), tuple(True for _ in tools),
                             tuple(i % 2 == 0 for i in range(len(tools))),
                             tuple(i % 2 == 1 for i in range(len(tools)))]
                    masks += [tuple(i == j for i in range(len(tools))) for j in range(len(tools))]
                for mask in masks:
                    visible = [aliases[i] if mask[i] else n for i, n in enumerate(vn)]
                    assert len(visible) == len(set(visible))
                    first = dict(zip(visible, vn))
                    # Every candidate, not only the target, must round-trip exactly.
                    calls = [{n: {"literal": n, "name": n}} for n in visible]
                    restored = restore_calls(calls, first, inverse)
                    assert [list(x) for x in restored] == [[n] for n in names]
                    assert all(next(iter(a.values())) == next(iter(b.values())) for a, b in zip(calls, restored))
                    raw = json.dumps([{"name": visible[g], "arguments": args}])
                    result = score_raw(raw, parser, check, functions, gold, first, inverse, split)
                    assert result == baseline
                    tests["official_ast_composed_equivalences"] += 1
                    tests["candidate_roundtrips"] += len(tools)
                questions = row["question"]
                assert len(questions) == 1
                messages = questions[0]
                assert messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str)
                assert all(m["role"] == "system" for m in messages[:-1])
                record = {"id": sid, "split": split, "view": view, "tools": tools,
                          "query": messages[-1]["content"], "system_messages": messages[:-1], "aliases": aliases,
                          "view_to_original": inverse, "target_slot": g}
                views.append(record)
                counts[view] += 1
            if len([s for s in samples if s["split"] == split]) < 5:
                samples.append({"split": split, "id": sid, "original": row,
                                "name_swapped": dict(row, function=swapped), "inverse": swap_map})
        audits[split] = {"requests": len(requests), "official_answers": len(answers),
                         "missing_answer_ids": 0, "extra_answer_ids": 0, "duplicate_ids": 0,
                         "gold_functions_not_in_candidates": 0, "single_gold_target_validated": len(requests),
                         "schema_gold_field_inconsistencies": field_issues,
                         "verb_flip_target_hits": sum(hits.values()), "target_verb_frequencies": dict(hits),
                         "views": dict(counts), "request_sha256": sha(request_path), "answer_sha256": sha(answer_path)}
        jsonl(DEST / f"bfcl/{split}/official_answers.jsonl", [{"id": i, "ground_truth": a} for i, a in answers.items()])
    jsonl(DEST / "bfcl/views.jsonl", views)
    jsonl(DEST / "bfcl/audit_samples.jsonl", samples)
    jsonl(DEST / "bfcl/representative_control_failures.jsonl", control_failures)
    return views, audits, tests


def minimum_tests(check, parser):
    schema = [{"name": "lookup_item", "parameters": {"type": "dict", "required": ["kind"],
               "properties": {"kind": {"type": "string", "enum": ["a", "b"]}, "limit": {"type": "integer"}}}}]
    gold = [{"lookup_item": {"kind": ["a", "b"], "limit": [1, 2]}}]
    cases = [("valid", {"kind": "a", "limit": 1}, True),
             ("invalid_enum", {"kind": "z", "limit": 1}, False),
             ("missing_required", {"limit": 1}, False),
             ("invalid_type", {"kind": "a", "limit": "1"}, False)]
    results = []
    for label, args, expected in cases:
        base = check(schema, [{"lookup_item": args}], gold)
        restored = score_raw(json.dumps([{"name": "func_0", "arguments": args}]), parser, check,
                             schema, gold, {"func_0": "lookup_item"}, {"lookup_item": "lookup_item"})
        alias_schema = deepcopy(schema)
        alias_schema[0]["name"] = "func_0"
        aliased = check(alias_schema, [{"func_0": args}], [{"func_0": gold[0]["lookup_item"]}])
        assert base["valid"] is expected and base == aliased == restored
        results.append({"case": label, "expected": expected, "actual": base["valid"], "alias_actual": aliased["valid"],
                        "restored_equal": restored == base, "error_type": base.get("error_type")})
    for raw, label in [("invalid JSON", "parse_failure"),
                       (json.dumps([{"name": "unknown", "arguments": {"kind": "a"}}]), "unknown_function"),
                       (json.dumps([{"name": "func_99", "arguments": {"kind": "a"}}]), "unknown_alias")]:
        r = score_raw(raw, parser, check, schema, gold, {"lookup_item": "lookup_item"}, {"lookup_item": "lookup_item"})
        assert r["valid"] is False
        results.append({"case": label, "expected": False, "actual": r["valid"], "error_type": r["error_type"]})
    wrong = deepcopy(schema) + [{"name": "other", "parameters": schema[0]["parameters"]}]
    assert check(wrong, [{"other": {"kind": "a", "limit": 1}}], gold)["valid"] is False
    results.append({"case": "known_wrong_candidate", "expected": False, "actual": False})
    return results


def assemble_static(views, dataset):
    prompts, schedule, triples = {}, [], {}
    maximum_input_bytes = 0
    prompt_template = strings(UPSTREAM)
    for r in views:
        longest_names = deepcopy(r["tools"])
        for i, t in enumerate(longest_names):
            if len(json.dumps(r["aliases"][i], ensure_ascii=False).encode()) > len(json.dumps(t["name"], ensure_ascii=False).encode()):
                t["name"] = r["aliases"][i]
        longest_messages = deepcopy(r.get("system_messages", [])) + [{"role": "user", "content": sft(r["query"], longest_names, [], prompt_template)["instruction"]}]
        maximum_input_bytes = max(maximum_input_bytes, len(canonical({"id": hashobj(longest_messages), "messages": longest_messages}).encode()) + 1)
        for t in r["tools"]:
            triple = {k: t.get(k) for k in ("name", "description", "parameters")}
            triples[hashobj(triple)] = triple
        for policy in ("Keep-Name", "Always-Mask", "Length-64"):
            tools = deepcopy(r["tools"])
            inverse = {}
            for i, t in enumerate(tools):
                active = policy == "Always-Mask" or policy == "Length-64" and len(t.get("description") or "") > 64
                if active:
                    inverse[r["aliases"][i]] = t["name"]
                    t["name"] = r["aliases"][i]
            assert len({t["name"] for t in tools}) == len(tools)
            messages = deepcopy(r.get("system_messages", [])) + [{"role": "user", "content": sft(r["query"], tools, [], prompt_template)["instruction"]}]
            pid = hashobj(messages)
            prompts.setdefault(pid, {"id": pid, "messages": messages})
            schedule.append({"id": r["id"], "split": r.get("split", dataset), "view": r["view"],
                             "policy": policy, "prompt": pid, "inverse": inverse,
                             "visible_to_view": {t["name"]: o["name"] for t, o in zip(tools, r["tools"])},
                             "view_to_original": r["view_to_original"]})
    jsonl(DEST / dataset / "static_unique_prompts.jsonl", prompts.values())
    jsonl(DEST / dataset / "static_schedule.jsonl", schedule)
    jsonl(DEST / dataset / "unique_candidate_triples.jsonl", [{"id": k, **v} for k, v in sorted(triples.items())])
    template_path = ROOT / "research_outputs/relgate_v1/data/external_gate_prompt_v1.txt"
    template = template_path.read_text(encoding="utf-8").strip()
    score_inputs = [{"id": k, "instruction": template.format(name=t["name"], description=t["description"] or "",
                     parameters=json.dumps(t["parameters"], ensure_ascii=False, sort_keys=True))} for k, t in sorted(triples.items())]
    jsonl(DEST / dataset / "score_inputs.jsonl", score_inputs)
    prompt_bytes = sum(len(canonical(x).encode()) + 1 for x in prompts.values())
    return {"view_records": len(views), "view_counts": dict(Counter(r["view"] for r in views)),
            "static_logical_units_per_caller": len(schedule), "static_unique_prompts_exact": len(prompts),
            "four_policy_logical_units_per_caller": len(views) * 4,
            "four_policy_unique_prompts_upper_bound": len(prompts) + len(views),
            "unique_candidate_triples_exact": len(triples), "static_prompt_bytes": prompt_bytes,
            "maximum_input_record_bytes_all_masks_bound": maximum_input_bytes,
            "gate_prompt_sha256": sha(template_path)}


def build_budget(bfcl, heldout):
    rates = {}
    for model in ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"):
        path = ROOT / f"research_outputs/step_b_plugplay/raw/{model}/full.report.json"
        r = json.loads(path.read_bytes())
        rates[model] = {"report": str(path.relative_to(ROOT)), "sha256": sha(path), "observed_prompts": r["count"],
                        "full_stage_seconds": r["full_stage_seconds_this_process"],
                        "seconds_per_prompt": r["full_stage_seconds_this_process"] / r["count"]}
    n = bfcl["four_policy_unique_prompts_upper_bound"]
    for model, r in rates.items():
        r["bfcl_conditional_seconds_at_upper_bound"] = n * r["seconds_per_prompt"]
        r["heldout_conditional_seconds_at_upper_bound"] = heldout["all_policies_unique_prompts_upper_bound"] * r["seconds_per_prompt"]
    max_prompt_bytes = bfcl["maximum_input_record_bytes_all_masks_bound"]
    # C2 output cap is 256 new tokens; reserve 64 KiB including raw text,
    # parsed JSON, metadata, and logging per unique prompt, explicitly a budget.
    io = n * (max_prompt_bytes + 65536) * 4
    heldout_io = heldout["all_policies_unique_prompts_upper_bound"] * (heldout["maximum_input_record_bytes_all_masks_bound"] + 65536) * 4
    heldout["four_callers_input_output_reserve_bytes"] = heldout_io
    heldout["max_serial_data_disk_gib_with_one_caller_artifacts"] = 18 + heldout_io / 4 / 2**30
    budget = {"phase": "C1.5 local preparation; C2 requires human authorization", "bfcl": bfcl, "heldout_334": heldout,
              "callers": ["Hammer-Repro 1.5B", "B2 1.5B", *rates], "step_b_reference": rates,
              "batching": {"reference": "vLLM BF16 RTX4090; queued 128, max_num_seqs=16; greedy max_new_tokens=256",
                           "interpretation": "amortized batched wall time, not online batch=1 latency",
                           "C2_condition": "same engine and batching, comparable input/output token lengths; larger schemas or more output tokens increase time",
                           "caller_1_5b": "no comparable Step B throughput measurement; planning proxy only using slower measured 8B rate; not a measured upper bound"},
              "disk": {"units": "GiB", "data_disk_nominal": 27,
                       "model_weight_planning": {"1.5B_BF16": 3.0, "Qwen7B_BF16": 15.0, "Llama8B_BF16": 16.0},
                       "weight_sizes_are": "rounded capacity allowances, not measured download bytes",
                       "bfcl_four_callers_input_and_output_reserve_bytes": io,
                       "bfcl_max_serial_data_disk_gib": 16 + io / 2**30 + 2,
                       "both_datasets_serial_peak_gib_with_one_caller_artifacts": 18 + (io + heldout_io) / 4 / 2**30,
                       "additional_environment_headroom_gib": 2,
                       "serial_plan": ["load frozen gate plus adapter, score unique triples, freeze actions and exact dedup schedules",
                                       "run Hammer-Repro then B2 using one base checkpoint plus adapter where supported",
                                       "run Qwen7B, return artifacts and verify counts/SHA256, delete remote Qwen weights",
                                       "run Llama8B, return artifacts and verify counts/SHA256, delete weights",
                                       "power off immediately after all authorized outputs pass local verification"],
                       "download_cache": "avoid a second physical copy of weights; one large caller at a time"},
              "inference": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256},
              "gate_scoring": {"pending": True, "tau": 0.50, "model": "RelGate-3Way-v2",
                               "bfcl_ms_per_candidate_reference": 98.14505365860909,
                               "bfcl_candidate_score_only_proxy_seconds": bfcl["unique_candidate_triples_exact"] * .09814505365860909,
                               "reference_condition": "Step B serial candidate batch=1, pure label likelihood score; batching changes this estimate; greedy excluded"},
              "manual_gate": "no GPU connection and no C2 launch in this phase"}
    bfcl_ids = {r["id"] for r in read(DEST / "bfcl/unique_candidate_triples.jsonl")}
    heldout_ids = {r["id"] for r in read(DEST / "heldout/unique_candidate_triples.jsonl")}
    envelope_ids = {r["id"] for r in read(DEST / "heldout/desc_degraded_candidate_envelope_NOT_FINAL.jsonl")}
    budget["combined_candidates"] = {
        "bfcl_plus_seven_holdout_views_unique_exact": len(bfcl_ids | heldout_ids),
        "including_unresolved_desc_cohort_upper_bound": len(bfcl_ids | heldout_ids | envelope_ids),
        "not_a_final_eight_view_scoring_manifest": True}
    slower = rates["Llama-3.1-8B-Instruct"]["seconds_per_prompt"]
    budget["bfcl"]["four_callers_conditional_seconds_using_8b_proxy_for_1_5b"] = n * (sum(r["seconds_per_prompt"] for r in rates.values()) + 2 * slower)
    budget["bfcl"]["download_load_and_microbench_time"] = "excluded; download network and caller 1.5B comparable throughput unmeasured"
    write(DEST / "bfcl_executable_budget.json", budget)
    return budget


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BFCL, check=True, capture_output=True, text=True).stdout.strip()
    assert head == REVISION
    subprocess.run(["git", "diff", "--exit-code", REVISION, "--", "bfcl_eval/data", "bfcl_eval/eval_checker/ast_eval"],
                   cwd=BFCL, check=True, capture_output=True)
    scorer_rel = "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py"
    pinned_scorer = subprocess.run(["git", "show", f"{REVISION}:{scorer_rel}"], cwd=BFCL, check=True, capture_output=True).stdout
    working_scorer = (BFCL / "bfcl_eval/eval_checker/ast_eval/ast_checker.py").read_bytes()
    assert pinned_scorer == working_scorer.replace(b"\r\n", b"\n")
    parser, check = frozen_json_parser(ROOT), official_checker(BFCL)
    unit_tests = minimum_tests(check, parser)
    restored, recovery = recover_holdout()
    heldout_materialized = holdout_views(restored)
    views, audit, transformations = bfcl_views(check, parser)
    bfcl = assemble_static(views, "bfcl")
    # C1.5 recovers the population; label-dependent Desc-Degraded cohort is not
    # silently inferred from question text or model outcomes.
    k3 = sum(len(r["record"]["tools"]) >= 3 for r in restored)
    flips = Counter()
    for r in restored:
        row = r["record"]
        target = row["answers"][0].get("name", row["answers"][0].get("tool"))
        if flip_name(target, {t["name"] for t in row["tools"]}):
            flips[re.match(r"^([a-z]+)", target).group(1)] += 1
    # Clean, Coarse, Name-Swapped, Verbose plus two >=3 candidate swaps,
    # target flip hits, and at most 334 Desc-Degraded requests.
    vmax = 4 * 334 + 2 * k3 + sum(flips.values()) + 334
    heldout = {**heldout_materialized, "restored_records": 334, "target_and_distractor_each_eligible": k3,
               "verb_flip_target_hits": sum(flips.values()), "verb_frequencies": dict(flips),
               "eight_view_logical_requests_upper_bound": vmax,
               "four_policy_logical_units_per_caller_upper_bound": 4 * vmax,
               "matched_seeds": [42, 123, 2026], "matched_logical_units_per_caller": 2 * k3 * 3,
               "all_policies_unique_prompts_upper_bound": 4 * vmax + 2 * k3 * 3,
               "unique_candidate_triples": "exact for seven materialized views; eight-view total bounded pending Desc-Degraded cohort",
               "desc_degraded": "frozen exploration implementation uses independently annotated unique-correct cohort; no corresponding 334-ID label manifest found; do not assign new labels",
               "input_bytes_raw_restored_exact": (DEST / "heldout/private/restored_334.jsonl").stat().st_size,
               "disk_reserve_method": "upper_bound_prompts * (max all-mask input record byte bound + 65536) * 4 callers; descriptions removed in pending eighth view cannot increase input bytes"}
    budget = build_budget(bfcl, heldout)
    write(DEST / "heldout_budget.json", heldout)
    report = {"git_head": head, "expected_revision": REVISION, "head_matches": True,
              "official_scorer_matches_git_after_checkout_eol_normalization": True,
              "scorer_git_blob_sha256": hashlib.sha256(pinned_scorer).hexdigest(),
              "scorer_sha256": sha(BFCL / "bfcl_eval/eval_checker/ast_eval/ast_checker.py"),
              "splits": audit, "minimum_ast_tests": unit_tests, "composed_inverse_tests": dict(transformations),
              "representative_controls": "synthetic controls from official accepted values; not model predictions or benchmark scores; two schema/gold inconsistencies retained",
              "adapter": "canonical JSON preserves dots; SDK registry isolated, official scorer unchanged",
              "legacy_candidate_pool_flips": "historical C1 files excluded from C1.5 evaluation, score manifest, and budgets",
              "human_independent_review": "pending", "gpu_used": False, "autodl_connected": False,
              "training": False, "rules_or_thresholds_changed": False,
              "holdout": recovery, "remaining_blockers": ["334 Desc-Degraded eligibility requires original independent label/cohort manifest or explicit protocol decision", "RelGate masks and exact fourth-policy dedup pending authorized C2 scoring", "C2 human authorization pending"],
              "official_data_findings": "two schema/gold field inconsistencies recorded; keep official scorer, all IDs and original labels unchanged"}
    write(DEST / "bfcl_audit.json", report)
    # Supersede the incomplete C1 summaries while preserving legacy raw artifacts.
    write(OUT / "bfcl_audit.json", report)
    write(OUT / "phase_c2_budget.json", budget)
    provenance = {}
    for relative in ("scripts/step_a_probes.py", "scripts/routeb_stage1.py", "scripts/build_diagnostic_views.py",
                     "scripts/build_routeB_swap_probe.py", "scripts/prepare_identifiability_pilot.py", "scripts/lock_phase1.py",
                     "scripts/step_b_caller_runner.py", "scripts/pareto_policies.py",
                     "upstream/Hammer/train/data_processing.py", "research_outputs/phase1_freeze_v1/heldout_ids.json"):
        provenance[relative] = sha(ROOT / relative)
    write(DEST / "frozen_implementation_provenance.json", provenance)
    bfcl_time = budget["bfcl"]["four_callers_conditional_seconds_using_8b_proxy_for_1_5b"] / 60
    gate_time = budget["gate_scoring"]["bfcl_candidate_score_only_proxy_seconds"] / 60
    disk = budget["disk"]["both_datasets_serial_peak_gib_with_one_caller_artifacts"]
    report_text = f"""# Phase C1.5 New Results

- Original 334 recovered by frozen ID membership and source SHA-256: sealtool 194, NexusRaven 140; pilot overlap 0; no resampling or manual content inspection.
- Historical calibration_v3_strict outputs are 200-row pilot outputs. Matching original source bytes were recovered from the local Hammer_release backup.
- Seven heldout views frozen: Clean/Coarse/Name-Swapped/Verbose 334 each, Target/Distractor 309 each, target Verb-Flip 156. Frozen transformations reproduce 1,224 existing exploration view records exactly.
- Remaining heldout blocker: frozen Desc-Degraded eligibility uses independent unique-correct labels; no corresponding 334-ID cohort manifest was found. No replacement labels assigned.
- BFCL actual Git HEAD matches {REVISION}; tracked data and AST scorer unchanged (Windows checkout line endings normalized when comparing scorer bytes).
- Official answers joined by ID: multiple 200, live_multiple 1,053. Missing/extra/duplicate IDs and gold-target pool mismatches: 0.
- True gold-target Verb-Flip hits: multiple 13, live_multiple 164, total 177. The earlier candidate-pool flips are excluded from this evaluation manifest.
- Four original AST tests give True/False/False/False. Unknown names, unknown aliases, malformed JSON and known wrong candidates all fail.
- Composed cyclic swap + selective func_k masking: 42,240 identical official AST results; 204,244 candidate round-trips; argument values untouched.
- Synthetic gold-value scorer controls: 1,251 valid / 1,253; 2 official Schema/gold field inconsistencies archived, retained in all evaluation denominators. These are controls, not model predictions.

## Independent BFCL Budget

| Item | BFCL | 334 holdout |
| --- | ---: | ---: |
| View records | 2,683 exact | 2,110 for seven views; <=2,444 for eight |
| Four-policy logical units per caller | 10,732 | <=9,776 |
| Matched logical units per caller | 0 | 1,854 (three seeds) |
| Three static policies unique prompts | 4,614 exact | 4,182 exact for seven views |
| All-policy unique prompts per caller | <=7,297 | <=11,630 including pending eighth view and Matched |
| Candidate triples | 3,312 exact | 4,265 for seven views; <=5,089 for eight |

BFCL conditional batched inference at the prompt upper bound: Qwen7B 21.9 min, Llama8B 24.2 min. There is no comparable Step B measurement for the two 1.5B callers; using the 8B rate as an explicit planning proxy gives four callers ~{bfcl_time:.1f} min, plus gate pure scoring ~{gate_time:.1f} min. Downloads, model loading and the separate online microbenchmark are excluded. These are conditional projections, not guaranteed wall-time bounds.

Step B reference: vLLM BF16 RTX4090, queued batches 128, max_num_seqs=16, greedy max_new_tokens=256; amortized throughput must not be described as online batch=1 latency. BFCL schemas can change sequence lengths and runtime.

Serial data-disk planning for both datasets: ~{disk:.2f} GiB with one caller's reserved input/output artifacts, a 16 GiB largest-model allowance and 2 GiB environment allowance. Weight allowances are estimates; avoid duplicate physical download caches. Return and SHA-verify each caller's output, then clear its weights/artifacts before the next large caller.

BFCL can be reviewed independently of the heldout cohort blocker. C2 has not been started. RelGate-3Way-v2, tau=0.50, prompt literals, verb dictionary, perturbation logic and Length-64 remain frozen. No AutoDL connection, GPU use, training or model-output evaluation occurred.
"""
    (DEST / "C15_NEW_RESULTS.md").write_text(report_text, encoding="utf-8")
    for name in ("step_c15_prepare.py", "step_c_bfcl_adapter.py"):
        path = ROOT / "scripts" / name
        (DEST / name).write_bytes(path.read_bytes())
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha(p)}  {p.relative_to(OUT).as_posix()}" for p in files]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert sha(OUT / relative) == digest
    print(json.dumps({"restored": recovery["by_source"], "bfcl_gold_audit": audit,
                      "ast_tests": unit_tests, "composed_tests": dict(transformations),
                      "bfcl_budget": bfcl, "heldout_budget": heldout, "hashes_verified": len(files)}, indent=2))


if __name__ == "__main__":
    main()
