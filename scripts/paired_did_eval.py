"""Score Phase 1 predictions. See --help. Arguments use strict JSON equality."""
import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def indexed(records):
    result = {}
    for record in records:
        if record["id"] in result:
            raise ValueError("Duplicate id: " + record["id"])
        result[record["id"]] = record
    return result


def call_parts(call):
    return call.get("name", call.get("tool")), call.get("arguments", call.get("parameters"))


def score(rec, key, renamed):
    if type(rec["parse_ok"]) is not bool:
        raise ValueError("Expected boolean parse_ok")
    gold, args = call_parts(key["ground_truth"][0])
    if renamed:
        mapping = rec["_meta_mapping"]
        funcs = mapping["func_map"]
        if set(funcs) != set(key["candidate_order"]) or len(set(funcs.values())) != len(funcs):
            raise ValueError("Incomplete or non-bijective function map")
        gold_view = funcs[gold]
        params = mapping["param_maps"].get(gold, {})
        if len(set(params.values())) != len(params) or not set(args) <= params.keys():
            raise ValueError("Expected complete, bijective parameter map for gold arguments")
    else:
        gold_view, params = gold, {}
    preds = rec["parsed_pred"]
    if not rec["parse_ok"] or not isinstance(preds, list) or len(preds) != 1 or not isinstance(preds[0], dict):
        return (int(not rec["parse_ok"]), 0, 0)
    name, actual = call_parts(preds[0])
    selected = name == gold_view
    if not selected or not isinstance(actual, dict):
        return (0, int(selected), 0)
    canon = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False)
    expected = {params.get(k, k): v for k, v in args.items()}
    exact = canon(expected) == canon(actual)
    if renamed:
        inverse = {v: k for k, v in params.items()}
        restored = {inverse[k]: v for k, v in actual.items() if k in inverse}
        if exact != (len(restored) == len(actual) and canon(restored) == canon(args)):
            raise ValueError("Forward and inverse scoring disagree")
    return (0, 1, int(exact))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("key", type=Path, help="private/key.jsonl from pilot export")
    p.add_argument("labels", type=Path, help="completed/adjudicated blind.csv")
    p.add_argument("--disagreement-archive", type=Path, help="new private JSONL for unique choices differing from gold")
    for arm in ("mask", "nomask"):
        for view in ("clean", "degraded", "renamed"):
            p.add_argument(f"--{arm}-{view}", required=view != "renamed", type=Path,
                           help="JSONL with id, parse_ok (boolean), parsed_pred (calls); renamed also needs _meta_mapping")
    a = p.parse_args()
    keys = indexed(rows(a.key))
    with a.labels.open(encoding="utf-8-sig", newline="") as f:
        labels = indexed(csv.DictReader(f))
    if labels.keys() != keys.keys():
        raise ValueError("Label IDs do not match sample manifest")
    groups = {}
    disagreements = []
    for i, row in labels.items():
        j = row["judgment"].strip().lower()
        if j not in ("unique", "ambiguous", "unsure"):
            raise ValueError("Missing or invalid human judgment for " + i)
        if j == "unique":
            letter = row["selected_letter"].strip().upper()
            if letter not in json.loads(row["candidates"]):
                raise ValueError("Invalid selected letter for " + i)
            j = "ident" if letter == keys[i]["gold_letter"] else "unique_disagrees_with_gold"
            if j == "unique_disagrees_with_gold":
                disagreements.append(dict(id=i, source=keys[i]["source"], row_index=keys[i]["row_index"],
                                          selected=letter, gold=keys[i]["gold_letter"], reason=row["reason"],
                                          reviewed_cause=None))
        groups.setdefault((keys[i]["source"], j), []).append(i)
    scores = {}
    for arm in ("mask", "nomask"):
        for view in ("clean", "degraded", "renamed"):
            path = getattr(a, arm + "_" + view)
            if path is None:
                continue
            data = indexed(rows(path))
            if data.keys() != keys.keys():
                raise ValueError(f"Missing or extra IDs: {path}")
            scores[arm, view] = {i: score(data[i], key, view == "renamed") for i, key in keys.items()}
    report = []
    for (source, group), ids in sorted(groups.items()):
        summary = {}
        for (arm, view), table in scores.items():
            fail, tool, args = map(sum, zip(*(table[i] for i in ids)))
            summary[arm + "_" + view] = dict(tool_acc=tool/len(ids), parse_fail_rate=fail/len(ids),
                                             arg_acc_given_tool=args/tool if tool else None, tool_correct=tool)
        diffs = [scores["mask", "clean"][i][1] - scores["mask", "degraded"][i][1]
                 - scores["nomask", "clean"][i][1] + scores["nomask", "degraded"][i][1] for i in ids]
        rng = random.Random(42)
        boot = sorted(sum(rng.choices(diffs, k=len(ids))) * 100 / len(ids) for _ in range(10000))
        report.append(dict(source=source, stratum=group, n=len(ids), metrics=summary,
                           extra_drop_pp=100*mean(diffs), request_bootstrap_ci95_pp=[boot[249], boot[9749]]))
    if a.disagreement_archive:
        with a.disagreement_archive.open("x", encoding="utf-8") as f:
            for case in disagreements:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
