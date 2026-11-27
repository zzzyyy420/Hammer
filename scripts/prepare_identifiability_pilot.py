"""Export a blind annotation sheet from Hammer's previously evaluated test suites.

Run: python scripts/prepare_identifiability_pilot.py SOURCE_DIR NEW_OUTPUT_DIR
Distribute only blind.csv. Fill judgment with unique/ambiguous/unsure and, for
unique, the selected letter. Lock independent labels before opening private/.
Use a unique judgment for S_ident only after comparing the choice with gold.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import string
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature(tool):
    params = tool["parameters"]
    pairs = ((p["name"], p.get("type")) for p in params) if isinstance(params, list) else (
        (k, v.get("type")) for k, v in params.items()
    )
    return dict(name=tool["name"], parameters=list(pairs))


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def eligible_rows(raw):
    eligible = []
    seen = set()
    for index, row in enumerate(json.loads(raw)):
        tools, answers = row["tools"], row["answers"]
        if len(tools) < 2 or len(answers) != 1:
            continue
        names = [t["name"] for t in tools]
        gold = answers[0].get("name", answers[0].get("tool"))
        if gold not in names or len(set(names)) != len(names):
            continue
        if any(isinstance(t["parameters"], list) and
               any(not isinstance(p.get("name"), str) or not p["name"].strip() for p in t["parameters"])
               for t in tools):
            continue
        if any(re.fullmatch(r"(?:func(?:tion)?|api|tool)[_-]?(?:\d+|[a-z])", n, re.I) for n in names):
            continue
        query_key = " ".join(row["query"].casefold().split())
        if query_key in seen:
            continue
        seen.add(query_key)
        eligible.append((index, row, gold))
    return eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    blind, keys, stats = [], [], {}
    for source in ("sealtool", "NexusRaven"):
        raw = (args.source / (source + ".json")).read_bytes()
        eligible = eligible_rows(raw)
        if not 0 < args.per_source <= len(eligible):
            raise ValueError(f"{source}: requested {args.per_source} of {len(eligible)} eligible")
        stats[source] = dict(sha256=digest(raw), eligible=len(eligible), sampled=args.per_source)
        for index, row, gold in rng.sample(eligible, args.per_source):
            sid = digest(f"{source}/{index}".encode())[:16]
            tools = list(row["tools"])
            rng.shuffle(tools)
            if len(tools) > len(string.ascii_uppercase):
                raise ValueError("More than 26 choices")
            choices = {letter: signature(t) for letter, t in zip(string.ascii_uppercase, tools)}
            choice = next(k for k, v in choices.items() if v["name"] == gold)
            blind.append(dict(id=sid, query=row["query"], candidates=encode(choices),
                              judgment="", selected_letter="", reason=""))
            keys.append(dict(id=sid, source=source, row_index=index, ground_truth=row["answers"],
                             gold_letter=choice, candidate_order=[t["name"] for t in tools]))
    rng.shuffle(blind)
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / "private"
    private.mkdir()
    with (args.output / "blind.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(blind[0]))
        writer.writeheader()
        writer.writerows(blind)
    (private / "key.jsonl").write_text("".join(encode(k) + "\n" for k in keys), encoding="utf-8")
    (args.output / "manifest.json").write_text(encode(dict(seed=args.seed, role="exploratory_from_test",
                                                         sources=stats)), encoding="utf-8")


if __name__ == "__main__":
    main()
