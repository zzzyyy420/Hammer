"""Build paired Hammer Alpaca JSONL inputs from checksum-pinned official files."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import string
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "configs/hammer15b_train_v1/raw_data_audit.json"
UPSTREAM = ROOT / "upstream/Hammer/train/data_processing.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strings(source):
    # Read literal prompt constants without executing upstream's training-data side effects.
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"task_instruction_hammer", "format_instruction_hammer"}}


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


@lru_cache(maxsize=4096)
def pattern(names):
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)", re.I)


def leak_hits(query, tools):
    fns = {t["name"] for t in tools}
    params = {k for t in tools for k in t["parameters"]}
    desc = "\n".join(s for t in tools for s in [t.get("description")] +
                     [v.get("description") for v in t["parameters"].values()] if isinstance(s, str))
    hits = []
    for kind, names in (("function", fns), ("parameter", params)):
        if not names:
            continue
        styled = {n.casefold() for n in names if "_" in n or re.search(r"[a-z][A-Z]|[A-Z][A-Z][a-z]", n)}
        regex = pattern(tuple(sorted(names)))
        for field, s in (("query", query), ("description", desc)):
            matches = {m.group().casefold() for m in regex.finditer(s)}
            if matches:
                hits.append(f"{kind}_in_{field}")
            if matches & styled:
                hits.append(f"styled_{kind}_in_{field}")
    return hits


def alias(rng, occupied, lo, hi):
    while True:
        name = "".join(rng.choices(string.ascii_letters + string.digits + "_.", k=rng.randint(lo, hi)))
        if name not in occupied:
            occupied.add(name)
            return name


def rewrite(tools, answers, fn, pm):
    out, labels = deepcopy(tools), deepcopy(answers)
    for t in out:
        name = t["name"]
        t["name"] = fn[name]
        t["parameters"] = {pm[name][k]: v for k, v in t["parameters"].items()}
        if "required" in t:
            t["required"] = [pm[name][k] for k in t["required"]]
    for a in labels:
        name = a["name"]
        a["name"] = fn[name]
        a["arguments"] = {pm[name][k]: v for k, v in a["arguments"].items()}
    return out, labels


def mask(tools, answers, rng):
    occupied = {t["name"] for t in tools} | {k for t in tools for k in t["parameters"]}
    fn, pm = {}, {}
    for t in tools:
        name = t["name"]
        if name not in fn:
            fn[name] = alias(rng, occupied, 5, 15)
            pm[name] = {}
        for k in t["parameters"]:
            if k not in pm[name]:
                pm[name][k] = alias(rng, occupied, 4, 10)
    out, labels = rewrite(tools, answers, fn, pm)
    inv = {v: k for k, v in fn.items()}
    back = {fn[n]: {v: k for k, v in m.items()} for n, m in pm.items()}
    if dump(rewrite(out, labels, inv, back)) != dump((tools, answers)):
        raise ValueError("Inverse transform changed row content or order")
    return out, labels


def sft(query, tools, answers, prompt):
    sections = [("TASK INSTRUCTION", prompt["task_instruction_hammer"]),
                ("AVAILABLE TOOLS", dump(tools)), ("FORMAT INSTRUCTION", prompt["format_instruction_hammer"]),
                ("QUERY", query)]
    content = "".join(f"[BEGIN OF {n}]\n{v}\n[END OF {n}]\n\n" for n, v in sections)
    fence = chr(96)*3
    return dict(instruction=content, input="", output=f"{fence}\n{dump(answers)}\n{fence}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("xlam", "irrelevance", "output"):
        p.add_argument(n, type=Path)
    p.add_argument("--size", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists() or not 9 <= a.size <= 67500:
        p.error("Specify a new output directory and size 9..67500")
    audit = json.loads(AUDIT.read_bytes())
    pools, hits, stats = {}, {}, {}
    for pool, path in (("xlam60k", a.xlam), ("irrelevance", a.irrelevance)):
        raw = path.read_bytes()
        if sha(raw) != audit[pool]["sha256"]:
            raise ValueError(f"{pool}: SHA-256 mismatch")
        data = json.loads(raw)
        if len(data) != audit[pool]["count"]:
            raise ValueError(f"{pool}: row count mismatch")
        pools[pool] = data
        counter = Counter()
        for i, r in enumerate(data):
            tools, answers = json.loads(r["tools"]), json.loads(r["answers"])
            if (pool == "irrelevance") != (len(answers) == 0):
                raise ValueError(f"Unexpected call/refusal label in {pool}:{i}")
            h = leak_hits(r["query"], tools)
            hits[pool,i] = h
            counter.update(h)
            counter["any"] += bool(h)
        stats[pool] = dict(rows=len(data), sha256=sha(raw), literal_match_rows=dict(counter))
        del raw
    rate = sum(bool(h) for h in hits.values())/len(hits)
    exclude = rate < 0.02
    rng = random.Random(a.seed)
    order = []
    nneg = round(a.size/9)
    for pool, n in (("xlam60k", a.size-nneg), ("irrelevance", nneg)):
        eligible = [i for i in range(len(pools[pool])) if not exclude or not hits[pool,i]]
        order.extend((pool,i) for i in rng.sample(eligible,n))
    rng.shuffle(order)
    prompt = strings(UPSTREAM)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.output.parent) as tmp:
        stage = Path(tmp)/"paired"
        stage.mkdir()
        counts = Counter()
        with (stage/"Pure-NoMask.jsonl").open("w",encoding="utf-8") as clean, (
            stage/"Pure-NameMask.jsonl").open("w",encoding="utf-8") as masked, (
            stage/"paired_rows.jsonl").open("w",encoding="utf-8") as ledger:
            for pool,i in order:
                r = pools[pool][i]
                ts, ans = json.loads(r["tools"]), json.loads(r["answers"])
                line = dump(sft(r["query"], ts, ans, prompt))+"\n"
                clean.write(line)
                arm_rng = random.Random(f"{a.seed}/{pool}/{i}")
                active = arm_rng.random() < 2/3
                if active:
                    t2, a2 = mask(ts, ans, arm_rng)
                    line = dump(sft(r["query"], t2, a2, prompt))+"\n"
                masked.write(line)
                counts[pool+"_masked"] += active
                counts[pool+"_literal_hits"] += bool(hits[pool,i])
                ledger.write(dump(dict(pool=pool,index=i,masked=active,literal_hits=hits[pool,i]))+"\n")
        hashes = {}
        for path in stage.iterdir():
            with path.open("rb") as f:
                h = hashlib.sha256()
                for block in iter(lambda:f.read(1<<20), b""):
                    h.update(block)
                hashes[path.name] = h.hexdigest()
        (stage/"dataset_info.json").write_text(dump({n:{"file_name":n+".jsonl"} for n in
                                                     ("Pure-NoMask","Pure-NameMask")}),encoding="utf-8")
        (stage/"manifest.json").write_text(dump(dict(seed=a.seed, rows=a.size, negatives=nneg,
            mask_probability=2/3, upstream_sha256=sha(UPSTREAM.read_bytes()), builder_sha256=sha(Path(__file__).read_bytes()),
            any_literal_rate=rate, excluded_literal_matches=exclude, pools=stats,
            selected_counts=counts, files_sha256=hashes)),encoding="utf-8")
        stage.rename(a.output)


if __name__ == "__main__":
    main()
