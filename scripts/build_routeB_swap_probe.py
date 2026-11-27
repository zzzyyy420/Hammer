"""Build the frozen diagnostic probe without reading held-out samples."""
import hashlib
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from paired_did_eval import call_parts, indexed, rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research_outputs"
DEST = DATA / "routeB_probe_v1"
VIEWS = ("Name-Swapped", "Partial-Swap", "Name-Masked")
BEGIN = "[BEGIN OF AVAILABLE TOOLS]\n"
END = "\n[END OF AVAILABLE TOOLS]"
COARSE = "Tool for operating on the requested resource with the specified parameters."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_prompt(row):
    prefix, block = row["messages"][0]["content"].split(BEGIN, 1)
    block, suffix = block.split(END, 1)
    return prefix + BEGIN, json.loads(block), END + suffix


def tokens(name):
    return set(re.findall("[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).lower()))


def overlap(a, b):
    return len(a & b) / max(len(a | b), 1)


def signature(t):
    return {(p, str(x.get("type"))) for p, x in t["parameters"].items()}


def design(row, key, aliases):
    prefix, tools, suffix = split_prompt(row)
    names = [t["name"] for t in tools]
    gold, args = call_parts(key["ground_truth"][0])
    assert len(key["ground_truth"]) == 1 and names == key["candidate_order"]
    assert len(set(names)) == len(names) >= 2 and set(aliases) == set(names)
    g = names.index(gold)
    rng = random.Random("routeB/42/" + row["id"])
    others = [i for i in range(len(names)) if i != g]
    rng.shuffle(others)
    h = max(others, key=lambda i: overlap(signature(tools[g]), signature(tools[i]))
            + overlap(tokens(gold), tokens(names[i])))
    rest = [i for i in others if i != h]
    partial = list(range(len(names)))
    partial[g], partial[h] = h, g
    full = partial.copy()
    if len(rest) == 1:
        full[h], full[rest[0]] = rest[0], g
    elif rest:
        for a, b in zip(rest, rest[1:] + rest[:1]):
            full[a] = b
    outputs = {}
    for view in VIEWS:
        perm = full if view == "Name-Swapped" else partial
        altered = deepcopy(tools)
        for i, t in enumerate(altered):
            name = names[i]
            t["name"] = aliases[name] if view == "Name-Masked" else names[perm[i]]
            desc = t["description"]
            if view == "Partial-Swap" and i in rest:
                t["description"] = COARSE
            elif desc:
                t["description"] = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                                          "this function", desc, flags=re.I)
        prompt = prefix + json.dumps(altered, ensure_ascii=False) + suffix
        inp = dict(id=row["id"], messages=[dict(role="user", content=prompt)])
        label = dict(id=row["id"], source=key["source"], original_gold=gold,
                     ground_truth=[dict(name=altered[g]["name"], arguments=args)],
                     g=g, h=h, k=len(names))
        outputs[view] = inp, label
    return outputs


def main():
    clean = DATA / "phase1_diag_views_v1"
    keyfile = DATA / "phase1_gpu_v1/key.jsonl"
    hashes = json.loads((clean/"manifest.json").read_text())["hashes"]
    for f in ("Clean.jsonl", "private/renamed_maps.jsonl"):
        assert digest(clean/f) == hashes[f]
    inputs = indexed(rows(clean/"Clean.jsonl"))
    keys = indexed(rows(keyfile))
    maps = indexed(rows(clean/"private/renamed_maps.jsonl"))
    assert len(inputs) == 200 and inputs.keys() == keys.keys() == maps.keys()
    with TemporaryDirectory(dir=DATA) as tmp:
        out = Path(tmp)/"probe"
        for folder in ("views", "private"):
            (out/folder).mkdir(parents=True)
        sizes = Counter()
        for i, row in inputs.items():
            for view, pair in design(row, keys[i], maps[i]["_meta_mapping"]["func_map"]).items():
                for dirname, obj in zip(("views", "private"), pair):
                    with (out/dirname/(view+".jsonl")).open("a", encoding="utf8", newline="\n") as f:
                        f.write(json.dumps(obj, ensure_ascii=False)+"\n")
            sizes[len(keys[i]["candidate_order"])] += 1
        manifest = dict(rows=len(inputs), pool_size_histogram=sizes, seed=42,
                        input_sha256={**{f:hashes[f] for f in ("Clean.jsonl","private/renamed_maps.jsonl")},
                                      "key.jsonl":digest(keyfile)},
                        sha256={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob("*.jsonl"))})
        (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        if DEST.exists():
            assert json.loads((DEST/"manifest.json").read_text()) == json.loads((out/"manifest.json").read_text())
            assert all(digest(DEST/f) == v for f,v in manifest["sha256"].items())
        else:
            out.rename(DEST)
        print(json.dumps(manifest))


if __name__ == "__main__":
    main()
