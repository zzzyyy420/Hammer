"""Materialize paired Hammer inference inputs; keep answer keys outside prompts."""
import argparse
import csv
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from build_pure_name_pair import UPSTREAM, alias, dump, rewrite, sft, sha, strings
from paired_did_eval import call_parts, score
from prepare_identifiability_pilot import signature


def normalized(t, descriptions):
    params = t["parameters"]
    entries = ((p["name"], p) for p in params) if isinstance(params, list) else params.items()
    result = {"name": t["name"], "parameters": {}}
    for name, p in entries:
        result["parameters"][name] = {"type": p.get("type")}
        if descriptions:
            result["parameters"][name]["description"] = p.get("description")
    if descriptions:
        result["description"] = t.get("description")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pilot", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    if a.output.exists():
        parser.error("Supply a new output folder")
    cfg = json.loads((a.pilot/"manifest.json").read_bytes())
    data = {}
    for name, spec in cfg["sources"].items():
        raw = (a.source/(name+".json")).read_bytes()
        if sha(raw) != spec["sha256"]:
            raise ValueError("Source hash mismatch: "+name)
        data[name] = json.loads(raw)
    with (a.pilot/"blind.csv").open(encoding="utf-8-sig", newline="") as f:
        blind = {r["id"]:r for r in csv.DictReader(f)}
    prompt = strings(UPSTREAM)
    files = defaultdict(list)
    keys = [json.loads(x) for x in (a.pilot/"private/key.jsonl").read_bytes().splitlines()]
    if len(keys)!=200 or {k["id"] for k in keys}!=set(blind):
        raise ValueError("Pilot ID mismatch")
    for key in keys:
        sid = key["id"]
        row = data[key["source"]][key["row_index"]]
        by_name = {t["name"]:t for t in row["tools"]}
        ordered = [by_name[n] for n in key["candidate_order"]]
        choices = list(json.loads(blind[sid]["candidates"]).values())
        if row["query"] != blind[sid]["query"] or json.loads(dump([signature(t) for t in ordered])) != choices:
            raise ValueError("Query or candidate signature mismatch: "+sid)
        clean = [normalized(t, True) for t in ordered]
        deg = [normalized(t, False) for t in ordered]
        rng = random.Random("phase1/eval/42/"+sid)
        occupied = set(by_name) | {p for t in clean for p in t["parameters"]}
        fn = {n:alias(rng,occupied,5,15) for n in key["candidate_order"]}
        pm = {t["name"]:{p:alias(rng,occupied,4,10) for p in t["parameters"]} for t in clean}
        n, args = call_parts(key["ground_truth"][0])
        renamed, gold = rewrite(clean,[{"name":n,"arguments":args}],fn,pm)
        mapping = dict(func_map=fn,param_maps=pm)
        if score(dict(parse_ok=True,parsed_pred=gold,_meta_mapping=mapping),key,True)!=(0,1,1):
            raise ValueError("Renamed gold failed canonical scorer: "+sid)
        files["private/renamed_maps"].append(dict(id=sid,_meta_mapping=mapping))
        for view, ts in (("Clean",clean),("Name-Masked",renamed),("Desc-Degraded",deg)):
            instruction = sft(row["query"],ts,[],prompt)["instruction"]
            files[view].append(dict(id=sid,messages=[dict(role="user",content=instruction)]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=a.output.parent) as tmp:
        stage = Path(tmp)/"views"
        (stage/"private").mkdir(parents=True)
        hashes = {}
        for stem, rows in files.items():
            path = stage/(stem+".jsonl")
            path.write_bytes("".join(dump(r)+"\n" for r in rows).encode())
            hashes[stem+".jsonl"]=sha(path.read_bytes())
        (stage/"manifest.json").write_text(json.dumps(dict(rows_per_view=len(keys),hashes=hashes,
            normalization="all arms use ordered signature plus optional tool/parameter descriptions; defaults, required and returns omitted throughout",
            degraded="drop all tool and parameter descriptions", renamed="rename schema identifiers only; retain natural-language descriptions",
            template="Hammer v1 instruction in user message; apply training qwen chat template for inference"),indent=2),encoding="utf-8")
        stage.rename(a.output)


if __name__ == "__main__":
    main()
