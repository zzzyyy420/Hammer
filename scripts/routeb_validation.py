"""Build derived inference inputs, score frozen outputs and compare matched cohorts."""
import argparse
import hashlib
from collections import defaultdict, Counter
from routeb_step2a import OUT, PROBE, OLD, rows, indexed, dump, split_prompt, msg, remap, outcome, save_table, fingerprint
from pathlib import Path
import csv
import json

R = OUT.parent / "routeB_step2b_v1"
VIEWS = ["Clean", "Name-Masked", "Name-Swapped", "Coarse-Desc", "Desc-Degraded", "Target-Swapped", "Distractor-Swapped"]
ARMS = ["B1", "B1-100", "B2"]
BASELINES = ["Pure-NoMask", "Pure-NameMask", "Hammer-1.5b"]
EXCLUDED = {"62aa178d54131a66", "3cf7103a0acae8ca", "71634fd867f412ec"}

def save(path, records):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes("".join(dump(row)+"\n" for row in records).encode())

def tid(tool):
    fields = {k: tool.get(k) for k in ("name", "description", "parameters")}
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), fields

def view_rows(view):
    directory = PROBE/"views" if view == "Name-Masked" else OUT/"eval_views"
    return [r for r in rows(directory/(view+".jsonl")) if r["id"] not in EXCLUDED]

def prepare():
    triples, prompts, cases = {}, {}, []
    aliases = indexed(view_rows("Clean__Always-Mask"))
    for view in VIEWS:
        for row in view_rows(view):
            sid = row["id"]
            _, tools, _ = split_prompt(row)
            names = [t["name"] for t in split_prompt(aliases[sid])[1]]
            assert len(names) == len(tools) == len(set(names))
            ids = []
            for t in tools:
                key, fields = tid(t)
                triples[key] = fields
                ids.append(key)
            h = fingerprint(row)
            prompts[h] = dict(id=h, messages=row["messages"])
            cases.append(dict(id=sid, view=view, prompt=h, triples=ids, aliases=names))
    save(R/"inputs/self_gate.jsonl", (dict(id=h, **t) for h,t in sorted(triples.items())))
    save(R/"inputs/keep.jsonl", prompts.values())
    save(R/"cases.jsonl", cases)
    print("triples",len(triples),"prompts",len(prompts),dict(Counter(c["view"] for c in cases)))

def labels():
    return {r["id"]:r["raw"].strip() for r in rows(R/"results/self_gate.jsonl")}

def gated():
    decisions = labels()
    assert set(decisions) == set(indexed(rows(R/"inputs/self_gate.jsonl")))
    clean = indexed(rows(R/"inputs/keep.jsonl"))
    requests, entries = {}, []
    for case in rows(R/"cases.jsonl"):
        if case["view"] == "Name-Masked":
            continue
        original = clean[case["prompt"]]
        prefix, tools, suffix = split_prompt(original)
        inverse = {}
        for t,alias,key in zip(tools, case["aliases"], case["triples"], strict=True):
            assert tid(t)[0] == key
            if decisions[key] == "CONTRADICTED":
                inverse[alias] = t["name"]
                t["name"] = alias
        assert len({t["name"] for t in tools}) == len(tools)
        messages = msg(prefix+dump(tools)+suffix) if inverse else original["messages"]
        h = fingerprint(dict(messages=messages))
        requests[h] = dict(id=h,messages=messages)
        entries.append(dict(id=case["id"],view=case["view"],prompt=h,inverse=inverse))
    save(R/"inputs/gated.jsonl",requests.values())
    save(R/"gated_cases.jsonl",entries)

def existing(model):
    cache = indexed(rows(OUT/"schedule"/model/"cached.jsonl"))
    cache.update(indexed(rows(OUT/"predictions"/(model+".jsonl"))))
    original = indexed(rows(PROBE/"predictions"/model/"Name-Masked.jsonl"))
    for row in view_rows("Name-Masked"):
        cache.setdefault(fingerprint(row),original[row["id"]])
    return cache

def gold():
    lookup = {(k["view"],k["id"]):k for k in rows(OUT/"eval_keys.jsonl")
              if k["mode"]=="Keep-Name" and k["id"] not in EXCLUDED}
    lookup.update({("Name-Masked",k["id"]):k for k in rows(PROBE/"private/Name-Masked.jsonl") if k["id"] not in EXCLUDED})
    return lookup

def quality(keys):
    decision = labels()
    counts = defaultdict(Counter)
    detail = []
    for c in rows(R/"cases.jsonl"):
        v = c["view"]
        flags = keys[v,c["id"]].get("flags")
        for slot,h in enumerate(c["triples"]):
            raw = decision[h]
            bad = raw not in {"SUPPORTED","CONTRADICTED","UNVERIFIED"}
            hit = raw == "CONTRADICTED"
            target = None if flags is None else flags[slot]
            detail.append(dict(id=c["id"],view=v,slot=slot,triple=h,raw=raw,parse_failure=bad,masked=hit,swapped=target))
            for group in (v,"all"):
                cc=counts[group]
                cc["n"]+=1
                cc["invalid"]+=bad
                if target is not None:
                    cc["tp" if target and hit else "fn" if target else "fp" if hit else "tn"]+=1
    save(R/"report/gate_candidate_occurrences.jsonl",detail)
    table=[]
    def div(a,b): return a/b if b else None
    for v,x in counts.items():
        tp,fp,tn,fn = [x[key] for key in ("tp","fp","tn","fn")]
        table.append(dict(view=v,n=x["n"],tp=tp,fp=fp,tn=tn,fn=fn,precision=div(tp,tp+fp),recall=div(tp,tp+fn),
                          fpr=div(fp,fp+tn),invalid=x["invalid"],parse_failure_rate=x["invalid"]/x["n"]))
    save_table(R/"report/gate_quality.csv",table)

def summarize(model,mode,cases,preds,keys):
    vals=defaultdict(list)
    records=[]
    for c in cases:
        k=keys[c["view"],c["id"]]
        p=preds[c["prompt"]]
        res=outcome(dict(p,parsed_pred=remap(p["parsed_pred"],c.get("inverse",{}))),k)
        vals[c["view"]].append(res)
        records.append(dict(id=c["id"],view=c["view"],source=k["source"],prompt=c["prompt"],
                            raw=p.get("raw"),inverse=c.get("inverse",{}),tool_correct=res[0],strict_correct=res[1],
                            name_hijack=res[2] if c["view"] in ("Name-Swapped","Target-Swapped") else None,
                            invalid=res[3]))
    save(R/"report/predictions"/(model+"__"+mode+".jsonl"),records)
    result=[]
    for v,data in vals.items():
        correct,strict,hij,invalid=map(sum,zip(*data))
        n=len(data)
        defined=v in ("Name-Swapped","Target-Swapped")
        result.append(dict(model=model,mode=mode,view=v,n=n,correct=correct,tool_acc=correct/n,strict_correct=strict,
                           strict_acc=strict/n,invalid=invalid,hijacks=hij if defined else None,
                           nhr_abs=hij/n if defined else None,nhr_err=hij/(n-correct) if defined and n>correct else None))
    return result

def score(final):
    report=R/"report"
    report.mkdir(exist_ok=True)
    keys=gold()
    cases=rows(R/"cases.jsonl")
    metrics=[]
    for arm in ARMS:
        metrics+=summarize(arm,"Keep-Name",cases,indexed(rows(R/"results"/(arm+".jsonl"))),keys)
    for model in BASELINES:
        cache=existing(model)
        metrics+=summarize(model,"Keep-Name",cases,cache,keys)
        with (OUT/"report/best_length_diagnostic.csv").open(encoding="utf-8-sig") as f:
            best={r["view"]:r["mode"] for r in csv.DictReader(f) if r["model"]==model and r["group"]=="all"}
        inputs={}
        selected=defaultdict(list)
        for k in rows(OUT/"eval_keys.jsonl"):
            if k["id"] in EXCLUDED: continue
            mode=k["mode"]
            if mode not in ("Always-Mask","Oracle",best.get(k["view"])): continue
            stem=k["stem"]
            if stem not in inputs: inputs[stem]=indexed(view_rows(stem))
            selected[mode].append(dict(id=k["id"],view=k["view"],prompt=fingerprint(inputs[stem][k["id"]]),inverse=k["inverse"]))
        for mode,cc in selected.items(): metrics+=summarize(model,mode,cc,cache,keys)
        frozen_length=[dict(r,mode="Best-Desc-Length",length_tau=int(r["mode"].split("-")[1])) for r in metrics
                       if r["model"]==model and r["mode"]==best.get(r["view"])]
        metrics.extend(frozen_length)
    hs={}
    for m in [*ARMS,*BASELINES]:
        acc={r["view"]:r["tool_acc"] for r in metrics if r["model"]==m and r["mode"]=="Keep-Name"}
        a,b=acc["Coarse-Desc"],acc["Name-Swapped"]
        hs[m]=2*a*b/(a+b) if a+b else 0
    best=max(ARMS,key=hs.get)
    (report/"selection.json").write_text(json.dumps(dict(best=best,H_dual=hs,positive="Coarse-Desc",negative="Name-Swapped",
                                                          tie_order=ARMS),indent=2))
    if final:
        for m in ("Pure-NameMask",best):
            src=existing(m) if m=="Pure-NameMask" else indexed(rows(R/"results"/(m+".jsonl")))
            file=R/"results"/(m+"__gated.jsonl")
            if file.exists(): src.update(indexed(rows(file)))
            metrics+=summarize(m,"Self-Gate",rows(R/"gated_cases.jsonl"),src,keys)
        quality(keys)
    for model in [*ARMS,*BASELINES]:
        for mode in {r["mode"] for r in metrics if r["model"]==model}:
            pair={r["view"]:r["tool_acc"] for r in metrics if r["model"]==model and r["mode"]==mode}
            a,b=pair.get("Coarse-Desc"),pair.get("Name-Swapped")
            for r in metrics:
                if r["model"]==model and r["mode"]==mode:
                    r["H_dual"]=None if a is None or b is None else 2*a*b/(a+b) if a+b else 0
    for r in metrics:
        r.setdefault("length_tau",int(r["mode"].split("-")[1]) if r["mode"].startswith("Length-") else None)
    save_table(report/"metrics.csv",metrics)
    return best

def gate_requests():
    best=score(False)
    gated()
    for m in ("Pure-NameMask",best):
        cache=existing(m) if m=="Pure-NameMask" else indexed(rows(R/"results"/(m+".jsonl")))
        save(R/"inputs"/(m+"__gated.jsonl"),(r for r in rows(R/"inputs/gated.jsonl") if r["id"] not in cache))

if __name__=="__main__":
    mode=argparse.ArgumentParser()
    mode.add_argument("action",choices=["prepare","gate_requests","score"])
    action=mode.parse_args().action
    if action=="prepare": prepare()
    elif action=="gate_requests": gate_requests()
    else: score(True)
