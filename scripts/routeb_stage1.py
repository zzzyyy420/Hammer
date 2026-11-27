"""Frozen Route B transformations. Keep private labels out of model inputs."""
import csv
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack

from build_routeB_swap_probe import DATA, split_prompt, signature, overlap, tokens, digest
from build_pure_name_pair import alias
from paired_did_eval import call_parts, indexed, rows

OUT = DATA/"routeB_stage1_v1"
AUDIT = DATA/"phase1_cpu_20261124/audited"
PROBE = DATA/"routeB_probe_v1"
DIAG = DATA/"phase1_diag_views_v1"
OLD = DATA/"phase1_gpu_v1"
FILES = ["train_b1_swap50_eligible", "train_b1_swap100_eligible", "train_b2_balanced_mix_eligible"]
TAUS = (32, 64, 128)
STOP = set("""a the of on to for from in at and with by as is v api this function tool endpoint given provided specified information data get post put set fetch list find create update delete remove add check calculate start stop search query read write new all status current details detail info retrieve returns return process processes manage manages perform performs operating predict generate validate edit make run do execute submit configure identify estimate assess analyze recommend download upload extract determine provide convert operation operations random specific using operates gets retrieves access use has can calls an or based parameters object requested your will be it that these takes used you""".split())


def dump(x):
    return json.dumps(x, ensure_ascii=False)


def msg(text):
    return [{"role":"user","content":text}]


def unpack(text):
    return split_prompt({"messages":msg(text)})


def cleaned(desc, name):
    return re.sub(r"(?<!\w)"+re.escape(name)+r"(?!\w)", "this function", desc, flags=re.I) if desc else desc


def words(text):
    return re.findall("[a-z]+", re.sub(r"([a-z0-9])([A-Z])",r"\1 \2",text or "").lower())


def coarse(tools):
    # Shared resource words take priority over tool-specific modifiers.
    names = [t["name"] for t in tools]
    groups = []
    for t in tools:
        parts = words(t["name"])
        nouns = [w for w in parts if w not in STOP and len(w)>1]
        if len(parts)>1 and parts[0] in nouns and parts[0] not in {"vt","emailrep"}:
            # Strip a leading action when the description repeats its verb stem.
            first = words(t.get("description"))[:1]
            if first and first[0].rstrip("s")==parts[0]:
                nouns.remove(parts[0])
        groups.append(nouns)
    counts = Counter(w for group in groups for w in set(group))
    out = []
    for t, group in zip(tools,groups):
        common = [w for w in group if counts[w]>1]
        resource = max(common,key=lambda w: counts[w]) if common else " ".join(group[-2:])
        if not resource:
            desc_words = [w for w in words(t.get("description"))[1:] if w not in STOP and len(w)>2]
            params = [w for p in t["parameters"] for w in words(p) if w not in STOP]
            resource = next(iter(params or desc_words),"resource")
        out.append(f"Manage {resource} using the provided parameters.")
    return out


def permutation(tools,g,sid):
    rng=random.Random("routeB/42/"+sid)
    rest=[i for i in range(len(tools)) if i!=g]
    rng.shuffle(rest)
    def similarity(i,j):
        return overlap(signature(tools[i]),signature(tools[j]))+overlap(tokens(tools[i]["name"]),tokens(tools[j]["name"]))
    h=max(rest,key=lambda i:similarity(g,i))
    tail=[i for i in rest if i!=h]
    full=list(range(len(tools)))
    full[g],full[h]=h,g
    if len(tail)==1:
        full[h],full[tail[0]]=tail[0],g
    elif tail:
        for a,b in zip(tail,tail[1:]+tail[:1]):
            full[a]=b
    return full,h,tail,similarity


def calls(row):
    return json.loads(row["output"].strip().strip(chr(96)).strip())


def eligibility(ts,answers):
    names=[t["name"] for t in ts]
    gs={c["name"] for c in answers}
    return len(gs)==1 and gs<=set(names) and bool(set(names)-gs) and len(names)==len(set(names))


def names_mask(ts,sid):
    rng=random.Random("phase1/eval/42/"+sid)
    occupied={t["name"] for t in ts}|{p for t in ts for p in t["parameters"]}
    return [alias(rng,occupied,5,15) for _ in ts]


def transformed(row,sid,kind):
    pre,ts,post=unpack(row["instruction"])
    ans=calls(row)
    names=[t["name"] for t in ts]
    if kind=="swap":
        perm,*_=permutation(ts,names.index(ans[0]["name"]),sid)
        new=[names[i] for i in perm]
    elif kind=="mask":
        new=names_mask(ts,sid)
    else:
        new=names
    descs=coarse(ts) if kind=="coarse" else [cleaned(t.get("description"),n) for t,n in zip(ts,names)]
    for t,n,d in zip(ts,new,descs):
        t["name"]=n
        if "description" in t or d is not None:
            t["description"]=d
    lookup=dict(zip(names,new))
    for c in ans:
        c["name"]=lookup[c["name"]]
    return dict(row,instruction=pre+dump(ts)+post,output=chr(96)*3+"\n"+dump(ans)+"\n"+chr(96)*3)


def train(out):
    raw=AUDIT/"Pure-NoMask.jsonl"
    manifest=json.loads((AUDIT/"cutoff_report.json").read_bytes())["training_input_sha256"]
    for p in (raw,AUDIT/"paired_rows.jsonl"):
        assert digest(p)==manifest[p.name]
    meta=rows(AUDIT/"paired_rows.jsonl")
    entries=[]
    with raw.open(encoding="utf8") as f:
        for line,m in zip(f,meta,strict=True):
            r=json.loads(line)
            sid=f"{m['pool']}:{m['index']}"
            if eligibility(unpack(r["instruction"])[1],calls(r)):
                entries.append((m["pool"],sid))
    # The locked training pool exposes exactly one eligible provenance stratum.
    assert len(entries)==10697 and {s for s,_ in entries}=={"xlam60k"}
    ids=sorted(i for _,i in entries)
    assert len(ids)==len(set(ids))
    rng=random.Random(42)
    rng.shuffle(ids)
    half=set(ids[:5349])
    order=sorted(ids)
    rng.shuffle(order)
    mix={i:("clean","mask","swap","coarse")[j%4] for j,i in enumerate(order)}
    with ExitStack() as stack:
        handles=[stack.enter_context((out/(n+".jsonl")).open("wb")) for n in FILES]
        log=stack.enter_context((out/"assignment.jsonl").open("wb"))
        with raw.open("rb") as f:
            for line,m in zip(f,meta,strict=True):
                r=json.loads(line)
                sid=f"{m['pool']}:{m['index']}"
                kinds=["swap" if sid in half else "clean","swap" if sid in mix else "clean",mix.get(sid,"clean")]
                variants={"clean":line}
                for k,h in zip(kinds,handles):
                    if k not in variants:
                        variants[k]=(dump(transformed(r,sid,k))+"\n").encode()
                    h.write(variants[k])
                log.write((dump(dict(id=sid,source=m["pool"],eligible=sid in mix,arms=kinds))+"\n").encode())
    (out/"dataset_info.json").write_text(dump({f:{"file_name":f+".jsonl"} for f in FILES}))


def remap(calls_,mapping):
    result=deepcopy(calls_)
    if isinstance(result,list):
        for c in result:
            if isinstance(c,dict):
                for k in ("tool","name"):
                    if isinstance(c.get(k),str):
                        c[k]=mapping.get(c[k],c[k])
    return result


def wrapper(ts,aliases,swapped,mode):
    new=deepcopy(ts)
    inverse={}
    for i,t in enumerate(new):
        active=(mode=="Always-Mask" or mode=="Oracle" and swapped[i] or
                mode.startswith("Length-") and len(t.get("description") or "")>int(mode.split("-")[1]))
        if active:
            inverse[aliases[i]]=t["name"]
            t["name"]=aliases[i]
    assert len({t["name"] for t in new})==len(new)
    return new,inverse


def evaluation(out):
    clean=indexed(rows(DIAG/"Clean.jsonl"))
    keys=indexed(rows(OLD/"key.jsonl"))
    deg=indexed(rows(DIAG/"Desc-Degraded.jsonl"))
    swap=indexed(rows(PROBE/"views/Name-Swapped.jsonl"))
    with (OLD/"strata.csv").open(encoding="utf-8-sig") as f:
        ident={r["id"] for r in csv.DictReader(f) if r["subset"]=="unique-correct"}
    k3=[]
    aliases=indexed(rows(DIAG/"private/renamed_maps.jsonl"))
    ledger=[]
    files={}
    for sid,r in clean.items():
        pre,ts,post=split_prompt(r)
        k=keys[sid]
        names=[t["name"] for t in ts]
        gold,args=call_parts(k["ground_truth"][0])
        g=names.index(gold)
        full,h,tail,sim=permutation(ts,g,sid)
        anon=[aliases[sid]["_meta_mapping"]["func_map"][n] for n in names]
        assert anon==names_mask(ts,sid)
        descs=coarse(ts)
        groups={"high_param_overlap":max(overlap(signature(ts[g]),signature(ts[i])) for i in range(len(ts)) if i!=g)>=.5}
        q=tokens(post.split("[BEGIN OF QUERY]")[-1].split("[END OF QUERY]")[0])
        cues=[overlap(tokens(n)-STOP,q) for n in names]
        groups["name_cue_proxy"]=cues[g]>0 and cues[g]>max(c for i,c in enumerate(cues) if i!=g)
        views={"Clean":ts,"Name-Swapped":split_prompt(swap[sid])[1],
               "Coarse-Desc":[dict(t,description=d) for t,d in zip(ts,descs)]}
        if sid in ident: views["Desc-Degraded"]=split_prompt(deg[sid])[1]
        if len(ts)>=3:
            k3.append(sid)
            others=[i for i in range(len(ts)) if i!=g]
            from itertools import combinations
            d1,d2=max(combinations(others,2), key=lambda ij:sim(*ij))
            for v,(a,b) in (("Target-Swapped",(g,h)),("Distractor-Swapped",(d1,d2))):
                saved={a,b}
                extras=[i for i in range(len(ts)) if i not in saved and (v=="Target-Swapped" or i!=g)]
                if extras: saved.add(random.Random("full/42/"+sid+"/"+v).choice(extras))
                new=deepcopy(ts)
                new[a]["name"],new[b]["name"]=names[b],names[a]
                for i,t in enumerate(new):
                    t["description"]=cleaned(ts[i].get("description"),names[i]) if i in saved else descs[i]
                views[v]=new
        for v,tools in views.items():
            changed=[t["name"]!=n for t,n in zip(tools,names)]
            modes=["Keep-Name","Always-Mask"]
            if v!="Clean":modes+=["Length-"+str(t) for t in TAUS]
            if v in ("Target-Swapped","Distractor-Swapped"):modes+=["Oracle"]
            for mode in modes:
                new,inv=wrapper(tools,anon,changed,mode)
                text=pre+dump(new)+post
                case=dict(id=sid,messages=msg(text))
                stem=v if mode=="Keep-Name" else v+"__"+mode
                files.setdefault(stem,[]).append(case)
                ledger.append(dict(id=sid,view=v,mode=mode,stem=stem,source=k["source"],g=g,flags=changed,
                                   full=[t.get("description")!=d and "description" in t for t,d in zip(tools,descs)],
                                   original_gold=gold,ground_truth=[dict(name=tools[g]["name"],arguments=args)],
                                   inverse=inv,groups=groups))
    assert len(ident)==168 and len(k3)==185
    for name,data in files.items():
        (out/(name+".jsonl")).write_bytes("".join(dump(r)+"\n" for r in data).encode())
    (out.parent/"eval_keys.jsonl").write_bytes("".join(dump(r)+"\n" for r in ledger).encode())
    (out.parent/"eligible_k3_ids.json").write_text(dump(k3))


def build():
    with TemporaryDirectory(dir=DATA) as tmp:
        stage=Path(tmp)/OUT.name
        (stage/"train").mkdir(parents=True)
        (stage/"eval_views").mkdir()
        train(stage/"train")
        evaluation(stage/"eval_views")
        if OUT.exists(): raise FileExistsError(OUT)
        stage.rename(OUT)


if __name__=="__main__":
    build()
