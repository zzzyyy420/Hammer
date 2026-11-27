"""Deduplicate on exact prompt bytes per model, score recovered tool names on paired IDs."""
import argparse
import hashlib
import math
from collections import defaultdict
from routeb_stage1 import *

MODELS=("Pure-NameMask","Pure-NoMask","Hammer-1.5b")
from score_routeB_probe import outcome


def fingerprint(r):
    return hashlib.sha256(dump(r["messages"]).encode()).hexdigest()


def plan():
    inputs={p.stem:indexed(rows(p)) for p in (OUT/"eval_views").glob("*.jsonl")}
    byhash={}
    for stem,records in inputs.items():
        for sid,r in records.items():
            byhash.setdefault(fingerprint(r),dict(id=fingerprint(r),messages=r["messages"]))
    for model in MODELS:
        cache={}
        for view,dir_,pr in [
            *((v,DIAG,OLD/"predictions"/model/"full") for v in ("Clean","Desc-Degraded")),
            *((v,PROBE/"views",PROBE/"predictions"/model) for v in ("Name-Swapped","Name-Masked","Partial-Swap"))]:
            preds=indexed(rows(pr/(view+".jsonl")))
            orig=indexed(rows(dir_/(view+".jsonl")))
            assert orig.keys()==preds.keys()
            for i,r in orig.items():
                h=fingerprint(r)
                if h in byhash and h not in cache:
                    cache[h]=dict(preds[i],id=h,reused=view,source_id=i)
        missing=[r for h,r in byhash.items() if h not in cache]
        folder=OUT/"schedule"/model
        folder.mkdir(parents=True,exist_ok=True)
        for stem,seq in (("cached",cache.values()),("requests",missing)):
            (folder/(stem+".jsonl")).write_bytes("".join(dump(r)+"\n" for r in seq).encode())
        print(model,len(cache),"reused;",len(missing),"new prompts")


def contrasts(base,other):
    # base minus other, measured on exactly the same requests.
    assert base.keys()==other.keys() and base
    wins=sum(base[i]>other[i] for i in base)
    losses=sum(base[i]<other[i] for i in base)
    n=wins+losses
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n)
    return dict(delta_pp=100*(wins-losses)/len(base),n=len(base),wins=wins,losses=losses,paired_exact_p=p)


def rates(n,correct,hijacks,defined):
    return (hijacks/n if defined else None,
            hijacks/(n-correct) if defined and n>correct else None)


def save_table(path,records):
    if records:
        with path.open("w",encoding="utf8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=records[0])
            w.writeheader()
            w.writerows(records)


def score_all(partial=False):
    dest=OUT/("cached_report" if partial else "report")
    dest.mkdir(exist_ok=True)
    inputs={p.stem:indexed(rows(p)) for p in (OUT/"eval_views").glob("*.jsonl")}
    keys=rows(OUT/"eval_keys.jsonl")
    metrics=[]; effects=[]; bounds=[]
    for model in MODELS:
        folder=OUT/"schedule"/model
        preds=indexed(rows(folder/"cached.jsonl"))
        if not partial:
            fresh=indexed(rows(OUT/"predictions"/(model+".jsonl")))
            assert fresh.keys()==indexed(rows(folder/"requests.jsonl")).keys() and not fresh.keys()&preds.keys()
            preds.update(fresh)
        cells=defaultdict(dict)
        for k in keys:
            sid=k["id"]; h=fingerprint(inputs[k["stem"]][sid])
            rec=preds.get(h)
            result=None
            if rec is not None:
                recovered=dict(rec,parsed_pred=remap(rec["parsed_pred"],k["inverse"]))
                result=outcome(recovered,k)
            for group in ["all",k["source"]]+[g for g,b in k["groups"].items() if b]:
                cells[k["view"],k["mode"],group][sid]=result
        acc={}
        if partial:
            for (view,mode,g),rr in cells.items():
                if mode!="Keep-Name" or view not in ("Clean","Coarse-Desc","Desc-Degraded"):
                    continue
                other=cells[view,"Always-Mask",g]
                observed=[r[0]-other[i][0] for i,r in rr.items() if r is not None and other[i] is not None]
                missing=len(rr)-len(observed)
                if 0<missing<=2 and all(r is not None for r in rr.values()):
                    lower=sum(observed)+sum(r[0]-1 for i,r in rr.items() if other[i] is None)
                    bounds.append(dict(model=model,view=view,group=g,n=len(rr),missing=missing,
                                       min_pp=100*lower/len(rr),max_pp=100*(lower+missing)/len(rr)))
        for (view,mode,group),rr in cells.items():
            if not all(r is not None for r in rr.values()): continue
            tool,strict,hij,invalid=map(sum,zip(*rr.values()))
            defined=view in ("Name-Swapped","Target-Swapped")
            n=len(rr)
            ar,er=rates(n,tool,hij,defined)
            metrics.append(dict(model=model,view=view,mode=mode,group=group,n=n,correct=tool,
                                tool_acc=tool/n,strict_acc=strict/n,invalid=invalid,hijacks=hij if defined else None,
                                nhr_abs=ar,nhr_err=er))
            acc[view,mode,group]={i:r[0] for i,r in rr.items()}
        for view,mode,g in acc:
            if mode!="Keep-Name": continue
            pairs=[("delta_negative","Always-Mask","Keep-Name")] if view=="Name-Swapped" else [
                ("delta_positive","Keep-Name","Always-Mask")] if view in ("Clean","Coarse-Desc","Desc-Degraded") else []
            if view in ("Target-Swapped","Distractor-Swapped"):
                pairs.extend([("oracle_gain_keep","Oracle","Keep-Name"),("oracle_gain_always","Oracle","Always-Mask")])
            for label,a,b in pairs:
                if (view,a,g) in acc and (view,b,g) in acc:
                    effects.append(dict(model=model,view=view,group=g,contrast=label,**contrasts(acc[view,a,g],acc[view,b,g])))
    save_table(dest/"metrics.csv",metrics)
    save_table(dest/"incomplete_positive_bounds.csv",bounds)
    save_table(dest/"contrasts.csv",effects)
    best=[]
    for model in MODELS:
        for view in {r["view"] for r in metrics}:
            options=[r for r in metrics if r["model"]==model and r["view"]==view
                     and r["group"]=="all" and r["mode"].startswith("Length-")]
            if len(options)==len(TAUS):
                chosen=max(options,key=lambda r:(r["correct"],-int(r["mode"].split("-")[1])))
                best.extend(r for r in metrics if r["model"]==model and r["view"]==view
                            and r["mode"]==chosen["mode"])
    save_table(dest/"best_length_diagnostic.csv",best)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["plan","score","cached"])
    mode=parser.parse_args().mode
    plan() if mode=="plan" else score_all(mode=="cached")
