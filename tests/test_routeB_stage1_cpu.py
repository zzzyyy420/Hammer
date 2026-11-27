import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from routeb_stage1 import *
from routeb_step2a import contrasts,rates


class FrozenCPU(unittest.TestCase):
    def test_all_training_rows(self):
        assign=rows(OUT/"train/assignment.jsonl")
        self.assertEqual((len(assign),len({r["id"] for r in assign})),(29977,29977))
        source=Counter(r["source"] for r in assign if r["eligible"])
        self.assertEqual(source,{"xlam60k":10697})
        with ExitStack() as stack:
            streams=[stack.enter_context(p.open("rb")) for p in [AUDIT/"Pure-NoMask.jsonl"]+[OUT/"train"/(s+".jsonl") for s in FILES]]
            for meta,(raw,*others) in zip(assign,zip(*streams,strict=True),strict=True):
                r=json.loads(raw)
                pre,ts,post=unpack(r["instruction"]); ans=calls(r)
                self.assertEqual(eligibility(ts,ans),meta["eligible"])
                for kind,variant in zip(meta["arms"],others):
                    if kind=="clean":
                        self.assertEqual(variant,raw);continue
                    self.assertTrue(meta["eligible"])
                    x=json.loads(variant)
                    a,vs,b=unpack(x["instruction"])
                    self.assertEqual((pre,post),(a,b))
                    self.assertEqual([t["parameters"] for t in ts],[t["parameters"] for t in vs])
                    m={t["name"]:v["name"] for t,v in zip(ts,vs,strict=True)}
                    self.assertEqual(calls(x),remap(ans,m))
                    if kind=="swap":
                        self.assertEqual(set(m),set(m.values()))
                        self.assertTrue(all(n!=v for n,v in m.items()))
                        for t,v in zip(ts,vs):
                            self.assertEqual(v.get("description"),cleaned(t.get("description"),t["name"]))
        counts=[Counter(r["arms"][i] for r in assign if r["eligible"]) for i in range(3)]
        self.assertEqual([counts[0]["swap"],counts[1]["swap"]],[5349,10697])
        self.assertEqual(sorted(counts[2].values()),[2674,2674,2674,2675])

    def test_derangements(self):
        for k in [2,3,4,5,12]:
            ts=[dict(name=f"t{i}",parameters={}) for i in range(k)]
            for g in range(k):
                perm,h,_,_=permutation(ts,g,"fixture")
                self.assertEqual(sorted(perm),list(range(k)))
                self.assertTrue(all(i!=j for i,j in enumerate(perm)))
                self.assertEqual(perm[g],h)
                self.assertEqual(remap([{"name":ts[g]["name"],"arguments":{}}],dict(zip([t["name"] for t in ts],[ts[i]["name"] for i in perm])))[0]["name"],ts[h]["name"])

    def test_mixed(self):
        ledger=rows(OUT/"eval_keys.jsonl")
        ids=set(json.loads((OUT/"eligible_k3_ids.json").read_text()))
        self.assertEqual(len(ids),185)
        for view in ("Target-Swapped","Distractor-Swapped"):
            kk=[r for r in ledger if r["view"]==view and r["mode"]=="Keep-Name"]
            self.assertEqual({r["id"] for r in kk},ids)
            for r in kk:
                self.assertEqual(sum(r["flags"]),2)
                self.assertTrue(all(f for b,f in zip(r["flags"],r["full"]) if b))
                if view=="Target-Swapped" or len(r["flags"])>=4:
                    self.assertGreater(sum(r["full"]),2)
                if view=="Distractor-Swapped":
                    self.assertFalse(r["full"][r["g"]] or r["flags"][r["g"]])

    def test_wrapper_roundtrip(self):
        for r in rows(OUT/"eval_keys.jsonl"):
            if r["mode"] not in ("Always-Mask","Oracle"):continue
            inv=r["inverse"]; mapping={v:k for k,v in inv.items()}
            self.assertEqual(len(inv),len(mapping))
            cs=[dict(name=n,arguments={"literal":n}) for n in mapping]
            self.assertEqual(remap(remap(cs,mapping),inv),cs)
        inputs=indexed(rows(OUT/"eval_views/Name-Swapped__Always-Mask.jsonl"))
        self.assertEqual(inputs,indexed(rows(PROBE/"views/Name-Masked.jsonl")))

    def test_metrics(self):
        a=dict(a=1,b=1,c=0,d=1); b=dict(a=0,b=0,c=1,d=1)
        self.assertEqual(contrasts(a,b)["delta_pp"],25)
        self.assertEqual(contrasts(b,a)["delta_pp"],-25)
        self.assertEqual(rates(200,96,85,True),(85/200,85/104))
        self.assertEqual(rates(185,100,0,False),(None,None))
