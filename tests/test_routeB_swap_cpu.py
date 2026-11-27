import json
import re
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_routeB_swap_probe import DATA, COARSE, design, split_prompt
from paired_did_eval import rows, indexed, call_parts
from score_routeB_probe import outcome


class RouteB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = DATA/"phase1_diag_views_v1"
        keys = indexed(rows(DATA/"phase1_gpu_v1/key.jsonl"))
        maps = indexed(rows(base/"private/renamed_maps.jsonl"))
        cls.cases = []
        for row in rows(base/"Clean.jsonl"):
            i = row["id"]
            pre, tools, post = split_prompt(row)
            for view,(inp,k) in design(row, keys[i], maps[i]["_meta_mapping"]["func_map"]).items():
                p, new, s = split_prompt(inp)
                assert (pre,post) == (p,s)
                cls.cases.append((tools,new,k,view,keys[i]))

    def test_permutations(self):
        for old,new,k,v,_ in self.cases:
            names = [t["name"] for t in new]
            original = [t["name"] for t in old]
            if v == "Name-Masked":
                self.assertTrue(set(original).isdisjoint(names))
                continue
            self.assertCountEqual(names, original)
            changed = {i for i,(x,y) in enumerate(zip(names,original)) if x != y}
            self.assertEqual(changed,set(range(len(old))) if v=="Name-Swapped" else {k["g"],k["h"]})

    def test_gold_binding(self):
        for old,new,k,v,key in self.cases:
            g = k["g"]
            name,args = call_parts(key["ground_truth"][0])
            self.assertEqual(old[g]["name"], name)
            self.assertEqual(k["ground_truth"], [dict(name=new[g]["name"], arguments=args)])
            for i,t in enumerate(old):
                expected = t["description"]
                if v == "Partial-Swap" and i not in (g,k["h"]):
                    expected = COARSE
                elif expected:
                    expected = re.sub(r"(?<!\w)"+re.escape(t["name"])+r"(?!\w)", "this function",expected,flags=re.I)
                self.assertEqual(new[i]["description"],expected)

    def test_no_self_reference(self):
        for old,new,_,_,_ in self.cases:
            for x,y in zip(old,new):
                self.assertIsNone(re.search(r"(?<!\w)"+re.escape(x["name"])+r"(?!\w)", y["description"] or "",re.I))

    def test_schema(self):
        for old,new,_,_,_ in self.cases:
            self.assertEqual([x["parameters"] for x in old],[x["parameters"] for x in new])

    def test_hijack(self):
        for _,_,k,v,_ in self.cases:
            if v == "Name-Masked":
                continue
            def pred(calls):
                return dict(parse_ok=True, parsed_pred=calls)
            self.assertEqual(outcome(pred(k["ground_truth"]),k), (1,1,0,0))
            self.assertEqual(outcome(pred([dict(name=k["original_gold"])]),k), (0,0,1,0))
            self.assertEqual(outcome(pred([]),k), (0,0,0,0))
