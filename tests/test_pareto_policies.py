import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pareto_policies import apply_name_mask, dose_matched_mask, random_mask


class ParetoPoliciesTest(unittest.TestCase):
    def test_endpoints_nested_and_shared_view_randomness(self):
        for seed in (42,123,2026):
            masks=[random_mask('request-a',12,r,seed) for r in (0,.25,.5,.75,1)]
            self.assertEqual(masks[0], [False]*12)
            self.assertEqual(masks[-1], [True]*12)
            for left,right in zip(masks,masks[1:]):
                self.assertTrue(all(not a or b for a,b in zip(left,right)))
            self.assertEqual(masks[2],random_mask('request-a',12,.5,seed))

    def test_exact_per_request_dose_and_not_global_only(self):
        for request,n in (('a',2),('b',7),('c',12)):
            for count in range(n+1):
                mask=dose_matched_mask(request,n,count,42)
                self.assertEqual(sum(mask),count)
                self.assertEqual(mask,dose_matched_mask(request,n,count,42))

    def test_only_names_change_and_inverse_is_lossless(self):
        tools=[dict(name='read_item',description='Read item.',parameters={'id':{'type':'str','description':'item id'}}),
               dict(name='delete_item',description='Delete item.',parameters={'id':{'type':'str'}})]
        changed,inv=apply_name_mask(tools,['aB3_cD7','Zy7.29'],[True,False])
        self.assertEqual(tools[0]['name'],'read_item')
        for original,new in zip(tools,changed):
            self.assertEqual({k:v for k,v in original.items() if k!='name'},
                             {k:v for k,v in new.items() if k!='name'})
        self.assertEqual([dict(t,name=inv.get(t['name'],t['name'])) for t in changed],tools)

    def test_invalid_or_colliding_policy_rejected(self):
        for ratio in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):random_mask('a',2,ratio,42)
        with self.assertRaises(ValueError):dose_matched_mask('a',2,3,42)
        with self.assertRaises(ValueError):
            apply_name_mask([{'name':'a'},{'name':'b'}],['b','c'],[True,False])


if __name__=='__main__':unittest.main()
