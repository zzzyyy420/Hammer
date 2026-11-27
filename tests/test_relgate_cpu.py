import hashlib
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_relgate_dataset import canonical,key,PLACEHOLDERS
from relgate_cpu_audit import DEST,R,GLOBAL_TAU,EXCLUDED
from routeb_validation import rows,view_rows,VIEWS


class RelGateCPU(unittest.TestCase):
    def test_manifest_and_balance(self):
        root=DEST/'data'
        for line in (root/'SHA256SUMS.txt').read_text().splitlines():
            h,p=line.split('  ',1)
            self.assertEqual(hashlib.sha256((root/p).read_bytes()).hexdigest(),h)
        train=rows(root/'train_3way.jsonl');val=rows(root/'validation_3way.jsonl')
        self.assertEqual(Counter(r['output'] for r in train),dict(SUPPORTED=4000,CONTRADICTED=4000,UNVERIFIED=4000))
        self.assertEqual(len(val),1000)
        for split in ('train','validation'):
            three=rows(root/f'{split}_3way.jsonl');two=rows(root/f'{split}_2way.jsonl')
            self.assertEqual(two,[r for r in three if r['output']!='UNVERIFIED'])

    def test_family_and_triple_disjoint(self):
        a,b=[rows(DEST/f'data/{s}_provenance.jsonl') for s in ('train','validation')]
        self.assertFalse({r['family'] for r in a}&{r['family'] for r in b})
        self.assertFalse({r['origin_triple'] for r in a}&{r['origin_triple'] for r in b})
        self.assertFalse({key(r['triple']) for r in a}&{key(r['triple']) for r in b})
        for rs in (a,b):
            self.assertEqual(len(rs),len({key(r['triple']) for r in rs}))

    def test_schema_and_cleaning(self):
        source=json.loads((DEST/'source/xlam_function_calling_60k.json').read_bytes())
        for split in ('train','validation'):
            for r in rows(DEST/f'data/{split}_provenance.jsonl'):
                original=json.loads(source[int(r['source_id'].split(':')[1])]['tools'])[r['source_slot']]
                self.assertEqual(r['triple']['parameters'],original['parameters'])
                self.assertEqual(key(original),r['origin_triple'])
                if r['label']=='CONTRADICTED':
                    self.assertNotEqual(r['triple']['name'],original['name'])
                    self.assertIsNone(re.search(r'(?<!\w)'+re.escape(original['name'])+r'(?!\w)',r['triple']['description'] or '',re.I))
                elif r['label']=='SUPPORTED':
                    self.assertEqual(canonical(r['triple']),canonical(original))

    def test_template_and_metadata_overlap(self):
        # Independent audit after construction. Builder itself opens no eval files.
        descs={t.get('description') for row in view_rows('Coarse-Desc') for t in __import__('routeb_stage1').split_prompt(row)[1]}
        evalkeys={key(r) for r in rows(R/'inputs/self_gate.jsonl')}
        found=[];training=[]
        for split in ('train','validation'):
            for r in rows(DEST/f'data/{split}_provenance.jsonl'):
                self.assertNotIn(r['triple_sha256'],evalkeys)
                if r['label']=='UNVERIFIED':self.assertNotIn(r['triple']['description'],descs)
        self.assertNotIn('Manage', ' '.join(PLACEHOLDERS))

    def test_exact_denominators_and_gate_audit(self):
        self.assertEqual([len(view_rows(v)) for v in VIEWS],[197,197,197,197,165,182,182])
        audit=rows(DEST/'audit/self_gate_10_complete.jsonl')
        self.assertEqual(len(audit),10)
        self.assertTrue(all(not r['parse_failure'] and not r['oracle_fields_in_gate_input'] for r in audit))
        self.assertEqual(GLOBAL_TAU,64)
        self.assertEqual(len(EXCLUDED),3)


if __name__=='__main__':unittest.main()
