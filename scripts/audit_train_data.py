import json,hashlib,collections
from pathlib import Path
r=Path('/root/autodl-tmp/Hammer'); out=r/'research_outputs/hammer15b_train_v1/configs'
files={'xlam60k':r/'source/Hammer/data/train/xlam_function_calling_60k.json','irrelevance':r/'data/xlam-7.5k-irrelevancek.json'}
report={}
for name,p in files.items():
 raw=p.read_bytes(); data=json.loads(raw); assert isinstance(data,list)
 assert len(data)==(60000 if name=='xlam60k' else 7500),(name,len(data))
 errors=[]; signatures=[]
 for i,x in enumerate(data):
  try:
   assert all(k in x for k in ['query','tools','answers'])
   assert isinstance(json.loads(x['tools']),list)
   assert isinstance(json.loads(x['answers']),list)
  except Exception as e: errors.append({'index':i,'type':type(e).__name__})
  signatures.append(hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest())
 assert not errors,(name,errors[:5])
 report[name]={'path':str(p),'count':len(data),'sha256':hashlib.sha256(raw).hexdigest(),'schema_errors':errors,'exact_duplicate_rows':len(signatures)-len(set(signatures)),'first_row_field_types':{k:type(v).__name__ for k,v in data[0].items()}}
(out/'raw_data_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
