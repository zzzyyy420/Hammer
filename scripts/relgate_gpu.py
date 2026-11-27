"""Separate tokenizer preparation and vLLM generation, with exact-prompt reuse."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/Hammer')
RUN=ROOT/'research_outputs/relgate_v1'
BASE=ROOT/'models/Qwen2-1.5B-Instruct'


def rows(p):return [json.loads(x) for x in p.read_bytes().splitlines()]


def tokenize(job):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    src=rows(RUN/'inputs'/f'{job}.jsonl')
    isgate=job.startswith('gate_')
    template=(RUN/'data/external_gate_prompt_v1.txt').read_text()
    assert hashlib.sha256(template.encode()).hexdigest()=='8b6238ca15111953a995cb6835b8073166aee216d0ebbf2c453d6afc819acdfa'
    if not isgate:
        from llamafactory.hparams import DataArguments
        from llamafactory.data.template import get_template_and_fix_tokenizer
        tmpl=get_template_and_fix_tokenizer(tok,DataArguments(template='qwen'))
    output=[]
    for r in src:
        if isgate:
            if 'instruction' in r:
                text=r['instruction']
            else:
                assert set(r)=={'id','name','description','parameters'}
                text=template.format(name=r['name'],description=r['description'] or '',
                       parameters=json.dumps(r['parameters'],ensure_ascii=False,sort_keys=True))
            ids=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)
        else:
            ids,_=tmpl.encode_oneturn(tok,r['messages']+[dict(role='assistant',content='')],system='',tools='')
        output.append(dict(id=r['id'],prompt_token_ids=ids))
    stops=sorted({tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')})
    (RUN/'tokens').mkdir(exist_ok=True)
    (RUN/'tokens'/f'{job}.json').write_text(json.dumps(dict(rows=output,stops=stops)))


def generate(job):
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    data=json.loads((RUN/'tokens'/f'{job}.json').read_bytes())
    cfg=json.loads((RUN/'jobs.json').read_bytes())[job]
    (RUN/'results').mkdir(exist_ok=True)
    dest=RUN/'results'/f'{job}.jsonl'
    done={r['id']:r for r in rows(dest)} if dest.exists() else {}
    assert done.keys()<={r['id'] for r in data['rows']}
    todo=[r for r in data['rows'] if r['id'] not in done]
    start=time.time()
    if todo:
        maxout=16 if job.startswith('gate_') else 512
        adapter=cfg.get('adapter')
        llm=LLM(model=cfg.get('base',str(BASE)),dtype='bfloat16',seed=42,enforce_eager=True,
            max_model_len=max(len(r['prompt_token_ids']) for r in todo)+maxout,
            gpu_memory_utilization=.84,max_num_seqs=16,enable_lora=bool(adapter),
            **({'max_lora_rank':32} if adapter else {}))
        lora=LoRARequest(job,1,adapter) if adapter else None
        params=SamplingParams(temperature=0,max_tokens=maxout,stop_token_ids=data['stops'])
        for at in range(0,len(todo),128):
            batch=todo[at:at+128]
            outputs=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in batch],params,lora_request=lora,use_tqdm=False)
            with dest.open('a',encoding='utf-8') as f:
                for r,o in zip(batch,outputs,strict=True):
                    raw=o.outputs[0].text.strip();rec=dict(id=r['id'],raw=raw,finish_reason=o.outputs[0].finish_reason)
                    if job.startswith('gate_'):
                        rec.update(label=raw if raw in {'SUPPORTED','UNVERIFIED','CONTRADICTED'} else None,parse_ok=raw in {'SUPPORTED','UNVERIFIED','CONTRADICTED'})
                    else:
                        m=re.fullmatch(chr(96)*3+r'(?:json)?\s*(.*?)\s*'+chr(96)*3,raw,re.S|re.I)
                        try:rec.update(parsed_pred=json.loads(m.group(1) if m else raw),parse_ok=True)
                        except ValueError:rec.update(parsed_pred=None,parse_ok=False)
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            print(job,min(at+128,len(todo)),'/',len(todo),flush=True)
    else:dest.touch(exist_ok=True)
    assert len(rows(dest))==len(data['rows'])
    result=dict(job=job,count=len(data['rows']),generated=len(todo),elapsed_seconds=time.time()-start,
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (RUN/'results'/f'{job}.done.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tokenize','generate']);p.add_argument('job');a=p.parse_args()
    (tokenize if a.action=='tokenize' else generate)(a.job)
