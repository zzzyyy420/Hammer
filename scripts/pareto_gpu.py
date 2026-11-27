"""Remote caller queue and metadata-only exact conditional label scorer."""
import argparse
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path

H=Path('/root/autodl-tmp/Hammer')
P=H/'research_outputs/relgate_pareto'
BASE=H/'models/Qwen2-1.5B-Instruct'


def read(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def tokenize(stage):
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    tmpl=get_template_and_fix_tokenizer(tok,DataArguments(template='qwen'))
    for job in json.loads((P/f'jobs_{stage}.json').read_bytes()):
        data=[]
        for r in read(P/'inputs'/f'{job}.jsonl'):
            ids,_=tmpl.encode_oneturn(tok,r['messages']+[dict(role='assistant',content='')],system='',tools='')
            data.append(dict(id=r['id'],prompt_token_ids=ids))
        (P/'tokens').mkdir(exist_ok=True)
        (P/'tokens'/f'{job}.json').write_text(json.dumps(dict(rows=data,
            stops=sorted({tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')}))))
        print('tokenized',job,len(data),flush=True)


def generate(stage,job):
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    import re
    cfg=json.loads((P/f'jobs_{stage}.json').read_bytes())[job]
    data=json.loads((P/'tokens'/f'{job}.json').read_bytes())
    dest=P/'results'/f'{job}.jsonl';dest.parent.mkdir(exist_ok=True)
    done={r['id']:r for r in read(dest)} if dest.exists() else {}
    assert len(done)==(len(read(dest)) if dest.exists() else 0)
    assert set(done)<={r['id'] for r in data['rows']}
    todo=[r for r in data['rows'] if r['id'] not in done];start=time.time()
    if todo:
        llm=LLM(model=cfg['base'],dtype='bfloat16',seed=42,enforce_eager=True,
            max_model_len=max(len(r['prompt_token_ids']) for r in todo)+512,
            gpu_memory_utilization=.84,max_num_seqs=16,enable_lora=True,max_lora_rank=32)
        lora=LoRARequest(job,1,cfg['adapter'])
        params=SamplingParams(temperature=0,max_tokens=512,stop_token_ids=data['stops'])
        for i in range(0,len(todo),128):
            batch=todo[i:i+128]
            outputs=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in batch],params,lora_request=lora,use_tqdm=False)
            with dest.open('a',encoding='utf-8') as f:
                for r,o in zip(batch,outputs,strict=True):
                    raw=o.outputs[0].text.strip();rec=dict(id=r['id'],raw=raw,finish_reason=o.outputs[0].finish_reason)
                    m=re.fullmatch(chr(96)*3+r'(?:json)?\s*(.*?)\s*'+chr(96)*3,raw,re.S|re.I)
                    try:rec.update(parsed_pred=json.loads(m.group(1) if m else raw),parse_ok=True)
                    except ValueError:rec.update(parsed_pred=None,parse_ok=False)
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            print(job,min(i+128,len(todo)),'/',len(todo),flush=True)
    else:dest.touch(exist_ok=True)
    assert len(read(dest))==len(data['rows'])
    report=dict(count=len(data['rows']),generated=len(todo),elapsed_seconds=time.time()-start,sha256=sha(dest),
        adapter_sha256=sha(Path(cfg['adapter'])/'adapter_model.safetensors'),
        runtime=dict(dtype='bfloat16',temperature=0,seed=42,max_num_seqs=16,max_tokens=512))
    (P/'results'/f'{job}.done.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def label_scores(way):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
    from peft import PeftModel
    gate=Path(os.environ['RELGATE_SCORE_ADAPTER']) if os.getenv('RELGATE_SCORE_ADAPTER') else H/'research_outputs/relgate_v1'/('RelGate-3Way' if way=='3way' else 'RelGate-2Way')/'adapter'
    source=Path(os.environ['RELGATE_SCORE_INPUT']) if os.getenv('RELGATE_SCORE_INPUT') else P/'score_inputs'/f'{way}.jsonl'
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(model,gate).to('cuda').eval()
    signature=inspect.signature(model.get_base_model().forward).parameters
    keepkey='logits_to_keep' if 'logits_to_keep' in signature else 'num_logits_to_keep' if 'num_logits_to_keep' in signature else None
    labels=['SUPPORTED','UNVERIFIED','CONTRADICTED'] if way=='3way' else ['SUPPORTED','CONTRADICTED']
    no_eos=os.getenv('PHASE25_LABELS_NO_EOS')=='1'
    labelids=[tok.encode(y,add_special_tokens=False)+([] if no_eos else [tok.eos_token_id]) for y in labels]
    score_dir=Path(os.environ['RELGATE_SCORE_OUTPUT']) if os.getenv('RELGATE_SCORE_OUTPUT') else P/('phase25_scores' if no_eos else 'scores')
    start=time.time();dest=score_dir/f'{way}.jsonl';dest.parent.mkdir(exist_ok=True)
    old=read(dest) if dest.exists() else []
    done={r['id'] for r in old};assert len(done)==len(old)
    inputs=sorted(read(source),key=lambda r:hashlib.sha256(r['id'].encode()).digest())
    assert len({r['id'] for r in inputs})==len(inputs) and done<={r['id'] for r in inputs}
    checks=[];timings=[]
    for row in inputs:
        if row['id'] in done:continue
        begin=time.time()
        ids=tok.apply_chat_template([dict(role='user',content=row['instruction'])],tokenize=True,add_generation_prompt=True)
        x=torch.tensor([ids],device='cuda');L=len(ids);B=len(labels)
        with torch.inference_mode():
            values=[]
            for a in labelids:
                full=model(input_ids=torch.tensor([ids+a[:-1]],device='cuda'),use_cache=False,
                    **({keepkey:len(a)} if keepkey else {})).logits[0]
                full=full[-len(a):].float().log_softmax(-1)
                values.append([float(full[j,a[j]]) for j in range(len(a))])
            if len(checks)<3:
                checks.append(dict(id=row['id'],method='full_forward',labels=labels))
            ell=[sum(a) for a in values];finite=all(math.isfinite(v) for v in ell)
            probs=torch.tensor(ell,dtype=torch.float64).softmax(0).tolist() if finite else [None]*B
            rec=dict(id=row['id'],q=dict(zip(labels,probs)),ell=dict(zip(labels,ell)),finite=finite,
                token_logprobs=dict(zip(labels,values)),label_token_ids=dict(zip(labels,labelids)),
                prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),prompt_tokens=L)
            with dest.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
            del full
        timings.append(time.time()-begin);done.add(row['id'])
        if len(timings)==32 or len(done)%100==0:
            print(way,'scores',len(done),'/',len(inputs),'sec_per_input',sum(timings)/len(timings),flush=True)
    report=dict(count=len(done),sha256=sha(dest),elapsed_seconds=time.time()-start,
        adapter_sha256=sha(gate/'adapter_model.safetensors'),input_sha256=sha(source),
        template_sha256=sha(P/'external_gate_prompt_v1.txt'),
        tokenizer_sha256={f.name:sha(f) for f in BASE.glob('*token*.json')},
        label_ids=dict(zip(labels,labelids)),prefix_full_checks=checks,score='full label likelihood' if no_eos else 'full label+EOS likelihood',
        peak_vram_gb=torch.cuda.max_memory_allocated()/2**30)
    assert len(done)==len(inputs)
    (score_dir/f'{way}.done.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tokenize','generate','scores']);p.add_argument('arg');p.add_argument('job',nargs='?');a=p.parse_args()
    {'tokenize':lambda:tokenize(a.arg),'generate':lambda:generate(a.arg,a.job),'scores':lambda:label_scores(a.arg)}[a.action]()
