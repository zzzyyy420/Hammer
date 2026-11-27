"""Short metadata SFT with exact frozen Self-Gate chat formatting and resume states."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_bytes())
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model
    set_seed(cfg['seed'])
    base=cfg['model_name_or_path'];root=Path(cfg['run_root']);arm=cfg['arm'];out=root/arm
    out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,padding_side='right')
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    records=[json.loads(l) for l in Path(cfg['train_path']).read_bytes().splitlines()]
    encoded=[];lengths=[]
    for r in records:
        messages=[dict(role='user',content=r['instruction'])]
        prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        ids=tokenizer.apply_chat_template(messages+[dict(role='assistant',content=r['output'])],tokenize=True)
        assert ids[:len(prefix)]==prefix
        assert len(ids)<=cfg['cutoff_len'],('would truncate',len(ids))
        encoded.append(dict(input_ids=ids,attention_mask=[1]*len(ids),labels=[-100]*len(prefix)+ids[len(prefix):]))
        lengths.append(len(ids))
    (out/'token_lengths.json').write_text(json.dumps(dict(n=len(lengths),max=max(lengths),mean=sum(lengths)/len(lengths),truncated=0)))
    class Data(Dataset):
        def __len__(self):return len(encoded)
        def __getitem__(self,i):return encoded[i]
    def collate(batch):
        n=((max(len(x['input_ids']) for x in batch)+7)//8)*8
        result={}
        for k,fill in (('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)):
            result[k]=torch.tensor([x[k]+[fill]*(n-len(x[k])) for x in batch],dtype=torch.long)
        return result
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=0.,
           target_modules=cfg['lora_target'],bias='none',task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    class Monitor(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            expected=int(cfg['epochs']*((len(encoded)+cfg['micro_batch']-1)//cfg['micro_batch']//cfg['gradient_accumulation_steps']))
            assert state.max_steps==expected,(state.max_steps,expected)
        def on_log(self,args,state,control,logs=None,**kw):
            if logs:
                rec=dict(step=state.global_step,elapsed_seconds=time.time()-start,**logs)
                with (out/'training_log.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                print(json.dumps(rec),flush=True)
                if any(not __import__('math').isfinite(float(logs[k])) for k in ('loss','grad_norm') if k in logs):raise RuntimeError('nonfinite training metric')
    args=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=cfg['epochs'],
         per_device_train_batch_size=cfg['micro_batch'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
         learning_rate=cfg['learning_rate'],lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=.01,
         bf16=True,tf32=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
         logging_steps=10,logging_first_step=True,save_strategy='steps',save_steps=100,save_total_limit=1,
         report_to='none',seed=42,data_seed=42,dataloader_num_workers=2,optim='adamw_torch_fused',
         remove_unused_columns=False)
    start=time.time();torch.cuda.reset_peak_memory_stats()
    class ConfiguredBatchTrainer(Trainer):
        def _inner_training_loop(self,batch_size=None,args=None,**kwargs):
            # HF 4.49 train() restores the old checkpoint micro-batch even when
            # args changed. Apply the explicitly recorded OOM fallback instead.
            return super()._inner_training_loop(batch_size=args.train_batch_size,args=args,**kwargs)
    trainer=ConfiguredBatchTrainer(model=model,args=args,train_dataset=Data(),data_collator=collate,callbacks=[Monitor()])
    trainer.train(resume_from_checkpoint=cfg.get('resume_from_checkpoint'))
    expected=int(cfg['epochs']*((len(encoded)+cfg['micro_batch']-1)//cfg['micro_batch']//cfg['gradient_accumulation_steps']))
    assert trainer.state.global_step==expected,(trainer.state.global_step,expected)
    trainer.save_model(str(out/'adapter'));tokenizer.save_pretrained(str(out/'adapter'))
    trainer.save_state()
    # Save a final resumable checkpoint, not only the adapter.
    trainer._save_checkpoint(trainer.model,trial=None)
    adapter=out/'adapter/adapter_model.safetensors'
    result=dict(arm=arm,steps=trainer.state.global_step,elapsed_seconds=time.time()-start,
           gpu=torch.cuda.get_device_name(),peak_vram_gb=torch.cuda.max_memory_allocated()/2**30,
           adapter=str(out/'adapter'),adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
           config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
           train_rows=len(encoded),effective_batch=cfg['micro_batch']*cfg['gradient_accumulation_steps'])
    result.update(micro_batch=cfg['micro_batch'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                  resumed_from=cfg.get('resume_from_checkpoint'))
    (out/'verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
