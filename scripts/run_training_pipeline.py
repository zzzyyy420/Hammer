from pathlib import Path
import subprocess,json,os,time,signal,shutil,math,datetime
r=Path('/root/autodl-tmp/Hammer'); run=r/'research_outputs/hammer15b_train_v1'; cfg=run/'configs'; code=r/'source/Hammer'
env=os.environ.copy();env.update({'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','FORCE_TORCHRUN':'1','NPROC_PER_NODE':'1'})
def status(stage,**kw):
 d={'stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw};(run/'progress.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
def execute(stage,args,extra_env=None):
 status(stage,command=args)
 e=dict(env);e.update(extra_env or {})
 with (run/f'logs/{stage}.log').open('w') as f:
  p=subprocess.Popen(args,cwd=code,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (run/f'logs/{stage}.pid').write_text(str(p.pid))
  while p.poll() is None:
   if shutil.disk_usage(run).free<5*1024**3:
    os.killpg(p.pid,signal.SIGTERM); raise RuntimeError('STOP_DISK_LT_5GiB; existing checkpoints preserved')
   training_log=run/'checkpoints/trainer_log.jsonl'
   if stage=='train_full' and training_log.exists():
    lines=training_log.read_text().splitlines()
    if lines:
     try:
      item=json.loads(lines[-1]); status(stage,latest_trainer_log=item,pid=p.pid)
      for k in ['loss','grad_norm']:
       if isinstance(item.get(k),(int,float)) and not math.isfinite(item[k]):
        os.killpg(p.pid,signal.SIGTERM); raise RuntimeError('STOP_NONFINITE_'+k)
     except json.JSONDecodeError: pass
   gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   with (run/'logs/gpu_usage.csv').open('a') as g:g.write(f'{time.time()},{stage},{gpu.stdout.strip()}\n')
   time.sleep(10)
  if p.returncode: raise RuntimeError(f'{stage} failed exit {p.returncode}; see logs/{stage}.log')
 status(stage+'_done')
try:
 status('waiting_base_model')
 while not (cfg/'base_model_revision.json').exists(): time.sleep(10)
 execute('train_smoke',['llamafactory-cli','train',str(cfg/'smoke.json')])
 for step in [1,2]:
  p=run/f'smoke_checkpoints/checkpoint-{step}'
  assert (p/'adapter_model.safetensors').exists(),str(p)
  assert (p/'trainer_state.json').exists(),str(p)
  assert any('optim' in x.name.lower() or 'global_step' in x.name for x in p.iterdir()),'Optimizer state missing'
 resume=json.loads((cfg/'smoke.json').read_text());resume.update({'max_steps':3,'resume_from_checkpoint':str(run/'smoke_checkpoints/checkpoint-2')})
 (cfg/'smoke_resume.json').write_text(json.dumps(resume,indent=2))
 execute('train_smoke_resume',['llamafactory-cli','train',str(cfg/'smoke_resume.json')])
 assert json.loads((run/'smoke_checkpoints/checkpoint-3/trainer_state.json').read_text())['global_step']==3
 execute('train_full',['llamafactory-cli','train',str(cfg/'train.json')])
 assert (run/'checkpoints/adapter_model.safetensors').exists(),'Final adapter missing'
 export={'model_name_or_path':str(r/'models/Qwen2-1.5B-Instruct'),'adapter_name_or_path':str(run/'checkpoints'),'template':'qwen','finetuning_type':'lora','export_dir':str(run/'merged_model'),'export_size':2,'export_device':'cpu','export_legacy_format':False}
 (cfg/'export.json').write_text(json.dumps(export,indent=2))
 execute('merge',['llamafactory-cli','export',str(cfg/'export.json')],{'FORCE_TORCHRUN':'0'})
 for name in ['apibank_l1','apibank_l2','NexusRaven','sealtool','toolalpaca']:
  execute('eval_'+name,['bash','/root/autodl-tmp/hammer_setup/eval.compat.sh',str(run/'merged_model'),name],{'FORCE_TORCHRUN':'0'})
  src=code/f'predict/merged_model/{name}_hammer';shutil.copytree(src,run/f'predictions/{name}',dirs_exist_ok=False)
  result=subprocess.check_output(['python','evaluation/evaluate.py',str(src)],cwd=code,env=env)
  (run/f'metrics/{name}.txt').write_bytes(result)
 (run/'PIPELINE_DONE').touch();status('completed',training_completed=True,evaluation_completed=True)
except Exception as e:
 status('failed',error=str(e));raise
