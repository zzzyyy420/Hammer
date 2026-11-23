from pathlib import Path
import time,json,os,subprocess,datetime,hashlib
run=Path('/root/autodl-tmp/Hammer/research_outputs/hammer15b_train_v1')
failed_since=None
while True:
    try: state=json.loads((run/'progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError): time.sleep(15);continue
    reason=None
    if (run/'PIPELINE_DONE').exists():
        required=[run/'checkpoints/adapter_model.safetensors',run/'checkpoints/trainer_state.json']+[run/f'metrics/{n}.txt' for n in ['apibank_l1','apibank_l2','NexusRaven','sealtool','toolalpaca']]
        if all(p.exists() and p.stat().st_size>0 for p in required) and list((run/'merged_model').glob('*.safetensors')):
            reason='Training, final adapter, merged weights and five-set evaluation completed'
    elif (run/'UNRECOVERABLE.json').exists():
        reason='Confirmed unrecoverable failure; see UNRECOVERABLE.json'
    elif state.get('stage')=='failed':
        if failed_since is None: failed_since=time.time()
        # Generic errors require diagnosis; elapsed time alone does not prove irrecoverability.
        # The agent creates UNRECOVERABLE.json only after diagnosis.
    else: failed_since=None
    if reason:
        records=[]
        for p in sorted(run.rglob('*')):
            if p.is_file() and 'tokenized_cache' not in str(p) and p.suffix in ['.json','.jsonl','.txt','.log','.safetensors','.pt','.bin','.csv']:
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
                records.append({'path':str(p.relative_to(run)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
        (run/'shutdown_artifact_manifest.json').write_text(json.dumps(records,indent=2))
        intent={'reason':reason,'state':state,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':'/usr/bin/shutdown','user_authorized':True,'deletes_instance':False}
        (run/'shutdown_intent.json').write_text(json.dumps(intent,indent=2));os.sync()
        print(json.dumps(intent),flush=True)
        with (run/'logs/shutdown.log').open('a') as f:
            result=subprocess.run(['/usr/bin/shutdown'],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode: print('SHUTDOWN_COMMAND_FAILED',result.returncode,flush=True)
        break
    time.sleep(15)
