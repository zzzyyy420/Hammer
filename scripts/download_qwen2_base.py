from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import json
repo='Qwen/Qwen2-1.5B-Instruct'
info=HfApi().model_info(repo,revision='2ebb13f1f55edeb5bd4b108d0ad589397a6ee4ac',token=False)
print('BASE_MODEL_REVISION',info.sha,flush=True)
p=snapshot_download(repo,revision=info.sha,token=False,local_dir='/root/autodl-tmp/Hammer/models/Qwen2-1.5B-Instruct',allow_patterns=['*.json','*.safetensors','*.txt','*.model','README.md','LICENSE'])
Path('/root/autodl-tmp/Hammer/research_outputs/hammer15b_train_v1/configs/base_model_revision.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'local_dir':p},indent=2))
print('BASE_MODEL_READY',p,flush=True)
