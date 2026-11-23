import importlib.metadata as md
import json, subprocess
import torch, transformers, peft, datasets, accelerate, trl, deepspeed
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
result={'versions':{p:md.version(p) for p in ['torch','transformers','peft','datasets','accelerate','trl','deepspeed','llamafactory']},'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'bf16':torch.cuda.is_bf16_supported()}
a=torch.randn(64,64,device='cuda',dtype=torch.bfloat16)
b=a@a.T
torch.cuda.synchronize()
assert torch.isfinite(b).all()
config=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)
model=get_peft_model(Qwen2ForCausalLM(config),LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).cuda().to(torch.bfloat16)
model.eval()
with torch.no_grad():
    out=model(input_ids=torch.randint(0,128,(1,16),device='cuda'))
assert out.logits.shape==(1,16,128)
result['gpu_bf16_matmul']='PASS'
result['qwen2_peft_forward']='PASS'
from deepspeed.ops.adam import FusedAdam
opt=FusedAdam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
result['deepspeed_fused_adam_load']='PASS'
print(json.dumps(result,indent=2))
with open('/root/autodl-tmp/hammer_setup/verification.json','w') as f: json.dump(result,f,indent=2)
