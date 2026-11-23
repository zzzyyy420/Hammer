#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/hammer_setup
ENV=/root/autodl-tmp/envs/hammer
export PATH="$ENV/bin:$PATH"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$ROOT/logs" "$ROOT/src"
python -m pip install 'pip<26' 'setuptools<76' wheel ninja packaging
python -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
python -m pip install 'transformers==4.49.0' 'peft==0.12.0' 'datasets==3.2.0' 'accelerate==1.2.1' 'trl==0.9.6' 'numpy==1.26.4' 'huggingface-hub<1' 'tokenizers==0.21.0'
DS_BUILD_OPS=0 python -m pip install --no-build-isolation 'deepspeed==0.16.4'
python -m pip install 'llamafactory==0.9.2' 'librosa>=0.10.2' 'matplotlib>=3.7.0' scipy einops sentencepiece tiktoken protobuf uvicorn pydantic fastapi sse-starlette fire pyyaml
python -m pip check
python -m pip freeze > "$ROOT/requirements.lock.txt"
touch "$ROOT/INSTALL_DONE"
