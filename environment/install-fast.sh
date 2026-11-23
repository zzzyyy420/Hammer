#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/hammer_setup/activate.sh
export UV_HTTP_TIMEOUT=600
export UV_CACHE_DIR=/root/autodl-tmp/hammer_setup/uv-cache
TORCH_PACKAGE='torch==2.5.1'
LOCAL_WHEEL=/root/autodl-tmp/hammer_setup/wheels/torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl
if [ -f "$LOCAL_WHEEL" ]; then TORCH_PACKAGE="$LOCAL_WHEEL"; fi
uv pip install --python "$CONDA_PREFIX/bin/python" --index-url https://mirrors.aliyun.com/pypi/simple "$TORCH_PACKAGE" 'torchvision==0.20.1' 'transformers==4.49.0' 'peft==0.12.0' 'datasets==3.2.0' 'accelerate==1.2.1' 'trl==0.9.6' 'numpy==1.26.4' 'huggingface-hub<1' 'tokenizers==0.21.0' 'llamafactory==0.9.2' 'librosa>=0.10.2' ninja wheel 'setuptools<76'
DS_BUILD_OPS=0 uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation --index-url https://mirrors.aliyun.com/pypi/simple 'deepspeed==0.16.4'
python -m pip check
python -m pip freeze > /root/autodl-tmp/hammer_setup/requirements.lock.txt
touch /root/autodl-tmp/hammer_setup/INSTALL_DONE
