source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/hammer
export CUDA_HOME=/root/autodl-tmp/envs/hammer
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/hammer_setup/torch_extensions
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
CUDA_INCLUDE_PATHS=$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -maxdepth 2 -type d -name include | paste -sd: -)
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CUDA_INCLUDE_PATHS${CPATH:+:$CPATH}"
