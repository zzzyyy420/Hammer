#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/hammer_setup/activate.sh
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PROJECT=/root/autodl-tmp/Hammer
CODE="$PROJECT/source/Hammer"
RUN="$PROJECT/research_outputs/hammer15b_baseline"
MODEL="$PROJECT/models/Hammer-1.5b"
mkdir -p "$RUN"/{logs,configs,metrics,predictions}
cd "$CODE"
python -m pip freeze > "$RUN/configs/requirements.lock.txt"
nvidia-smi > "$RUN/logs/nvidia-smi-before.txt"
python -m llamafactory.cli env > "$RUN/logs/llamafactory-env.txt" 2>&1
python "$CODE/train/LLaMA-Factory/src/train.py" --help > "$RUN/logs/train-cli-help.txt" 2>&1
python "$CODE/train/LLaMA-Factory/src/train.py" --stage sft --do_predict --model_name_or_path "$MODEL" --eval_dataset NexusRaven_hammer --dataset_dir "$CODE/data" --template qwen --output_dir "$RUN/predictions/smoke" --per_device_eval_batch_size 1 --predict_with_generate --max_samples 2 --cutoff_len 6000 --temperature 0.0001 --max_new_tokens 512 --bf16 > "$RUN/logs/smoke.log" 2>&1
printf '%s\n' 'SMOKE_DONE'
for DATASET in apibank_l1 apibank_l2 NexusRaven sealtool toolalpaca; do
    printf '%s START %s\n' "$(date -Is)" "$DATASET"
    bash /root/autodl-tmp/hammer_setup/eval.compat.sh "$MODEL" "$DATASET" 2>&1 | tee "$RUN/logs/eval_${DATASET}.log"
    DEST="$CODE/predict/Hammer-1.5b/${DATASET}_hammer"
    cp -a "$DEST" "$RUN/predictions/${DATASET}"
    python evaluation/evaluate.py "$DEST" > "$RUN/metrics/${DATASET}.txt"
    printf '%s DONE %s\n' "$(date -Is)" "$DATASET"
done
nvidia-smi > "$RUN/logs/nvidia-smi-after.txt"
touch "$RUN/EVALUATION_DONE"
printf '%s EVALUATION_DONE\n' "$(date -Is)"
