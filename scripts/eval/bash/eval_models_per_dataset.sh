BASE_PATH=/mnt/petrelfs/jiangshuyang.p/oss
TASK_PATH=/mnt/hwfile/medai/jiangshuyang.p/datasets

domains=("gsm8k_cot")

MODEL_BASE="$1"
# MODEL_BASE=/mnt/hwfile/medai/jiangshuyang.p/checkpoints/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_mergelora

# TRAINING_DATA=ming-moe-clinical-v2
TRAINING_DATA="$2"
LOGS_BASE_PATH=./logs/${TRAINING_DATA}

# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_2_fix
# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_4_fix
CKPT="$3"
MODEL_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}

while [ ! -f "${MODEL_PATH}/adapter_config.json" ]; do
    echo "Waiting for ${MODEL_PATH}/adapter_config.json to appear..."
    sleep 60
done

for domain in "${domains[@]}"; do
    sbatch scripts/eval/slurm/eval_parallel_peft_batch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 2
    
done
