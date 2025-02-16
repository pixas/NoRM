CHECKPOINT_DIR=""
TASK_PATH=""

domains=("gsm8k_cot")

MODEL_BASE="$1"



TRAINING_DATA="$2"
LOGS_BASE_PATH=./logs/${TRAINING_DATA}


CKPT="$3"
MODEL_PATH=${CHEKPOINT_DIR}/${TRAINING_DATA}-${CKPT}

while [ ! -f "${MODEL_PATH}/adapter_config.json" ]; do
    echo "Waiting for ${MODEL_PATH}/adapter_config.json to appear..."
    sleep 60
done

for domain in "${domains[@]}"; do
    sbatch scripts/eval/slurm/eval_parallel_peft_batch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 2
    
done
