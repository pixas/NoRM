#!/bin/bash
#SBATCH -J eval_qwen
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=4G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1


TASK_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"
CKPT="$4" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$5"
DATASET="$6"
NEW_LORA_NAME="$7"
DATA_PATH=${TASK_PATH}/taia_test

# echo ${CKPT}
# echo ${LOGS_BASE_PATH}
# echo ${DATASET}


if [[ $CKPT == *"llama2-7b"* ]]; then
    conv_mode="llama2"
elif [[ $CKPT == *"llama3-8b"* ]]; then
    conv_mode="llama3"
else
    conv_mode="qwen"
fi

if [[ $DATASET == "tydiqa_cot" ]]; then
    bs=2
else
    bs=16
fi

bash ~/add_oss.sh

# 输出结果

mkdir -p ${LOGS_BASE_PATH}/${CKPT}/${DATASET}

echo "Processing ${DATASET}"
srun -p medai_llm --quotatype=auto --gres=gpu:1 --output=${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.log  python -m ming.eval.model_diverse_loss_batch \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --temperature 0 \
    --conv-mode $conv_mode \
    --use-logit-bias \
    --batch-size $bs \
    --resume \
    --new_lora_name ${NEW_LORA_NAME}
