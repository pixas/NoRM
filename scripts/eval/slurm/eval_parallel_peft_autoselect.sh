#!/bin/bash
#SBATCH -J eval
#SBATCH --partition=partition_name
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
CKPT="$4" 
LOGS_BASE_PATH="$5"
DATASET="$6"
NEW_LORA_NAME="$7"
DATA_PATH=${TASK_PATH}/taia_test


name=$NEW_LORA_NAME
echo $CKPT
echo $name

if [[ $CKPT == *"llama2-7b"* ]]; then
    conv_mode="llama2"
elif [[ $CKPT == *"llama3-8b"* ]]; then
    conv_mode="llama3"
elif [[ $CKPT == *"mistral"* ]]; then 
    conv_mode="mistral"
else
    conv_mode="qwen"
fi

if [[ $DATASET == "tydiqa_cot" ]]; then
    bs=2
else
    bs=16
fi


echo $conv_mode

NAME="${name}"
mkdir -p ${LOGS_BASE_PATH}/${CKPT}/${DATASET}

echo "Processing ${DATASET}"
srun -p medai_llm --quotatype=auto --gres=gpu:1 --output=${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.log  python -m ming.eval.model_diverse_gen_batch \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --temperature 0 \
    --conv-mode $conv_mode \
    --use-logit-bias \
    --batch-size $bs \
    --resume \
    --new_lora_name ${NEW_LORA_NAME}


if [[ $DATASET == *"plus" ]]; then
    echo "Sanitizing ${DATASET}"
    srun -p medai_llm evalplus.sanitize ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl
    if [[ $DATASET == "humaneval"* ]]; then
        srun -p medai_llm --output=${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log evalplus.evaluate --dataset humaneval --samples ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer-sanitized.jsonl
    else 
        srun -p medai_llm --output=${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log evalplus.evaluate --dataset mbpp --samples ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer-sanitized.jsonl
    fi

else

    echo "Evaluating ${DATASET}"
    srun -p medai_llm --quotatype=auto --output=${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log python -m evaluation.eval_em \
        --input_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
        --output_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/wrong.jsonl
fi