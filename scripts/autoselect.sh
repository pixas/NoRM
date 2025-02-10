#!/bin/bash
#SBATCH -J autoselect
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=4G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1


BASE_PATH=/mnt/petrelfs/jiangshuyang.p/oss
# TASK_PATH=s3://syj_test/datasets/diverse_domain/train_wo_triviaqa
TASK_PATH=/mnt/hwfile/medai/jiangshuyang.p/datasets/
# TRAINING_DATA=gsm8k
# TRAINING_DATA=metamathqa
TRAINING_DATA=tulu_v2_0_3
# TRAINING_DATA=magicoder
# TRAINING_DATA=mmed_en_zh_cmexam_80k

MODEL_BASE=/mnt/petrelfs/jiangshuyang.p/models/Meta-Llama-3-8B-Instruct
domains=("math_cot" "bbh_cot" "logiqa_en_cot" "commonsense_qa_cot" "gsm8k_cot" "mmedbench_en_cot" "mmlu_cot")
# domains=("CBLUE" "cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
# domains=("humaneval_plus" "mbpp_plus")
# domains=("wikitext103_test")
# domains=("gsm8k_cot" "math_cot")
domains=("mmlu_cot" "bbh_cot" "tydiqa_cot")
# domains=("bbh_prompt_cot")

# CKPT=llama3-8b-lora-r32a128
# CKPT=llama3-8b-lora-r32a64-strangecqa-1epoch
# CKPT=llama3-8b-lora-r16a32
# CKPT=llama3-8b-lora-r32a64-gsm8k-10epoch
# CKPT=llama3-8b-lora-r64a128-med-1epoch
# CKPT=llama3-8b-lora-r64a128-metamathqa-1epoch
# CKPT=llama3-8b-lora-r16a32-metamathqa-1epoch
# CKPT=llama3-8b-lora-r8a16-metamathqa-1epoch
# CKPT=llama3-8b-lora-r16a32-tuluv2_03-1epoch
# CKPT=llama3-8b-lora-r64a128-tuluv2_03-1epoch
CKPT=llama3-8b-lora-r64a128-1epoch
# CKPT=qwen2-7b-lora-r64a128-tuluv2_03-1epoch
# CKPT=qwen2-7b-lora-r64a128-magicoder-1epoch
# CKPT=llama3-8b-lora-r64a128-magiccoder-1epoch
# CKPT=llama3-8b-lora-r16a32-magiccoder-1epoch
# CKPT=llama3-8b-lora-r8a16-magiccoder-1epoch
# CKPT=llama3-8b-lora-r32a64
# CKPT=llama3-8b-lora-r32a64-metamathqa-1epoch
# CKPT=llama3-8b-lora-r8a16-tuluv2_03-1epoch
MODEL_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
LORA_NAME_OR_PATH=None
# thr=0.1
step=0.1
range_start=1
# SAVE_NAME=minorthcos-maxorthcos-${thr}
select_method=lora
# select_method=minlora
# select_method=minorlora
# select_method=l2lora
# select_method=maxcoslora
# select_method=pcalora
SAVE_NAME=autosvd-${select_method}-${step}-${range_start}
# SAVE_NAME=norm5-orthcos-${thr}
# MODEL_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}-automerge-${SAVE_NAME}
LOG_FILE=${LOGS_BASE_PATH}/${CKPT}-automerge-${SAVE_NAME}/autoselect.log


srun -o ${LOG_FILE} python evaluation/auto_select.py --model_base $MODEL_BASE --model_path $MODEL_PATH --save_name ${SAVE_NAME} --step ${step} --select_method ${select_method} --range_start ${range_start}

for domain in "${domains[@]}"; do
    sbatch scripts/eval/slurm/eval_parallel_peft_batch_autoselect.sh $TASK_PATH $MODEL_BASE $MODEL_PATH ${CKPT}-automerge-${SAVE_NAME} ${LOGS_BASE_PATH} $domain $SAVE_NAME
# bash scripts/eval/bash/eval_models_per_dataset.sh $MODEL_BASE $TRAINING_DATA $CKPT & sleep 1
done
