#!/bin/bash
#SBATCH -J autoselect
#SBATCH --partition=partition_name
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=4G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1


CHECKPOINT_DIR=""

TASK_PATH=""

TRAINING_DATA=metamathqa


MODEL_BASE=""


CKPT=llama3-8b-exp

MODEL_PATH=${CHECKPOINT_DIR}/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
LORA_NAME_OR_PATH=None

step=0.1
range_start=1
select_method=lora

SAVE_NAME=autosvd-${select_method}-${step}-${range_start}

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}-automerge-${SAVE_NAME}
LOG_FILE=${LOGS_BASE_PATH}/${CKPT}-automerge-${SAVE_NAME}/autoselect.log


srun -o ${LOG_FILE} python evaluation/auto_select.py --model_base $MODEL_BASE --model_path $MODEL_PATH --save_name ${SAVE_NAME} --step ${step} --select_method ${select_method} --range_start ${range_start}

