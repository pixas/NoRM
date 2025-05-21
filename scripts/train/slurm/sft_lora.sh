#!/bin/bash

#SBATCH -J sft
#SBATCH --partition=partion_name
#SBATCH -N1
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:2 
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=4G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=2
NNODES=$SLURM_NNODES

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

export LOGLEVEL=INFO
# export NCCL_SOCKET_IFNAME="eth0"
MASTER_PORT=$((RANDOM % 1001 + 20000))
export NCCL_DEBUG=ERROR


TASK_PATH="$1"
TRAINING_DATA="$2"
MODEL_BASE="$3"
SAVE_PATH="$4"
CKPT="$5"
LORA_NAME_OR_PATH="$6"
other_params="$7"
DATA_PATH=${TASK_PATH}/${TRAINING_DATA}.json

echo $LORA_NAME_OR_PATH

if [[ $CKPT == *"llama2-7b"* ]]; then
    prompt_type="llama2"
elif [[ $CKPT == *"llama3-8b"* ]]; then
    prompt_type="llama3"
elif [[ $CKPT == *"mistral-7b"* ]]; then 
    prompt_type="mistral"
else
    prompt_type="qwen"
fi
echo $prompt_type




argv=()
read -ra argv <<< "$other_params"
# echo ${argv[@]}

if [[ "$other_params" != *"--lora_r"* ]]; then 
    argv+=("--lora_r" "16" "--lora_alpha" "32")
fi

if [[ "$other_params" != *"--learning_rate"* ]]; then 
    argv+=("--learning_rate" "1e-6")
fi


CUDA_LAUNCH_BLOCKING=1
srun --jobid $SLURM_JOBID python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_backend c10d \
    --rdzv_id $MASTER_PORT --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    -m  train.train_mem \
    --lora_enable True  \
    --lora_name_or_path $LORA_NAME_OR_PATH \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_BASE \
    --train_data_path $DATA_PATH \
    --prompt_type $prompt_type \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to wandb \
    ${argv[@]}
