BASE_PATH=/mnt/petrelfs/jiangshuyang.p/oss
# TASK_PATH=s3://syj_test/datasets/diverse_domain/train_wo_triviaqa
TASK_PATH=/mnt/hwfile/medai/jiangshuyang.p/datasets/taia_train

TRAINING_DATA=tulu_v2_0_3


MODEL_BASE=/mnt/petrelfs/jiangshuyang.p/models/Meta-Llama-3-8B-Instruct

CKPT=test_new_env


SAVE_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
LORA_NAME_OR_PATH=None

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
if [ ! -f "${SAVE_PATH}/adapter_config.json" ]; then 
    sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/slurm/sft_lora.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $CKPT $LORA_NAME_OR_PATH & sleep 1
fi

# bash scripts/eval/bash/eval_models_per_dataset.sh $MODEL_BASE $TRAINING_DATA $CKPT