CHECKPOINT_DIR="" # the checkpoint directory in the local machine

TASK_PATH="" # the path to the task data

TRAINING_DATA=metamathqa  # ${TASK_PATH}/${TRAINING_DATA}.json should exist


MODEL_BASE="" # the base model path, must be Llama3-8B Instruct, Qwen 2-7B Instruct, or Mistral 7B Instruct

CKPT=llama3-8b-exp # as we set the prompt_type based on the CKPT, we need to set it following the below rule
# if [[ $CKPT == *"llama2-7b"* ]]; then
#     prompt_type="llama2"
# elif [[ $CKPT == *"llama3-8b"* ]]; then
#     prompt_type="llama3"
# elif [[ $CKPT == *"mistral-7b"* ]]; then 
#     prompt_type="mistral"
# else
#     prompt_type="qwen"
# fi


SAVE_PATH=${CHEKPOINT_DIR}/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
LORA_NAME_OR_PATH=None  # can reproduce continual lora experiments, which is the previous task's lora path

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
if [ ! -f "${SAVE_PATH}/adapter_config.json" ]; then 
    sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/slurm/sft_lora.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $CKPT $LORA_NAME_OR_PATH & sleep 1
fi

bash scripts/eval/bash/eval_models_per_dataset.sh $MODEL_BASE $TRAINING_DATA $CKPT