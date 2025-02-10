import argparse
import torch
import os
import json
from tqdm import tqdm, trange
# import shortuuid
from evalplus.data import get_human_eval_plus
from evalplus.data import get_mbpp_plus

from conversations import conv_templates, SeparatorStyle
from model.builder import load_multi_lora_model, load_multi_norm_model, load_pretrained_model, load_molora_pretrained_model, load_automerge_model
from utils import disable_torch_init, get_model_name_from_path
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import DataLoader
import pandas as pd 
from utils import client
from copy import deepcopy
# from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_loss(logits, labels, attention_mask, vocab_size):
    from torch.nn import CrossEntropyLoss
    labels = labels.masked_fill(~attention_mask, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    B, N, C = shift_logits.shape
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # this loss is [-1, ], we need to reshape it to [B, N]
    loss = loss.reshape(B, N)
    # we must know that some positions are 0-loss because of ignore_index, we need to ignore these
    loss_sum = loss.sum(dim=-1)
    loss_actual_position = torch.not_equal(loss, 0).sum(dim=-1)
    loss = loss_sum / loss_actual_position  # [B, ]
    return loss


def generate_func(model, input_ids, **kwargs):
    if input_ids.dim() == 1:
        # only one item
        input_ids = input_ids.unsqueeze(0)
    max_new_tokens = kwargs.pop("max_new_tokens", args.max_new_tokens)
    tokenizer = kwargs.pop("tokenizer")
    sequence_bias = kwargs.pop("sequence_bias", None)
    if args.conv_mode == 'llama3':
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            sequence_bias=sequence_bias,
            use_cache=True)
    return output_ids

def switch_expert(one_model, other_model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  **kwargs):
    # calculate the loss of input_ids on base_model
    print(input_ids.shape)
    with torch.inference_mode():
        one_model_logits = one_model(input_ids).logits
        one_model_loss = get_loss(one_model_logits, input_ids, attention_mask, one_model.config.vocab_size)
        # calculate the loss of input_ids on loaded_model
        other_model_logits = other_model(input_ids).logits
        other_model_loss = get_loss(other_model_logits, input_ids, attention_mask, other_model.config.vocab_size)

    # based on the loss scale, partition the inputs to different models
    mask = one_model_loss < other_model_loss
    one_model_inputs_id = torch.nonzero(mask).squeeze(-1) # [k, ]
    other_model_inputs_id = torch.nonzero(~mask).squeeze(-1) # [k, ]
    # print(one_model_inputs_id, other_model_inputs_id)
    if one_model_inputs_id.shape[0] == 0:
        one_model_outputs = []
    else:
        one_model_outputs = generate_func(one_model, input_ids[one_model_inputs_id], **kwargs)
    if other_model_inputs_id.shape[0] == 0:
        other_model_outputs = []
    else:
        other_model_outputs = generate_func(other_model, input_ids[other_model_inputs_id], **kwargs)
    if one_model_outputs == []:
        return other_model_outputs 
    if other_model_outputs == []:
        return one_model_outputs
    
    tokenizer = kwargs.pop("tokenizer")
    # concat one_model_outputs and other_model_outputs with tokenizer.eos_token_id 
    max_len = max(one_model_outputs.shape[1], other_model_outputs.shape[1])
    # print(max_len)
    one_model_outputs = torch.cat([one_model_outputs, 
                                   torch.full((one_model_outputs.shape[0], max_len-one_model_outputs.shape[1]), tokenizer.eos_token_id, dtype=torch.long, device=one_model_outputs.device)], dim=1)
    other_model_outputs = torch.cat([other_model_outputs, 
                                     torch.full((other_model_outputs.shape[0], max_len-other_model_outputs.shape[1]), tokenizer.eos_token_id, dtype=torch.long, device=other_model_outputs.device)], dim=1)
    # print(one_model_outputs.shape, other_model_outputs.shape)
    total_outputs = torch.cat([one_model_outputs, other_model_outputs], dim=0)
    # merge the outputs
    results = torch.zeros(total_outputs.shape[0], total_outputs.shape[1]).to(total_outputs)
    
    results = torch.scatter_add(results, 0, torch.cat([one_model_inputs_id, other_model_inputs_id], dim=0).unsqueeze(1).expand(-1, total_outputs.shape[1]), total_outputs)

    
    return results
    

# Custom dataset class
class CustomDataset:
    def __init__(self, questions, batch_size, task_specific_prompt, dataset_name='default', tokenizer=None):
        self.questions = questions
        self.batch_size = batch_size
        self.size = len(questions)

        self.task_specific_prompt = task_specific_prompt
        self.dataset_name = dataset_name

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        bz = self.batch_size

        # return question, ansewr, additional info
        questions = []
        prompts = []
        answers = []
        additional_infos = []
        for i in range(index*bz, (index+1)*bz):
            if i < self.size:
                # conv = self.conv.copy()
                if self.dataset_name.endswith("plus"):
                    question = self.questions[i]['prompt']
                    questions.append(question)

                    prompt_text = self.tokenizer.apply_chat_template(question + self.task_specific_prompt, tokenize=False, add_generation_prompt=True)
                        # conv.append_message(conv.roles[0], question+self.task_specific_prompt)
                        # conv.append_message(conv.roles[1], None)
                    prompts.append(prompt_text)
                    answers.append(None)
                    additional_infos.append(self.questions[i]['task_id'])
                else:
                    line = self.questions[i]
                    question = line['conversations'][0]['value']
                    questions.append(question)
                    prompt_text = self.tokenizer.apply_chat_template(question + self.task_specific_prompt, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt_text) 
                    
                    answers.append(line['conversations'][1]['value'] if len(line['conversations']) > 1 else None)
                    additional_infos.append(line['eval'] if 'eval' in line else None)

        return questions, prompts, answers, additional_infos

    def __len__(self):
        return len(self.questions) // self.batch_size + 1

    def __iter__(self):
        # 返回迭代器对象本身
        return self
    
    def __next__(self):
        if self.index < len(self.questions):
            # 返回下一个值并更新索引
            item = self.questions[self.index]
            self.index += 1
            return item
        else:
            # 没有更多元素时抛出StopIteration异常
            raise StopIteration


# DataLoader
def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def convert_to_json(questions):
    # questions is a pandas dataframe, which is to be converted to a list object
    # each element in the list is a dictionary
    # the column name of questions is the key of the dictionary
    # the value of the dictionary is the value of the corresponding column
    questions = questions.to_dict(orient='records')
    return questions


# def get_fewshot_examples(dataset_name, num_samples=3):
#     dataset = dataset_name.split("_")[0]
#     mapping = {
#         "bbh": "ming/eval/fewshot_examples/bbh_prompt_cot/example.txt",
#         "svamp": "ming/eval/fewshot_examples/svamp_prompt_cot/example.txt",
#         "math": "ming/eval/fewshot_examples/math_prompt_cot/example.txt",
#         "commonsense": "ming/eval/fewshot_examples/commonsense_qa_prompt_cot/example.txt",
#         "logiqa": "ming/eval/fewshot_examples/logiqa_en_prompt_cot/example.txt",
#         "mmlu": "ming/eval/fewshot_examples/mmlu_prompt_cot/example.txt",
#         "mmedbench": "ming/eval/fewshot_examples/mmedbench_en_prompt_cot/example.txt"
#     }
#     samples = {
#         "mmlu": 3,
#         "commonsense": 1,
#         "logiqa": 10,
#         "mmedbench": 10,
#         "svamp": 10
#     }
#     if dataset in mapping:
#         print("Loading few shot examples...")
#         few_shot_prompt = open(mapping[dataset], 'r').read()
#         each_examples = few_shot_prompt.split("Problem: ")
#         few_shot_prompt = "Problem: ".join(each_examples[:samples[dataset]])
#         print(few_shot_prompt, flush=True)
#     else:
#         few_shot_prompt = "Please directly answer with the answer letter.\n"
#     return few_shot_prompt
#     # pass

four_choices_datasets = ["logiqa_en_cot", "mmedbench_en_cot", "mmlu_cot", "sat_math_cot", "mmlu_math_cot", "mmedbench_zh_cot", "medmcqa_cot", 'logiqa_en_prompt_cot', "mmedbench_en_prompt_cot", "mmlu_prompt_cot", 'med_mmlu_cot']
five_choices_datasets = ['commonsense_qa_cot', "CMExam_zh_cot", "medqa_cot", 'commonsense_qa_prompt_cot', "MedQA_cot"]
three_choices_datasets = ['pubmedqa_cot', 'pubmedqa_c_cot', "MNLI_cot"]
two_choices_datasets = ["CoLA_cot", "QNLI_cot", "SST-2_cot", "MRPC_cot", "RTE_cot", "QQP_cot"]

def eval_model(args):
    # Model
    dataset_name = args.question_file.split("/")[-1].split(".")[0]

    # load args.question_file, which is a csv file
    if dataset_name == "mbpp_plus":
        
        questions = get_mbpp_plus()
        questions = [{"prompt": problem['prompt'], "task_id": task_id} for task_id, problem in questions.items()]
    elif dataset_name == "humaneval_plus":
        questions = get_human_eval_plus()
        # print(questions)
        questions = [{"prompt": problem['prompt'], "task_id": task_id} for task_id, problem in questions.items()]
    else:
        if args.question_file.endswith(".csv"):
            questions = pd.read_csv(args.question_file)
            questions = convert_to_json(questions)
        elif args.question_file.endswith(".jsonl"):
            questions = client.read_jsonl(os.path.expanduser(args.question_file))
        else:
            # a json file
            questions = client.read_json(os.path.expanduser(args.question_file))
    
    


    # data_loader = create_data_loader(questions, tokenizer, model.config)

    sequence_bias = None
    def get_tokens_as_tuple(word):
        return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])

    task_specific_prompt = ""
    

    if dataset_name in ['bbh_prompt_cot', 'math_prompt_cot']:
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."
    elif dataset_name in ['mmedbench_en_prompt_cot', "mmlu_prompt_cot"]:
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
    elif dataset_name in ['logiqa_en_prompt_cot']:
        task_specific_prompt = "\n\nPlease think step by step and give your answer in the end."
    elif dataset_name == "commonsense_qa_prompt_cot":
        task_specific_prompt = "\n\nLet's think step by step. Please format the final answer at the end of the response as: The answer is {answer}."
    elif dataset_name == 'svamp_100_prompt_cot':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)


    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    if args.resume and os.path.exists(answers_file):
        data = client.read_jsonl(answers_file)
        current_file_num = len(data)

        questions = questions[current_file_num:]
        ans_file = open(answers_file, "a", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')

    # print(tokenizer.pad_token, tokenizer.eos_token)
    if len(questions) == 0:
        exit(0)
    else:
        
        disable_torch_init()
        print(args.model_path)
        if len(args.model_path) == 1:
            args.model_path = args.model_path[0]
            model_path = os.path.expanduser(args.model_path)

        else:
            model_path = [os.path.expanduser(path) for path in args.model_path]

        
        if args.new_lora_name and isinstance(model_path, str):
            tokenizer, model, context_len, tokenizer_with_prefix_space, base_model = load_automerge_model(model_path, args.model_base, args)
        elif args.new_lora_name and isinstance(model_path, list):
            tokenizer, model, context_len, tokenizer_with_prefix_space, base_model = load_multi_norm_model(model_path, args.model_base, args)
        elif isinstance(model_path, list):
            tokenizer, model, context_len, tokenizer_with_prefix_space, base_model = load_multi_lora_model(model_path, args.model_base, args)
        else:
            tokenizer, model, context_len, tokenizer_with_prefix_space, base_model = load_pretrained_model(model_path, args.model_base, args=args)
        tokenizer.padding_side = "left"
        tokenizer_with_prefix_space.padding_side = "left"
        model: torch.nn.Module
        model.eval()
        if isinstance(model_path, list):
            model_path = model_path[0]
        if "32b" in model_path.lower() or (args.model_base is not None and "32b" in args.model_base.lower()):
            args.batch_size = 4
        if "truthfulqa_mc1" in dataset_name:
            args.batch_size = 1

        dataset = CustomDataset(questions, batch_size=args.batch_size, conv_mode=args.conv_mode, task_specific_prompt=task_specific_prompt, dataset_name=dataset_name , tokenizer=tokenizer, is_base=args.is_base, add_few_shot=args.add_few_shot, num_samples=args.fewshot_samples)

    for idx in trange(len(dataset)):
        questions, prompts, answers, additional_infos = dataset[idx]
        if len(questions) == 0:
            break

        
            # print(sequence_bias)
        input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)

        if args.conv_mode == 'llama3':
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            terminators = tokenizer.eos_token_id
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True if args.temperature > 0 else False,
                attention_mask=attention_mask,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                sequence_bias=sequence_bias,
                use_cache=True)
        # print(input_ids.shape, output_ids.shape)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # print("original outputs: ",tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True) )
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # print("cut outputs: ", outputs)
        
        # print("[FIRST OUTPUT]: ", outputs)

        if "cot" in dataset_name:
            if "zh" in prompts[0]:
                cot_prompt = "\n答案为"
            elif dataset_name in ["CMExam_cot", "CMB_cot", "cmmlu_cot", "ceval_cot", "medqa_mainland_cot"]:
                cot_prompt = "\n答案为"
            else:
                cot_prompt = "\nThe answer is "

            # cut_length = len(conv.sep2)
            cot_prompts = [(prompt + output + f"{' ' if output.strip().endswith('.') else '. '}{cot_prompt}") for prompt, output in zip(prompts, outputs)]
            input_ids = tokenizer(cot_prompts, return_tensors='pt', padding=True).input_ids.to(device='cuda', non_blocking=True)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)


            if dataset_name in four_choices_datasets:
                cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
                cot_max_new_tokens = 1
            elif dataset_name in five_choices_datasets:
                cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D", "E"]}
                cot_max_new_tokens = 1
            elif dataset_name in three_choices_datasets:
                cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C"]}
                cot_max_new_tokens = 1
            elif dataset_name in two_choices_datasets:
                cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B"]}
                cot_max_new_tokens = 1
            elif dataset_name in ['winogrande_cot']:
                cot_sequence_bias = {get_tokens_as_tuple(x): 100.0 for x in ["A", "B"]}
                cot_max_new_tokens = 1

            elif dataset_name in ['truthfulqa_mc1_cot']:
                if "num_choices" in additional_infos[0]:
                    num_choices = additional_infos[0]['num_choices']
                    choice_idx = "ABCDEFGHIJKLMNOPQSTUVWXYZ"[:num_choices]
                    cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in list(choice_idx)}
                    cot_max_new_tokens = 1
            else:
                cot_sequence_bias = None
                cot_max_new_tokens = 50

            with torch.inference_mode():
                answer_output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=cot_max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                sequence_bias=cot_sequence_bias,
                use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != answer_output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
            answer_outputs = tokenizer.batch_decode(answer_output_ids[:, input_token_len:], skip_special_tokens=True)
            # print(answer_outputs)
            outputs = [f"{output}{' ' if output.strip().endswith('.') else '. '}{cot_prompt}{answer_output}" for output, answer_output in zip(outputs, answer_outputs)]
            
        if dataset_name == 'humaneval_plus' or dataset_name == 'mbpp_plus':
            for question, output, answer, additional_info in zip(questions, outputs, answers, additional_infos):
                ans_file.write(json.dumps({
                    "task_id": additional_info,
                    "solution": output
                }) + "\n")
        else:
            for question, output, answer, additional_info in zip(questions, outputs, answers, additional_infos):
                ans_file.write(json.dumps({"prompt": question,
                                        "text": output,
                                        "solution": answer,
                                        "additional_info": additional_info,
                                        "metadata": {}}, ensure_ascii=False) + "\n",)
        ans_file.flush()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m", nargs='+')
    parser.add_argument("--model-base", type=str, default=None)


    parser.add_argument("--fewshot_samples", type=int, default=1)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")


    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-logit-bias", action='store_true')
    parser.add_argument("--logit-score", default=100.0)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--infer-answer", action="store_true")
    parser.add_argument("--only-load", type=str, default=None)
    # parser.add_argument("--use-loracl", action="store_true")
    parser.add_argument('--batch-size', type=int, default=8)
    
    


    parser.add_argument("--new_lora_name", default=None, type=str, help="the new state dict name for automerge lora")
    args = parser.parse_args()

    eval_model(args)