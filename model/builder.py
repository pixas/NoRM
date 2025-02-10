#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
from tqdm import tqdm, trange
import shutil
from copy import copy, deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from model import MoLoRAQwenForCausalLM, MoLoRALlamaForCausalLM, MoLoRAQwenDecoderLayer, MoLoRALlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import json
from utils import client
from safetensors.torch import load_file
import re 
from peft import get_peft_model





def get_model_class_from_path(model_path: str):
    model_path = model_path.lower()
    if "qwen" in model_path or "ming" in model_path:
        return Qwen2DecoderLayer
    if "llama" in model_path:
        return LlamaDecoderLayer
    


def find_pair(lora_state_dict,):
    pairs = {}
    for key in lora_state_dict.keys():
        prefix = "base_model.model."
        suffix = ".lora_A.weight"
        base_key = key[len(prefix):-len(suffix)] + ".weight"
        pairs[key] = base_key 
    return pairs




def load_automerge_model(model_path, model_base, args, load_8bit=False, load_4bit=False, use_logit_bias=False, device_map="auto", device="cuda", **bnb_model_kwargs):
    # only_load = getattr(args, "only_load", None)
    new_lora_name = getattr(args, "new_lora_name", None)
    return_base = getattr(args, "return_base", False)
    use_logit_bias = getattr(args, "use_logit_bias", False)
    
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    if model_base is not None:
        # PEFT model
        from peft import LoraConfig
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
        if bnb_model_kwargs != {}:
            model = AutoModelForCausalLM.from_pretrained(model_base, trust_remote_code=True, **bnb_model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, trust_remote_code=True,**kwargs)
        # base_state_dict = model.state_dict()
        base_model = model
        print(f"Loading LoRA weights from {model_path}")
        lora_config = LoraConfig.from_pretrained(model_path)
        new_state_dict = load_file(os.path.join(model_path, f"{new_lora_name}.safetensors"))
        memory = {}
        for key, param in new_state_dict.items():
            if "lora_A" in key:
                cur_r = param.shape[0]
                memory[cur_r] = memory.get(cur_r, [])
                memory[cur_r].extend([key, key.replace("lora_A", "lora_B")])
        
        for rank, key_list in tqdm(memory.items()):
            print(rank)
            related_layers = sorted(list(set([int(x.split(".")[4]) for x in key_list])))
            print(related_layers)
            related_modules = list(set([".".join(x.split(".")[5:7]) for x in key_list]))
            print(related_modules)
            print("=" * 100)
            cur_config = deepcopy(lora_config)
            cur_config.r = rank 
            cur_config.lora_alpha = rank # NOTE: this is a non-trivial trial from empirical results
            target_modules = list(set([".".join(x.split(".")[3:7]) for x in key_list]))
            cur_config.target_modules = target_modules 
            model = get_peft_model(model, cur_config)
            model.load_state_dict(new_state_dict, strict=False)
            model = model.merge_and_unload()
            
        print('Convert to FP16...')
        model.to(torch.float16)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
        model.to(torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
        if tokenizer_with_prefix_space.pad_token_id is None:
            tokenizer_with_prefix_space.pad_token_id = tokenizer_with_prefix_space.eos_token_id
    else:
        tokenizer_with_prefix_space = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
        
    if return_base:
        return tokenizer, model, context_len, tokenizer_with_prefix_space, base_model 
    return tokenizer, model, context_len, tokenizer_with_prefix_space, None

def load_pretrained_model(model_path, model_base, args, load_8bit=False, load_4bit=False, use_logit_bias=False, device_map="auto", device="cuda"):
    # unload_ffn = getattr(args, "unload_ffn", False)

    return_base = getattr(args, "return_base", False)
    # unload_attn = getattr(args, "unload_attn", False)
    only_load = getattr(args, "only_load", None)

    use_logit_bias = getattr(args, "use_logit_bias", False)


    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    
    if model_base is not None:
        # PEFT model
        from peft import PeftModel, LoraConfig
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, trust_remote_code=True,**kwargs)
        
        base_model = model
        print(f"Loading LoRA weights from {model_path}")
        lora_config = LoraConfig.from_pretrained(model_path)

            
        if only_load == 'attn':
            lora_config.target_modules = ['v_proj', 'k_proj', 'q_proj', 'o_proj']
        if only_load == 'ffn':
            lora_config.target_modules = ['up_proj', 'down_proj', 'gate_proj']
                    
        print(lora_config)
        
        model = PeftModel.from_pretrained(model, model_path, config=lora_config)
        print(f"Merging weights")
        model = model.merge_and_unload()
        print('Convert to FP16...')
        model.to(torch.float16)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
        model.to(torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
        if tokenizer_with_prefix_space.pad_token_id is None:
            tokenizer_with_prefix_space.pad_token_id = tokenizer_with_prefix_space.unk_token_id if tokenizer_with_prefix_space.unk_token_id is not None else tokenizer_with_prefix_space.eos_token_id
    else:
        tokenizer_with_prefix_space = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
        
    if return_base:
        return tokenizer, model, context_len, tokenizer_with_prefix_space, base_model 
    return tokenizer, model, context_len, tokenizer_with_prefix_space, None



def load_multi_lora_model(model_path, model_base, args, load_8bit=False, load_4bit=False, use_logit_bias=False, device_map="auto", device="cuda"):
    # unload_ffn = getattr(args, "unload_ffn", False)

    use_logit_bias = getattr(args, "use_logit_bias", False)



    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if model_base is not None:
        # PEFT model
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, trust_remote_code=True,**kwargs)
        
        # base_model = model
        print(f"Loading LoRA weights from {model_path}")
        assert isinstance(model_path, list), "model_path should be a list of paths"
        for each_model_path in model_path:
            model = PeftModel.from_pretrained(model, each_model_path,)
            model = model.merge_and_unload()
            print(f"Load Lora weights from {each_model_path}")
        print('Convert to FP16...')
        model.to(torch.float16)
    else:
        raise NotImplementedError
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
        if tokenizer_with_prefix_space.pad_token_id is None:
            tokenizer_with_prefix_space.pad_token_id = tokenizer_with_prefix_space.unk_token_id if tokenizer_with_prefix_space.unk_token_id is not None else tokenizer_with_prefix_space.eos_token_id
    else:
        tokenizer_with_prefix_space = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
        
    # if return_base:
    #     return tokenizer, model, context_len, tokenizer_with_prefix_space, base_model 
    return tokenizer, model, context_len, tokenizer_with_prefix_space, None


def load_multi_norm_model(model_path, model_base, args, load_8bit=False, load_4bit=False, use_logit_bias=False, device_map="auto", device="cuda", **bnb_model_kwargs):

    new_lora_name = getattr(args, "new_lora_name", None)
    return_base = getattr(args, "return_base", False)
    use_logit_bias = getattr(args, "use_logit_bias", False)
    
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    # if model_base is not None:
    # PEFT model
    from peft import PeftModel, LoraConfig
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
    if bnb_model_kwargs != {}:
        model = AutoModelForCausalLM.from_pretrained(model_base, trust_remote_code=True, **bnb_model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, trust_remote_code=True,**kwargs)
    # base_state_dict = model.state_dict()
    base_model = model
    assert isinstance(model_path, list), "model_path should be a list of paths"
    for each_model_path in model_path:
        
        print(f"Loading LoRA weights from {each_model_path}")
        lora_config = LoraConfig.from_pretrained(each_model_path)
        new_state_dict = load_file(os.path.join(each_model_path, f"{new_lora_name}.safetensors"))
        memory = {}
        for key, param in new_state_dict.items():
            if "lora_A" in key:
                cur_r = param.shape[0]
                memory[cur_r] = memory.get(cur_r, [])
                memory[cur_r].extend([key, key.replace("lora_A", "lora_B")])
        
        for rank, key_list in tqdm(memory.items()):
            print(rank)
            related_layers = sorted(list(set([int(x.split(".")[4]) for x in key_list])))
            print(related_layers)
            related_modules = list(set([".".join(x.split(".")[5:7]) for x in key_list]))
            print(related_modules)
            print("=" * 100)
            cur_config = deepcopy(lora_config)
            cur_config.r = rank 
            cur_config.lora_alpha = rank # NOTE: this is a non-trivial trial from empirical results
            target_modules = list(set([".".join(x.split(".")[3:7]) for x in key_list]))
            cur_config.target_modules = target_modules 
            model = get_peft_model(model, cur_config)
            model.load_state_dict(new_state_dict, strict=False)
            model = model.merge_and_unload()
            
    print('Convert to FP16...')
    model.to(torch.float16)
        
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
        if tokenizer_with_prefix_space.pad_token_id is None:
            tokenizer_with_prefix_space.pad_token_id = tokenizer_with_prefix_space.eos_token_id
    else:
        tokenizer_with_prefix_space = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
        
    if return_base:
        return tokenizer, model, context_len, tokenizer_with_prefix_space, base_model 
    return tokenizer, model, context_len, tokenizer_with_prefix_space, None