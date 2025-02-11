import torch 
import torch.nn as nn 

import torch.nn.functional as F 
from copy import deepcopy
import json
from petrel_client.client import Client
import io
import os 
from functools import wraps
from contextlib import contextmanager

def proxy_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ori_http_proxy = os.environ.get('http_proxy')  # 获取原始的http_proxy值
        ori_https_proxy = os.environ.get("https_proxy")
        os.environ['http_proxy'] = ''  # 在函数执行前将http_proxy设为空字符串
        os.environ['https_proxy'] = ''
        result = func(*args, **kwargs)  # 执行函数
        os.environ['http_proxy'] = ori_http_proxy if ori_http_proxy is not None else ''  # 函数执行后恢复原始的http_proxy值
        os.environ['https_proxy'] = ori_https_proxy if ori_https_proxy is not None else ''
        return result
    return wrapper

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class CephOSSClient:
    
    @proxy_decorator
    def __init__(self, conf_path: str = "~/petreloss.conf") -> None:
        self.client = Client(conf_path)
    
    # @proxy_decorator
    def read_json(self, json_path):
        if json_path.startswith("s3://"):
            data = json.loads(self.client.get(json_path))
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data 

    # @proxy_decorator
    def write_json(self, json_data, json_path, **kwargs):
        if json_path.startswith("s3://"):
            self.client.put(json_path, json.dumps(json_data, **kwargs).encode("utf-8"))
        else:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, **kwargs)
        return 1

    def read_jsonl(self, jsonl_path):
        if jsonl_path.startswith("s3://"):
            bytes = self.client.get(jsonl_path)
            data = bytes.decode('utf-8').split("\n")
            data = [json.loads(x) for x in data if x != ""]
        else:
            data = [json.loads(x) for x in open(jsonl_path, encoding='utf-8', mode='r')]
        return data 
    
    def write_jsonl(self, jsonl_data, jsonl_path, **kwargs):
        if jsonl_path.startswith("s3://"):
            if isinstance(jsonl_data, list):
                large_bytes = "\n".join([json.dumps(x, ensure_ascii=False) for x in jsonl_data]).encode("utf-8")
            else:
                large_bytes = (json.dumps(x, ensure_ascii=False) + "\n").encode('utf-8')
            with io.BytesIO(large_bytes) as f:
                self.client.put(jsonl_path, f)
        else:
            with open(jsonl_path, 'w', **kwargs) as f:
                for x in jsonl_data:
                    f.write(json.dumps(x, ensure_ascii=False))
                    f.write("\n")
        return 1

    # @proxy_decorator
    def read_txt(self, txt_path):
        if txt_path.startswith("s3://"):
            bytes = self.client.get(txt_path)
            data = bytes.decode('utf-8')
        else:
            with open(txt_path, 'r', encoding='utf-8') as f:
                data = f.read()
        return data 

    # @proxy_decorator
    def write_text(self, txt_data, txt_path):
        if txt_path.startswith("s3://"):
            large_bytes = txt_data.encode("utf-8")
            with io.BytesIO(large_bytes) as f:
                self.client.put(txt_path, f)
        else:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(txt_data)
        return 1
    
    # @proxy_decorator
    def save_checkpoint(self, data, path):
        if "s3://" not in path:
            assert os.path.exists(path), f'No such file: {path}'
            torch.save(data, path)
        else:
            with io.BytesIO() as f:
                torch.save(data, f)
                self.client.put(f.getvalue(), path)
        return 1 

    # @proxy_decorator
    def load_checkpoint(self, path, map_location=None):
        if "s3://" not in path:
            assert os.path.exists(path), f'No such file: {path}'
            return torch.load(path, map_location=map_location)
        else:
            file_bytes = self.client.get(path)
            buffer = io.BytesIO(file_bytes)
            res = torch.load(buffer, map_location=map_location)
            return res
    
    # @proxy_decorator
    def exists(self, file_path):
        if "s3://" not in file_path:
            return os.path.exists(file_path)
        else:
            return self.client.contains(file_path)
    
    def read_csv(self, path):
        if "s3://" in path:
            bytes = self.client.get(path)
            data = bytes.decode('utf-8').split("\n")
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = f.readlines()
        return data

    def read(self, path: str):
        mapping_processing = {
            "csv": self.read_csv,
            "json": self.read_json,
            "jsonl": self.read_jsonl,
            "txt": self.read_txt
        }
        suffix = path.split(".")[-1]
        return mapping_processing[suffix](path)
    
    def write(self, data, path: str):
        mapping_processing = {
            "csv": self.write_text,
            "json": self.write_json,
            "jsonl": self.write_jsonl,
            "txt": self.write_text
        }
        suffix = path.split(".")[-1]
        return mapping_processing[suffix](data, path)

client = CephOSSClient("~/petreloss.conf")

@contextmanager
def add_proxy():
    proxy_url = os.environ.get("proxy_url", "")
    os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = proxy_url
    try:
        yield
    finally:
        os.environ['http_proxy'] = os.environ['HTTP_PROXY'] = os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = ''
        