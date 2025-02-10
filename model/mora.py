
import torch 
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from peft.utils import _get_submodules

import torch.nn as nn

import math


from typing import Optional, List, Union


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
        
def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def check_target_module_exists(lora_config, key, target_modules):
    target_module_found = any(key.endswith(module_name) for module_name in target_modules)
    return target_module_found


def mark_only_lora_as_trainable(model: nn.Module, bias) -> None:
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False


    if bias == "none":
        return

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, MoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
    
    
def create_mora_module(lora_config, target, add_bias=True, mora_type=1):
    in_features, out_features = target.in_features, target.out_features
    new_module = MoRALinear(target, 
                              r=lora_config.r,
                              lora_alpha=lora_config.lora_alpha,
                              lora_dropout=lora_config.lora_dropout,
                              use_rslora=lora_config.use_rslora,
                              use_mora=True,
                              mora_type=mora_type,
                              bias=add_bias)
    return new_module



def merge_and_unload(model):
    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue
        if isinstance(target, MoRALayer):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            target.merge()
            replace_module(parent, target_name, new_module, target)



    return model

def replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias"):
        if old_module.bias is not None:
            new_module.bias = old_module.bias

    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state

    new_module.to(old_module.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            module.to(old_module.weight.device)

def get_mora_model(model, lora_config,  **kwargs):

    decoder_type = kwargs.pop("decoder_type", Qwen2DecoderLayer)
    inference_mode = kwargs.pop("inference_mode", False)

    wrap_modules = kwargs.pop("wrap_modules", ("mlp"))
    mora_type = kwargs.pop("mora_type", 1)
    # find linear modules with "switch" in their attributes
    key_list = [key for key, _ in model.named_modules()]
    target_module_names = set()

    for name, module in model.named_modules():
        rank0_print(name, module)
        if isinstance(module, torch.nn.Linear):
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), decoder_type) and any(module in name for module in wrap_modules):
                names = name.split(".")
                target_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_module_names = list(target_module_names)
    for key in key_list:
        if not check_target_module_exists(lora_config, key, target_module_names):
            continue
            
        parent, target, target_name = _get_submodules(model, key)
        # print(parent, target_name)
        if hasattr(target, "bias"):
            if target.bias is not None:
                add_bias = True 
            else: 
                add_bias = False
        else:
            add_bias = False
        new_module = create_mora_module(
            lora_config, 
            target, 
            add_bias=add_bias,
            mora_type=mora_type)
        setattr(parent, target_name, new_module)
        
        # singular matrix decomposition, 
        # first decompose the target.weight into (m-r) and r, the former as the pretrained weight and the latter as the initialization of the new_module's lora_A and lora_B

         
        new_module.base_layer.weight.data = target.weight.data
        if hasattr(target, "bias"):
            if target.bias is not None:
                new_module.base_layer.bias = target.bias

        new_module.to(target.weight.device)
        
        if getattr(target, "state", None) is not None:
            new_module.state = target.state
            new_module.to(target.weight.device)
        
        del target
    if not inference_mode:
        mark_only_lora_as_trainable(model, getattr(lora_config, "bias", "none"))
    if inference_mode:
        # pass
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
    else:
        for m, p in model.named_modules():
            if isinstance(m, MoRALinear):
                m.reset_parameters()

    
    return model



class MoRALayer():
    def __init__(
        self, 
        base_layer: nn.Module, 
        **kwargs
    ):
        self.r = None
        self.lora_alpha = None
        # Optional dropout
        self.lora_dropout = None
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = None
        self.mora_type = None 
        self.use_mora = False 
        self.base_layer = base_layer
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        self.in_features = in_features
        self.out_features = out_features    
    
    def update_layer(
        self, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora=False, use_mora: bool = True, mora_type: int = 1,
    ):
        self.r = r 
        self.lora_alpha = lora_alpha 
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout = lora_dropout_layer
        self.use_mora = False 
        self.mora_type = mora_type 
        
        if use_mora:
            new_r = int(math.sqrt((self.in_features + self.out_features)*r)+0.5)
            if self.mora_type == 6:
                # type 6 require new_r to be even for RoPE
                new_r = new_r//2*2
            self.lora_A = nn.Linear(new_r, new_r, bias=False)
            self.r = new_r
            nn.init.zeros_(self.lora_A.weight)
            self.lora_B = self.lora_A
            self.scaling = 1.0
            self.use_mora = True 
            self.mora_type = mora_type
    
    def reset_lora_parameters(self, init_lora_weights):
        if not init_lora_weights:
            return 
        
        if self.use_mora:
            nn.init.zeros_(self.lora_A.weight)
            self.lora_B = self.lora_A
            if self.mora_type is not None:
                self.mora_type = self.mora_type
            return
    
    def _apply_mora(self, x, lora_A, lora_B, scaling):
        in_f, out_f = self.in_features, self.out_features
        r = self.r 
        mora_type = self.mora_type
        if mora_type == 1 or mora_type == 4:
            sum_inter = in_f // r
            if in_f % r != 0:
                pad_size = r - in_f % r
                # x = torch.cat([x, torch.zeros_like(x)[..., :pad_size]], dim=-1)
                x = torch.cat([x, x[..., :pad_size]], dim=-1)
                sum_inter += 1
            in_x = x.view(*x.shape[:-1], sum_inter, r).sum(dim=-2)
        elif mora_type == 2 or mora_type == 3:
            mr, nr = in_f//r+1, in_f//r
            m, n = in_f - r*nr, r*mr - in_f
            mm, nn = m*mr, n * nr
            if m > 0:
                x_m, x_n = x[..., :mm], x[..., mm:]
                x_m = x_m.view(*x.shape[:-1], m, mr).sum(dim=-1)
                x_n = x_n.view(*x.shape[:-1], n, nr).sum(dim=-1)
                in_x = torch.cat([x_m, x_n ], dim=-1)
            else:
                in_x = x.view(*x.shape[:-1], n, nr).sum(dim=-1)
        elif mora_type == 6:
            sum_inter = in_f // r
            rb1 = in_f//r if in_f % r == 0 else in_f//r + 1
            if in_f % r != 0:
                pad_size = r - in_f % r
                x = torch.cat([x, x[..., :pad_size]], dim=-1)
                sum_inter += 1
            in_x = x.view(*x.shape[:-1], sum_inter, r)
            if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
                inv_freq = 1.0 / (10000 ** (torch.arange(0, r, 2).float() / r))
                t = torch.arange(rb1)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
                self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
            rh_in_x = torch.cat((-in_x[..., r//2:], in_x[..., :r//2]), dim=-1)
            in_x = in_x*self.cos + rh_in_x*self.sin
        
        out_x = lora_A(in_x)

        if mora_type == 1 or mora_type == 3:
            repeat_time = out_f // r
            if out_f % r != 0:
                repeat_time += 1
            out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :out_f]
        elif mora_type == 2 or mora_type == 4:
            mr, nr = out_f//r+1, out_f//r
            m, n = out_f - r*nr, r*mr - out_f
            mm, nn = m*mr, n * nr
            if m > 0:
                out_x = torch.cat([torch.repeat_interleave(out_x[..., :m], mr, dim=-1),
                                   torch.repeat_interleave(out_x[..., m:], nr, dim=-1)]
                                  , dim=-1)
            else:
                out_x = torch.repeat_interleave(out_x, nr, dim=-1)
        elif mora_type == 6:
            out_x = out_x.view(*x.shape[:-1], -1)[..., :out_f]
            if out_x.shape[-1] < out_f:
                repeat_time = out_f // out_x.shape[-1]
                if out_f % out_x.shape[-1] != 0:
                    repeat_time += 1
                out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :out_f]

        return out_x



class MoRALinear(nn.Module, MoRALayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_mora: bool = False,
        mora_type: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        MoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_mora=use_mora,
            mora_type=mora_type,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        result = self.base_layer(x,)
        torch_result_dtype = result.dtype

        lora_A = self.lora_A
        lora_B = self.lora_B
        dropout = self.lora_dropout
        scaling = self.scaling
        x = x.to(lora_A.weight.dtype)

        if self.use_mora:
            # x = dropout(x)
            # delta = self._apply_mora(x, lora_A, lora_B, scaling, active_adapter)
            # print(delta.abs().mean().item())
            # with open('mora.txt', 'w') as f:
            #     print(delta.abs().mean().item(), file=f)
            # result = result + delta

            x = dropout(x)
            result = result + self._apply_mora(x, lora_A, lora_B, scaling,)


        result = result.to(torch_result_dtype)
        return result
    
    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer
    
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """



        if self.lora_A:
            base_layer = self.get_base_layer()
            if safe_merge:
                # Note that safe_merge will be slower than the normal merge
                # because of the copy operation.
                orig_weights = base_layer.weight.data.clone()
                delta_weight = self.get_delta_weight()

                orig_weights += delta_weight
        

                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter seems to be broken"
                    )

                base_layer.weight.data = orig_weights
            else:
                delta_weight = self.get_delta_weight()

                base_layer.weight.data += delta_weight



    
    def get_delta_weight(self, ) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B.weight.device
        dtype = self.lora_B.weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A.weight
        weight_B = self.lora_B.weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        if self.use_mora:
            in_f, out_f = self.in_features, self.out_features
            r = self.r
            if in_f % r != 0:
                pad_size = r - in_f % r
            else:
                pad_size = 0
            repeat_time = out_f // r
            if out_f % r != 0:
                repeat_time += 1

            if self.mora_type is not None or self.mora_type == 1:
                w = torch.zeros(r, in_f).to(device, dtype=dtype)
                aw = weight_A
                for i in range(in_f + pad_size):
                    w[:, i % in_f] += aw[:, i % r]
                w = torch.cat([w]*repeat_time, dim=0)[:out_f]
            elif self.mora_type == 2:
                w = weight_A
                mr, nr = in_f//r+1, in_f//r
                m, n = in_f - r*nr, r*mr - in_f

                # mm, nn = m*mr, n * nr
                w = torch.cat([torch.repeat_interleave(w[:, :m], mr, dim=1),
                            torch.repeat_interleave(w[:, m:], nr, dim=1)], dim=1)

                mr, nr = out_f//r+1, out_f//r
                m, n = out_f - r*nr, r*mr - out_f
                # mm, nn = m*mr, n * nr
                w = torch.cat([torch.repeat_interleave(w[:m], mr, dim=0),
                            torch.repeat_interleave(w[m:], nr, dim=0)], dim=0)
            elif self.mora_type == 3:
                w = weight_A
                mr, nr = in_f//r+1, in_f//r
                m, n = in_f - r*nr, r*mr - in_f
                # mm, nn = m*mr, n * nr
                w = torch.cat([torch.repeat_interleave(w[:, :m], mr, dim=1),
                            torch.repeat_interleave(w[:, m:], nr, dim=1)], dim=1)

                w = torch.cat([w]*repeat_time, dim=0)[:out_f]
            elif self.mora_type == 4:
                w = torch.zeros(r, in_f).to(device, dtype=dtype)
                aw = weight_A
                for i in range(in_f + pad_size):
                    w[:, i % in_f] += aw[:, i % r]

                mr, nr = out_f//r+1, out_f//r
                m, n = out_f - r*nr, r*mr - out_f
                # mm, nn = m*mr, n * nr
                w = torch.cat([torch.repeat_interleave(w[:m], mr, dim=0),
                            torch.repeat_interleave(w[m:], nr, dim=0)], dim=0)
            elif self.mora_type == 6:
                w = torch.zeros(in_f+pad_size, in_f).to(device, dtype=dtype)
                rb1 = in_f//r if in_f % r == 0 else in_f//r + 1
                rb2 = out_f//r if out_f % r == 0 else out_f//r + 1
                sum_inter, repeat_time = rb1, rb2
                if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
                    inv_freq = 1.0 / (10000 ** (torch.arange(0, r, 2).float() / r))
                    t = torch.arange(rb1)
                    freqs = torch.outer(t, inv_freq)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    self.cos = emb.cos().unsqueeze(0).to(w.device).to(w.dtype)
                    self.sin = emb.sin().unsqueeze(0).to(w.device).to(w.dtype)
                cos, sin = self.cos, self.sin
                aw = weight_A
                aw2 = torch.cat((aw[:, r//2:], -aw[:, :r//2]), dim=-1)
                for i in range(sum_inter-1):
                    w[i*r:(i+1)*r, i*r:(i+1)*r] = aw2*sin[:, i] + aw*cos[:, i]
                i+=1
                w[i*r:, i*r:]  = (aw2*sin[:, i] + aw*cos[:, i])[:, :r-pad_size] #+ aw2*sin[:, i])[:, :r-pad_size]
                if pad_size > 0:
                    w[i*r:, :pad_size] = (aw2*sin[:, i] + aw*cos[:, i])[:, r-pad_size:]
                if in_f < out_f:
                    w = torch.cat([w]*repeat_time, dim=0)[:out_f]
                else:
                    w = w[:out_f]
            else:
                # old
                w = torch.zeros(r, in_f).to(device, dtype=dtype)
                aw = weight_A
                for i in range(in_f):
                    w[:, i % in_f] += aw[:, i % r]
                #w = torch.cat([w]*repeat_time, dim=0)[:out_f]
                w = torch.cat([torch.repeat_interleave(w, out_f//r, dim=0), w], dim=0)[:out_f]
            output_tensor = w
        else:
            output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A.weight.data = weight_A.to(dtype)
            self.lora_B.weight.data = weight_B.to(dtype)

        # print rank of output_tensor
        # print(f'rank: {torch.linalg.matrix_rank(output_tensor.float())}')
        return output_tensor