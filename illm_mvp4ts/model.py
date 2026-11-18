"""Equations (2)-(7); channel tokens, full-vocabulary alignment and attention LoRA.

This newly assembled model is distinct from the historical TimeLLM patch model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRAProjection(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        # Transformers GPT-2 Conv1D stores [in,out]; nn.Linear stores [out,in].
        if isinstance(base, nn.Linear):
            n_in, n_out = base.in_features, base.out_features
        elif base.__class__.__name__ == "Conv1D":
            n_in, n_out = base.weight.shape
        else:
            raise TypeError("Unsupported LoRA projection.")
        self.A = nn.Parameter(base.weight.new_empty(rank, n_in))
        self.B = nn.Parameter(base.weight.new_zeros(n_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def attach_lora(backbone, rank=16):
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("LoRA rank must be a nonnegative integer.")
    backbone.requires_grad_(False)
    if rank == 0:
        return []
    selected = []
    for name, module in list(backbone.named_modules()):
        is_qwen = ".self_attn." in name and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
        is_gpt2 = ".attn." in name and name.rsplit(".", 1)[-1] in {"c_attn", "c_proj"}
        if is_qwen or is_gpt2:
            parent, leaf = name.rsplit(".", 1)
            setattr(backbone.get_submodule(parent), leaf, LoRAProjection(module, rank, 2 * rank))
            selected.append(name)
    if not selected:
        raise ValueError("No supported attention projections found; use a Qwen2 or GPT-2 backbone.")
    return selected


class ForecastModel(nn.Module):
    def __init__(self, backbone, seq_len, n_variables, target_index, horizon=1, heads=4, lora_rank=16):
        super().__init__()
        self.backbone = backbone
        self.adapter_modules = attach_lora(backbone, lora_rank)
        self.seq_len, self.n_variables = seq_len, n_variables
        self.target_index, self.horizon = target_index, horizon
        width = backbone.config.hidden_size
        if width % heads or not 0 <= target_index < n_variables:
            raise ValueError("Hidden size must divide heads; target index must exist.")
        self.heads = heads
        self.time_projection = nn.Linear(seq_len, width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output_projection = nn.Linear(width, width)
        self.head = nn.Linear(width, horizon)

    def align(self, time_tokens):
        b, c, width = time_tokens.shape
        words = self.backbone.get_input_embeddings().weight.detach()
        q = self.query(time_tokens).view(b, c, self.heads, width // self.heads).transpose(1, 2)
        k = self.key(words).view(1, -1, self.heads, width // self.heads).transpose(1, 2)
        v = self.value(words).view(1, -1, self.heads, width // self.heads).transpose(1, 2)
        aligned = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.output_projection(aligned.transpose(1, 2).reshape(b, c, width))

    def forward(self, history, input_ids, attention_mask):
        if history.shape[1:] != (self.seq_len, self.n_variables):
            raise ValueError("History has incompatible time/variable dimensions.")
        mean = history.mean(dim=1, keepdim=True).detach()
        std = history.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
        normalized = (history - mean) / std
        tokens = self.time_projection(normalized.transpose(1, 2))
        tokens, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        aligned = self.align(tokens)
        prompt = self.backbone.get_input_embeddings()(input_ids)
        inputs = torch.cat([prompt, aligned], dim=1)
        mask = torch.cat([attention_mask, attention_mask.new_ones(history.size(0), self.n_variables)], dim=1)
        capacity = getattr(self.backbone.config, "max_position_embeddings", None)
        if capacity is not None and inputs.size(1) > capacity:
            raise ValueError("Prompt plus variable tokens exceeds backbone context capacity.")
        # Padding positions do not shift the positions of process-variable tokens.
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.backbone(inputs_embeds=inputs, attention_mask=mask, position_ids=positions,
                               use_cache=False, return_dict=True).last_hidden_state
        # MHSA has already mixed all variables; use designated target token for regression.
        target = hidden[:, -self.n_variables + self.target_index, :]
        predicted = self.head(target)
        return predicted * std[:, 0, self.target_index, None] + mean[:, 0, self.target_index, None]


def load_backbone(name, revision=None, local_files_only=False):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an EOS or PAD token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    backbone = AutoModel.from_pretrained(name, revision=revision, local_files_only=local_files_only)
    return backbone, tokenizer
