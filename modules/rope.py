# SpanFusionLM/modules/rope.py
import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 初始化缓存
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device or torch.device('cpu'))

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        # 计算频率矩阵 (seq_len, dim//2)
        freqs = torch.outer(t, self.inv_freq.to(device))
        # 重复以匹配完整维度 (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None, position_ids=None):
        if position_ids is not None:
            max_pos = position_ids.max().item() + 1
            if max_pos > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=max_pos, device=x.device)

            # 根据position_ids选择对应的cos/sin值
            cos = self.cos_cached[position_ids]  # (B, seq_len, dim)
            sin = self.sin_cached[position_ids]  # (B, seq_len, dim)
        else:
            if seq_len is None:
                seq_len = x.shape[-2]

            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

            cos = self.cos_cached[:seq_len]  # (seq_len, dim)
            sin = self.sin_cached[:seq_len]  # (seq_len, dim)

            # 扩展维度以匹配输入
            cos = cos.unsqueeze(0)  # (1, seq_len, dim)
            sin = sin.unsqueeze(0)  # (1, seq_len, dim)

        # 确保维度匹配 (B, 1, seq_len, dim) for attention
        if cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    """将输入的后半部分维度旋转"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    # 确保cos和sin的形状正确
    if cos.ndim == 4 and q.ndim == 4:  # (B, 1, seq_len, dim)
        cos = cos.expand(-1, q.shape[1], -1, -1)  # (B, num_heads, seq_len, dim)
        sin = sin.expand(-1, q.shape[1], -1, -1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
