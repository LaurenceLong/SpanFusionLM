import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import RotaryEmbedding, apply_rotary_pos_emb

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_theta=10000.0, max_position_embeddings=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim,
                                            max_position_embeddings=max_position_embeddings,
                                            base=rope_theta)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_value=None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        past_key_value = (key_states, value_states)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, rope_theta=10000.0, max_position_embeddings=2048):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads, rope_theta=rope_theta,
                                         max_position_embeddings=max_position_embeddings)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, kv_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out
        return hidden_states, kv_cache

class SpanDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config.hidden_size,
                              config.num_attention_heads,
                              config.intermediate_size,
                              rope_theta=config.rope_theta,
                              max_position_embeddings=config.max_position_embeddings)
            for _ in range(config.num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def _prepare_decoder_attention_mask(self, input_shape, dtype, device, past_kv_length):
        B, seq_len = input_shape
        q_len = seq_len
        kv_len = past_kv_length + seq_len
        causal_mask = torch.full((q_len, kv_len), float("-inf"), dtype=dtype, device=device)
        for i in range(q_len):
            causal_mask[i, :past_kv_length + i + 1] = 0
        return causal_mask[None, None, :, :].expand(B, 1, q_len, kv_len)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                past_key_values=None,
                use_cache: bool = True):
        batch_size, seq_length, _ = hidden_states.shape
        past_kv_length = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_kv_length = past_key_values[0][0].shape[2]
        if position_ids is None:
            position_ids = torch.arange(past_kv_length, seq_length + past_kv_length,
                                        dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        if attention_mask is None:
            attention_mask = self._prepare_decoder_attention_mask(
                (batch_size, seq_length), hidden_states.dtype, hidden_states.device, past_kv_length
            )
        next_kv_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, layer_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv
            )
            if use_cache:
                next_kv_cache.append(layer_kv)
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(next_kv_cache)
        else:
            return hidden_states, None
