# SpanFusionLM/modules/decoder.py
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
        # 分别计算两路全连接
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # 对 gate 输出使用激活函数
        activated = self.act_fn(gate)
        # 为了防止数值溢出或出现 NaN，对激活后的值和 up_proj 的输出进行数值稳定性处理
        activated = torch.nan_to_num(activated, nan=0.0, posinf=1e6, neginf=-1e6)
        up = torch.nan_to_num(up, nan=0.0, posinf=1e6, neginf=-1e6)
        hidden = activated * up
        hidden = torch.clamp(hidden, min=-1e6, max=1e6)
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.down_proj(hidden)


class LlamaAttention(nn.Module):
    def __init__(self, config):  # 传入完整配置
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,  # 由 SpanDecoder 构造的因果 mask
                position_ids: torch.Tensor,
                past_key_value=None):  # Tuple of (past_key, past_value)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        # 应用 RoPE（旋转位置编码）
        cos, sin = self.rotary_emb(
            value_states,  # x argument in RoPE, mainly for device/dtype. freqs determined by pos_ids/seq_len
            seq_len=position_ids.max().item() + 1 if position_ids.numel() > 0 else q_len,
            position_ids=position_ids
        )
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,  # 使用传入的因果 mask
            dropout_p=0.0  # 此处不添加 dropout
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):  # 传入完整配置
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_value=None):  # 当前层的过去KV (past_key, past_value)
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        attn_output, present_key_value = self.self_attn(
            hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        hidden_states = residual + attn_output  # add residual 前

        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        hidden_states = residual + mlp_output

        return hidden_states, present_key_value  # 返回当前隐藏状态与当前层 KV


class SpanDecoder(nn.Module):
    def __init__(self, config):  # 传入完整配置
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def _prepare_decoder_attention_mask(self, input_shape, dtype, device, past_kv_length):
        # input_shape 为 (batch_size, q_len)——当前输入 tokens 数量
        batch_size, q_len = input_shape
        # kv_seq_len 是 K和V的总长度 (past + current)
        kv_seq_len = past_kv_length + q_len

        # 构造一个大小为 (q_len, kv_seq_len) 的因果 mask
        # 对于 query pos i (0 to q_len-1),它可以关注 key pos j (0 to kv_seq_len-1)
        # 当 j <= past_kv_length + i 时
        mask = torch.full((q_len, kv_seq_len), float("-inf"), dtype=dtype, device=device)
        for i in range(q_len):
            # 当前 query token (index i) 可以关注所有 past_kv_tokens 以及到当前 token 为止的所有 current tokens
            mask[i, :(past_kv_length + i + 1)] = 0.0  # Corrected from 0 to 0.0 for float type mask

        # 扩展成 (bsz, 1, q_len, kv_seq_len) 以便 F.scaled_dot_product_attention 使用
        # F.sdpa 会自动广播 (1,1,q_len,kv_seq_len) 或 (B,1,q_len,kv_seq_len)
        return mask.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, q_len, kv_seq_len)

    def forward(self,
                hidden_states: torch.Tensor,  # (B, current_seq_len, hidden_size)
                attention_mask: torch.Tensor = None,  # 用户提供的 attention_mask (例如，padding mask). 通常对于因果LM为None.
                position_ids: torch.Tensor = None,  # (B, current_seq_len) - 绝对位置
                past_key_values=None,  # 包含各层 KV cache 的元组: ((k,v), (k,v), …)
                use_cache: bool = True):

        batch_size, current_seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        past_kv_length = 0
        if past_key_values is not None and \
                len(past_key_values) > 0 and \
                past_key_values[0] is not None and \
                past_key_values[0][0] is not None:  # 检查第一层的 key tensor
            # K_cache: (batch_size, num_heads, past_seq_len, head_dim)
            past_kv_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            # 如果未提供 position_ids, 则从 past_kv_length 开始创建它们
            position_ids = torch.arange(
                past_kv_length, current_seq_len + past_kv_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)  # Shape (1, current_seq_len) 以便广播

        # `LlamaAttention.forward` 中的 `attention_mask` 参数是最终传递给 `F.scaled_dot_product_attention` 的掩码。
        # 如果 `SpanDecoder.forward` 的输入 `attention_mask` 为 None, 我们需要构建因果掩码。
        if attention_mask is None:
            # `input_shape` for `_prepare_decoder_attention_mask` is (batch_size, query_length)
            # `query_length` is `current_seq_len` (新 tokens 的长度)
            # `past_kv_length` 是 cache 中 K/V 的长度
            # 生成的掩码将是 (1, 1, current_seq_len, past_kv_length + current_seq_len)
            # F.scaled_dot_product_attention 会将其广播到 (batch_size, num_heads, current_seq_len, past_kv_length + current_seq_len)
            final_attention_mask = self._prepare_decoder_attention_mask(
                (batch_size, current_seq_len),
                dtype=dtype,  # Should be float for -inf
                device=device,
                past_kv_length=past_kv_length
            )
        else:
            # 如果提供了特定的掩码 (例如，padding mask 与因果掩码的组合)，则使用它。
            # 它应该具有正确的形状：(batch_size, 1, current_seq_len, past_kv_length + current_seq_len) 或可广播的形状。
            final_attention_mask = attention_mask

        all_present_key_values = [] if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None and i < len(
                past_key_values) else None

            # LlamaDecoderLayer.forward 需要:
            # hidden_states, attention_mask, position_ids, past_key_value
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=final_attention_mask,  # 这是当前 token块 的因果掩码
                position_ids=position_ids,  # 当前 token块 的位置
                past_key_value=layer_past_key_value  # 该层上一步的 KV cache
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                all_present_key_values.append(layer_outputs[1])  # layer_outputs[1] 是 (present_key, present_value)

        hidden_states = self.norm(hidden_states)  # 最终的层归一化

        next_kv_cache = None
        if use_cache and all_present_key_values:  # Check if list is not empty
            next_kv_cache = tuple(all_present_key_values)

        return hidden_states, next_kv_cache