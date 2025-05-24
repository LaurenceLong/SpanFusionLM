# modules/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary_pos_emb


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for module in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SiLU激活函数
        gate = F.silu(gate)

        # 数值稳定性处理
        gate = torch.clamp(gate, min=-10, max=10)
        up = torch.clamp(up, min=-10, max=10)

        hidden = gate * up
        hidden = torch.clamp(hidden, min=-10, max=10)

        return self.down_proj(hidden)

class LlamaAttention(nn.Module):
    def __init__(self, config):
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

        # 初始化权重
        self._init_weights()

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                past_key_value=None):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        # 应用RoPE
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 处理KV缓存
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states)

        # 注意力计算
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None)  # 只有在没有显式mask时使用因果mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                past_key_value=None):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_key_value

class SpanDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def _prepare_causal_attention_mask(self, input_shape, past_kv_length, device, dtype):
        """准备因果注意力掩码"""
        batch_size, seq_length = input_shape
        mask = torch.full((seq_length, seq_length), float("-inf"), device=device, dtype=dtype)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        if past_kv_length > 0:
            # 对于增量解码，需要扩展mask
            mask = torch.cat([torch.zeros(seq_length, past_kv_length, device=device, dtype=dtype), mask], dim=-1)

        return mask[None, None, :, :].expand(batch_size, 1, seq_length, -1)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                padding_mask: torch.Tensor = None,
                past_key_values=None,
                use_cache: bool = True):

        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device

        # 计算past_kv_length
        past_kv_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            if past_key_values[0] is not None and past_key_values[0][0] is not None:
                past_kv_length = past_key_values[0][0].shape[2]

        # 生成position_ids
        if position_ids is None:
            position_ids = torch.arange(
                past_kv_length, past_kv_length + seq_length,
                dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # 准备注意力掩码
        if attention_mask is None:
            attention_mask = self._prepare_causal_attention_mask(
                (batch_size, seq_length),
                past_kv_length,
                device,
                hidden_states.dtype
            )
        if padding_mask is not None:
            if padding_mask.ndim != 2 or padding_mask.shape[0] != batch_size:
                raise ValueError("Provided padding_mask must be 2D (batch_size, kv_sequence_length)")

            # The key/value sequence length for the current forward pass
            # If past_key_values are used, the full key/value sequence length includes past_kv_length
            current_kv_seq_len = past_kv_length + seq_length

            if padding_mask.shape[1] != current_kv_seq_len:
                # This can happen if padding_mask only covers current tokens, not past ones.
                # Assuming padding_mask covers the full effective sequence length (past + current)
                # If model.py passes padding_mask for `seq` (length `seq_length` + `past_kv_length`), then this is fine.
                # If model.py passes padding_mask for current `hidden_states` (length `seq_length`), then:
                if padding_mask.shape[1] == seq_length and past_kv_length > 0:
                    # Prepend 'not padded' for past keys/values, assuming they were valid
                    past_padding = torch.zeros((batch_size, past_kv_length), dtype=torch.bool, device=device)
                    padding_mask = torch.cat((past_padding, padding_mask), dim=1)
                elif padding_mask.shape[1] != current_kv_seq_len:
                    raise ValueError(
                        f"Padding mask seq len {padding_mask.shape[1]} doesn't match "
                        f"KV seq len {current_kv_seq_len}"
                    )

            # Expand the 2D padding mask (True for padded) to 4D
            # Masked positions (True in padding_mask) should be -inf in final_attention_mask
            expanded_padding_mask = padding_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, current_kv_seq_len
            )
            attention_mask = attention_mask.masked_fill(
                expanded_padding_mask,  # Where True (is a pad token)
                float("-inf")
            )
        # 通过所有层
        all_present_key_values = [] if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_key_value
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                all_present_key_values.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        next_cache = tuple(all_present_key_values) if use_cache else None
        return hidden_states, next_cache
