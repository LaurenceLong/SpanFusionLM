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
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, config): # Pass full config
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
                attention_mask: torch.Tensor, # This is the causal mask prepared by SpanDecoder
                position_ids: torch.Tensor,
                past_key_value=None): # Tuple of (past_key, past_value)
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        # RoPE application
        # position_ids are (bsz, q_len)
        # rotary_emb needs seq_len for cache, but uses position_ids for actual freqs
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1 if position_ids.numel() > 0 else q_len, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # Current key and value for caching in this pass
        present_key_value = (key_states, value_states)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask, # Use the passed causal mask
            dropout_p=0.0 # No dropout during attention itself if not specified
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config): # Pass full config
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_value=None): # Tuple of (past_key, past_value) for this layer
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)
        
        attn_output, present_key_value = self.self_attn(
            hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        hidden_states = residual + attn_output # Add back residual before second norm

        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        hidden_states = residual + mlp_output
        
        return hidden_states, present_key_value # Return hidden states and KV for this layer

class SpanDecoder(nn.Module):
    def __init__(self, config): # Pass full config
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def _prepare_decoder_attention_mask(self, input_shape, dtype, device, past_kv_length):
        # input_shape is (batch_size, seq_len) of current input tokens
        # past_kv_length is the length of sequences in past_key_values
        batch_size, q_len = input_shape
        
        # Total length of keys/values will be past_kv_length + q_len
        kv_seq_len = past_kv_length + q_len
        
        # Create a causal mask of shape (q_len, kv_seq_len)
        # Mask value of True indicates masking, False allows attention
        # F.scaled_dot_product_attention expects:
        # -inf for positions to be masked, 0 for allowed positions.
        mask = torch.full((q_len, kv_seq_len), float("-inf"), dtype=dtype, device=device)
        
        # Allow attention to all past keys and current/past tokens in current input
        for i in range(q_len):
            mask[i, :(past_kv_length + i + 1)] = 0
            
        # Expand to (bsz, 1, q_len, kv_seq_len) for multi-head attention
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_seq_len)


    def forward(self,
                hidden_states: torch.Tensor,  # (B, current_seq_len, hidden_size)
                attention_mask: torch.Tensor = None, # Optional external mask
                position_ids: torch.Tensor = None,   # (B, current_seq_len)
                past_key_values=None, # Tuple of tuples, one for each layer: ((k,v), (k,v), ...)
                use_cache: bool = True):
        
        batch_size, current_seq_len, _ = hidden_states.shape
        
        # Determine past_kv_length from past_key_values if provided
        past_kv_length = 0
        if past_key_values is not None and past_key_values[0] is not None and past_key_values[0][0] is not None:
            past_kv_length = past_key_values[0][0].shape[2] # k_cache is (B, num_heads, seq_len_cached, head_dim)

        # If position_ids are not provided, create them starting from past_kv_length
        if position_ids is None:
            position_ids = torch.arange(
                past_kv_length, 
                current_seq_len + past_kv_length,
                dtype=torch.long, 
                device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # If an explicit attention_mask is not provided, create the causal one
        if attention_mask is None:
            attention_mask = self._prepare_decoder_attention_mask(
                (batch_size, current_seq_len), 
                hidden_states.dtype, 
                hidden_states.device, 
                past_kv_length
            )
        # Else, the provided attention_mask should be correctly shaped (B, 1, Q_len, KV_len)

        output_hidden_states = hidden_states
        
        next_kv_cache_list = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            
            output_hidden_states, layer_present_kv = layer(
                output_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_kv
            )
            if use_cache:
                next_kv_cache_list.append(layer_present_kv)
        
        output_hidden_states = self.norm(output_hidden_states)
        
        final_kv_cache = tuple(next_kv_cache_list) if use_cache and next_kv_cache_list else None
        
        return output_hidden_states, final_kv_cache
