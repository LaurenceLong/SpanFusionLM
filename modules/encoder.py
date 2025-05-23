# SpanFusionLM/modules/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .decoder import LlamaMLP # Reuse MLP from decoder

class SpanEncoderSelfAttention(nn.Module):
    def __init__(self, config): # Pass full config
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings, # Encoder might see global positions
            base=config.rope_theta
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor,  # (B, K, d) - This is latent_z
                position_ids: torch.Tensor,   # (B, K) - These are global positions for z
                attention_mask: torch.Tensor = None): # (B, 1, K, K) - Should be all False for bidirectional
        bsz, k_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, k_len, bsz)
        key_states = self._shape(key_states, k_len, bsz)
        value_states = self._shape(value_states, k_len, bsz)

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1 if position_ids.numel() > 0 else k_len, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # For bidirectional self-attention, attention_mask should allow all connections.
        # If mask is None or all False, F.scaled_dot_product_attention handles it.
        # A mask of 0s (or float -inf where masked) is expected by F.sdpa.
        # If bool, True means masked. So a mask of all False is correct.
        
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask, # Expects True for masked positions
            dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, k_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class SpanEncoderCrossAttention(nn.Module):
    def __init__(self, config): # Pass full config
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # K and V come from h_dec, which has the same hidden_size
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE for query (latent_z) based on its global positions
        self.rotary_emb_q = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        # K/V from h_dec already have their positions encoded from the decoder pass.
        # No separate RoPE for K/V here.

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor,       # latent_z (B, K_q, d)
                encoder_hidden_states: torch.Tensor,  # h_dec_k_positions (B, K_kv, d)
                position_ids_q: torch.Tensor,       # Global positions for latent_z (B, K_q)
                attention_mask: torch.Tensor = None): # (B, 1, K_q, K_kv) - Should be all False
        bsz_q, k_len_q, _ = hidden_states.size()
        bsz_kv, k_len_kv, _ = encoder_hidden_states.size() # K_q and K_kv should be same (K)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = self._shape(query_states, k_len_q, bsz_q)
        key_states = self._shape(key_states, k_len_kv, bsz_kv)
        value_states = self._shape(value_states, k_len_kv, bsz_kv)

        # Apply RoPE to query_states using its global positions
        cos_q, sin_q = self.rotary_emb_q(query_states, seq_len=position_ids_q.max().item() + 1 if position_ids_q.numel() > 0 else k_len_q, position_ids=position_ids_q)
        # Apply RoPE only to query, not key, as key comes from decoder which is already positionally encoded.
        query_states = (query_states * cos_q) + (rotate_half(query_states) * sin_q)
        # key_states are not rotated again here.

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask, # Expects True for masked positions
            dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz_q, k_len_q, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class SpanEncoderLayer(nn.Module):
    def __init__(self, config): # Pass full config
        super().__init__()
        self.self_attn = SpanEncoderSelfAttention(config)
        self.cross_attn = SpanEncoderCrossAttention(config)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = nn.LayerNorm(config.hidden_size) # For self-attention input
        self.post_self_attn_layernorm = nn.LayerNorm(config.hidden_size) # For cross-attention input
        self.post_cross_attn_layernorm = nn.LayerNorm(config.hidden_size) # For MLP input

    def forward(self,
                latent_z: torch.Tensor,          # (B, K, d)
                h_dec_k_positions: torch.Tensor, # (B, K, d) - from decoder, for cross-attn K/V
                global_position_ids_k: torch.Tensor,  # (B, K) - global positions for latent_z
                self_attn_mask: torch.Tensor = None, # (B, 1, K, K)
                cross_attn_mask: torch.Tensor = None): # (B, 1, K, K)
        
        # Self-Attention block for latent_z
        residual = latent_z
        z_norm = self.input_layernorm(latent_z)
        z_self_attn_out = self.self_attn(
            z_norm, 
            position_ids=global_position_ids_k, 
            attention_mask=self_attn_mask
        )
        latent_z = residual + z_self_attn_out

        # Cross-Attention block (latent_z queries h_dec_k_positions)
        residual = latent_z
        z_norm = self.post_self_attn_layernorm(latent_z)
        z_cross_attn_out = self.cross_attn(
            z_norm, # Query from latent_z
            encoder_hidden_states=h_dec_k_positions, # Key/Value from decoder hidden states
            position_ids_q=global_position_ids_k, # Positions for the query
            attention_mask=cross_attn_mask
        )
        latent_z = residual + z_cross_attn_out
        
        # MLP block
        residual = latent_z
        z_norm = self.post_cross_attn_layernorm(latent_z)
        mlp_out = self.mlp(z_norm)
        latent_z = residual + mlp_out
        
        return latent_z

class SpanEncoder(nn.Module):
    def __init__(self, config): # Pass full config
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SpanEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])
        # No final norm here, typically applied before projection head if needed,
        # or each layer has post-LN. ProjHead has its own LN.

    def step(self,
             latent_z: torch.Tensor,          # (B, K, d)
             h_dec_k_positions: torch.Tensor, # (B, K, d) - hidden states from decoder for K PRED positions
             prompt_len: int):                # Scalar, length of the prompt part
        """
        Performs one refinement step of the latent variable z.
        RoPE is applied within each attention layer using global positions.
        """
        bsz, K, d_model = latent_z.shape
        device = latent_z.device

        # Create global position IDs for the K tokens in the span
        # These are relative to the start of the entire sequence.
        k_indices_relative_to_span_start = torch.arange(K, device=device) # (K,)
        # global_position_ids_k = prompt_len + k_indices_relative_to_span_start # This would broadcast incorrectly
        global_position_ids_k = prompt_len + k_indices_relative_to_span_start.unsqueeze(0).expand(bsz, K) # (B, K)

        # Attention masks for encoder:
        # Self-attention is bidirectional within the K tokens. Mask is all False (attend all).
        # (B, num_heads, K, K) or (B, 1, K, K)
        # F.sdpa expects True for masked. So a mask of all False is correct.
        self_attn_mask = torch.zeros(bsz, 1, K, K, dtype=torch.bool, device=device)
        
        # Cross-attention: latent_z (queries, length K) attends to h_dec_k_positions (keys/values, length K).
        # Mask is all False (attend all).
        cross_attn_mask = torch.zeros(bsz, 1, K, K, dtype=torch.bool, device=device) # Q_len=K, KV_len=K

        output_z = latent_z
        for layer in self.layers:
            output_z = layer(
                output_z,
                h_dec_k_positions,
                global_position_ids_k,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask
            )
        return output_z
