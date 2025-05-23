import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .decoder import LlamaMLP  # 复用 MLP

class SpanEncoderSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_theta=10000.0, max_position_embeddings=2048+32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

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
                hidden_states: torch.Tensor,  # (B, K, d)
                position_ids: torch.Tensor,   # (B, K)
                attention_mask: torch.Tensor = None):
        bsz, k_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, k_len, bsz)
        key_states = self._shape(key_states, k_len, bsz)
        value_states = self._shape(value_states, k_len, bsz)

        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, k_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class SpanEncoderCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_theta=10000.0, max_position_embeddings=2048+32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

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
                hidden_states: torch.Tensor,       # latent z (B, K, d)
                encoder_hidden_states: torch.Tensor,  # h_dec (B, K, d)
                position_ids_q: torch.Tensor,           # (B, K)
                attention_mask: torch.Tensor = None):
        bsz_q, k_len_q, _ = hidden_states.size()
        bsz_kv, k_len_kv, _ = encoder_hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = self._shape(query_states, k_len_q, bsz_q)
        key_states = self._shape(key_states, k_len_kv, bsz_kv)
        value_states = self._shape(value_states, k_len_kv, bsz_kv)

        cos, sin = self.rotary_emb(query_states, seq_len=position_ids_q.max().item() + 1, position_ids=position_ids_q)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz_q, k_len_q, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class SpanEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, rope_theta=10000.0, max_position_embeddings=2048+32):
        super().__init__()
        self.self_attn = SpanEncoderSelfAttention(hidden_size, num_heads,
                                                    rope_theta=rope_theta,
                                                    max_position_embeddings=max_position_embeddings)
        self.cross_attn = SpanEncoderCrossAttention(hidden_size, num_heads,
                                                      rope_theta=rope_theta,
                                                      max_position_embeddings=max_position_embeddings)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_self_attn_layernorm = nn.LayerNorm(hidden_size)
        self.post_cross_attn_layernorm = nn.LayerNorm(hidden_size)

    def forward(self,
                latent_z: torch.Tensor,          # (B, K, d)
                h_dec_k_positions: torch.Tensor, # (B, K, d)
                global_position_ids_k: torch.Tensor,  # (B, K)
                self_attn_mask: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None):
        # Self-Attention
        residual = latent_z
        z_norm = self.input_layernorm(latent_z)
        z_self = self.self_attn(z_norm, position_ids=global_position_ids_k, attention_mask=self_attn_mask)
        latent_z = residual + z_self

        # Cross-Attention
        residual = latent_z
        z_norm = self.post_self_attn_layernorm(latent_z)
        z_cross = self.cross_attn(z_norm,
                                  encoder_hidden_states=h_dec_k_positions,
                                  position_ids_q=global_position_ids_k,
                                  attention_mask=cross_attn_mask)
        latent_z = residual + z_cross

        # MLP
        residual = latent_z
        z_norm = self.post_cross_attn_layernorm(latent_z)
        latent_z = residual + self.mlp(z_norm)
        return latent_z

class SpanEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SpanEncoderLayer(config.hidden_size,
                             config.num_attention_heads,
                             config.intermediate_size,
                             rope_theta=config.rope_theta,
                             max_position_embeddings=config.max_position_embeddings)
            for _ in range(config.num_encoder_layers)
        ])

    def step(self,
             latent_z: torch.Tensor,          # (B, K, d)
             h_dec_k_positions: torch.Tensor, # (B, K, d)
             prompt_len: int):
        """
        对全部 K 个槽位执行一次 refinement，将 prompt_len 与 span 内偏移组合计算全局位置。
        """
        bsz, K, _ = latent_z.shape
        k_idx = torch.arange(K, device=latent_z.device)
        global_position_ids_k = (prompt_len + k_idx).unsqueeze(0).expand(bsz, K)
        # 构造全 0 mask（bidirectional attention，不屏蔽任何位置）
        self_attn_mask = torch.zeros(bsz, 1, K, K, dtype=torch.bool, device=latent_z.device)
        cross_attn_mask = torch.zeros(bsz, 1, K, K, dtype=torch.bool, device=latent_z.device)

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
