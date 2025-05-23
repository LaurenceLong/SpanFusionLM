# SpanFusionLM/modules/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .decoder import LlamaMLP

class SpanEncoderSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

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
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor = None):
        bsz, k_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, k_len, bsz)
        key_states = self._shape(key_states, k_len, bsz)
        value_states = self._shape(value_states, k_len, bsz)

        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False  # Encoder使用双向注意力
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, k_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class SpanEncoderCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # 初始化权重
        self._init_weights()

        self.rotary_emb_q = RotaryEmbedding(
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
                encoder_hidden_states: torch.Tensor,
                position_ids_q: torch.Tensor,
                attention_mask: torch.Tensor = None):
        bsz_q, k_len_q, _ = hidden_states.size()
        bsz_kv, k_len_kv, _ = encoder_hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = self._shape(query_states, k_len_q, bsz_q)
        key_states = self._shape(key_states, k_len_kv, bsz_kv)
        value_states = self._shape(value_states, k_len_kv, bsz_kv)

        # 只对query应用RoPE
        cos_q, sin_q = self.rotary_emb_q(query_states, position_ids=position_ids_q)
        query_states = (query_states * cos_q) + (rotate_half(query_states) * sin_q)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz_q, k_len_q, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class SpanEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SpanEncoderSelfAttention(config)
        self.cross_attn = SpanEncoderCrossAttention(config)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.post_self_attn_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.post_cross_attn_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self,
                latent_z: torch.Tensor,
                h_dec_k_positions: torch.Tensor,
                global_position_ids_k: torch.Tensor,
                self_attn_mask: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None):

        # Self-Attention
        residual = latent_z
        z_norm = self.input_layernorm(latent_z)
        z_self_attn_out = self.self_attn(
            z_norm,
            position_ids=global_position_ids_k,
            attention_mask=self_attn_mask
        )
        latent_z = residual + z_self_attn_out

        # Cross-Attention
        residual = latent_z
        z_norm = self.post_self_attn_layernorm(latent_z)
        z_cross_attn_out = self.cross_attn(
            z_norm,
            encoder_hidden_states=h_dec_k_positions,
            position_ids_q=global_position_ids_k,
            attention_mask=cross_attn_mask
        )
        latent_z = residual + z_cross_attn_out

        # MLP
        residual = latent_z
        z_norm = self.post_cross_attn_layernorm(latent_z)
        mlp_out = self.mlp(z_norm)
        latent_z = residual + mlp_out

        return latent_z

class SpanEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SpanEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])

    def step(self,
             latent_z: torch.Tensor,
             h_dec_k_positions: torch.Tensor,
             prompt_len: int):
        """
        执行一步latent refinement
        """
        bsz, K, d_model = latent_z.shape
        device = latent_z.device

        # 创建全局位置ID
        k_indices = torch.arange(K, device=device)
        global_position_ids_k = prompt_len + k_indices.unsqueeze(0).expand(bsz, K)

        # 注意力掩码（encoder使用双向注意力，所以都是False）
        self_attn_mask = None  # 双向自注意力不需要掩码
        cross_attn_mask = None  # 交叉注意力不需要掩码

        # 通过所有encoder层
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
