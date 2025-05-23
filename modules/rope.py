# SpanFusionLM/modules/rope.py
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Precompute inv_freq
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Initialize cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device if device is None else device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype) # Use inv_freq.dtype for t
        
        # freqs = t @ self.inv_freq.T -> this would be (max_seq_len_cached, dim/2)
        # but torch.outer(t, self.inv_freq) is (max_seq_len_cached, dim/2)
        freqs = torch.outer(t, self.inv_freq) # (max_seq_len_cached, dim / 2)
        
        # Different from Llama, Pytorch Transformer, etc. where freqs is (max_seq_len, dim/2)
        # and emb = torch.cat((freqs, freqs), dim=-1) is (max_seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1) # (max_seq_len_cached, dim)
        
        if hasattr(self, 'cos_cached'): # Update existing buffers
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
        else: # Register new buffers
            self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None, position_ids=None):
        # x: (bsz, num_heads, seq_len, head_dim) or (bsz, seq_len, num_heads, head_dim)
        # or any tensor where the last dim is head_dim and second to last is seq_len related
        # For RoPE, we care about the sequence length dimension for fetching freqs
        # and the feature dimension (dim).
        
        # If position_ids are provided, use them directly.
        # position_ids: (bsz, seq_len_actual_tokens)
        # If not, assume a contiguous sequence up to seq_len.
        
        if position_ids is None:
            if seq_len is None: # Infer from one of the sequence dimensions of x
                # This is tricky as x can have various shapes.
                # Let's assume if position_ids is None, seq_len must be provided or x.shape[-2] is seq_len
                seq_len = x.shape[-2] if x.ndim > 1 else x.shape[0] # A guess
            actual_seq_len = seq_len
            current_positions = torch.arange(actual_seq_len, device=x.device)
        else:
            actual_seq_len = position_ids.max().item() + 1 # Max position ID determines required cache size
            current_positions = position_ids # (bsz, seq_len_actual_tokens)

        if actual_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=actual_seq_len, device=x.device, dtype=x.dtype)
        
        # Fetch cos and sin based on current_positions
        # cos_cached/sin_cached are (max_seq_len_cached, dim)
        # current_positions can be (seq_len_actual) or (bsz, seq_len_actual)
        cos = self.cos_cached[current_positions] # (seq_len_actual, dim) or (bsz, seq_len_actual, dim)
        sin = self.sin_cached[current_positions] # (seq_len_actual, dim) or (bsz, seq_len_actual, dim)

        # Reshape cos/sin to be broadcastable with x: (bsz, num_heads, seq_len, head_dim)
        # Typically, RoPE is applied after q/k are shaped to (bsz, num_heads, seq_len, head_dim)
        # So, cos/sin should be (bsz or 1, 1 or num_heads, seq_len_actual, dim or head_dim)
        # If current_positions was (seq_len_actual), cos/sin are (seq_len_actual, dim)
        # We need it like (1, 1, seq_len_actual, dim) for broadcasting with (B, H, S, D)
        # Or if current_positions was (B, S), cos/sin are (B, S, D), need (B, 1, S, D)
        
        if cos.ndim == 2: # (seq_len_actual, dim) from arange
            cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len_actual, dim)
            sin = sin.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len_actual, dim)
        elif cos.ndim == 3: # (bsz, seq_len_actual, dim) from position_ids (B,S)
            cos = cos.unsqueeze(1) # (bsz, 1, seq_len_actual, dim)
            sin = sin.unsqueeze(1) # (bsz, 1, seq_len_actual, dim)
        
        # Ensure dtype matches x
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., 0 : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 : x.shape[-1]]
    # return torch.cat((-x2, x1), dim=-1)
    # More robust slicing for various tensor dimensions:
    x_part1 = x[..., : x.shape[-1] // 2]
    x_part2 = x[..., x.shape[-1] // 2 :]
    return torch.cat(((-1) * x_part2, x_part1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (bsz, num_heads, seq_len, head_dim)
    # cos, sin: (bsz or 1, 1 or num_heads, seq_len, head_dim) - ensure broadcasting
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

