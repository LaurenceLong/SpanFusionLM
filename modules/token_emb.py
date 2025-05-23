# SpanFusionLM/modules/token_emb.py
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size # Often models scale embeddings by sqrt(hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, seq_len)
        embs = self.embedding(input_ids)
        # Optional: scale embedding (common in some Transformer variants)
        # embs = embs * (self.hidden_size**0.5) 
        return embs
