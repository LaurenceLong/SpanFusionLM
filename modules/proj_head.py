# SpanFusionLM/modules/proj_head.py
import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, tied_embedding_weight=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        if tied_embedding_weight is not None:
            # Share weights with token embedding
            self.decoder_weight = tied_embedding_weight # This is E or W from description
        else:
            # Initialize new weights if not tied (should ideally be tied)
            self.decoder_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
            nn.init.xavier_uniform_(self.decoder_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, hidden_size) or (N, hidden_size)
        x_norm = self.layer_norm(x)
        # Project to vocab size: (B, seq_len, V) or (N, V)
        # Using W^T, so matmul(x_norm, W.T) where W is (V, D)
        logits = torch.matmul(x_norm, self.decoder_weight.t())
        return logits
