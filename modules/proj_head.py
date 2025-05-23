import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, tied_embedding_weight=None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        if tied_embedding_weight is not None:
            self.weight = tied_embedding_weight
        else:
            self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return torch.matmul(hidden_states, self.weight.T)
