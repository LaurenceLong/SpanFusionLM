import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    包含 LayerNorm 后接一个线性投影（权重共享至 TokenEmbedding.embedding）。
    """
    def __init__(self, hidden_size, vocab_size, tied_embedding_weight=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        if tied_embedding_weight is not None:
            # 直接共享嵌入权重
            self.tied_weight = tied_embedding_weight
        else:
            self.tied_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
            nn.init.xavier_uniform_(self.tied_weight)

    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        x = self.layer_norm(x)
        # logits 形状：(B, seq_len, vocab_size)
        logits = torch.matmul(x, self.tied_weight.t())
        return logits
