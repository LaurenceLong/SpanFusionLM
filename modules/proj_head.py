# SpanFusionLM/modules/proj_head.py
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, tied_embedding_weight=None):
        super().__init__()
        # 设置 eps 参数以增强数值稳定性
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        if tied_embedding_weight is not None:
            # 使用共享的权重
            self.decoder_weight = tied_embedding_weight  # 共享token_embedding.weight
        else:
            # 若不共享则初始化一份新的权重
            self.decoder_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
            nn.init.xavier_uniform_(self.decoder_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, hidden_size) 或 (N, hidden_size)
        返回 logits: (B, seq_len, vocab_size) 或 (N, vocab_size)
        """
        # 先进行 layer norm 归一化
        x_norm = self.layer_norm(x)
        # 对归一化的结果进行 clamping，防止出现极端值
        x_norm = torch.clamp(x_norm, min=-1e6, max=1e6)
        # 计算 logits，即矩阵相乘（使用共享权重的转置）
        logits = torch.matmul(x_norm, self.decoder_weight.t())
        # 对 logits 同样进行 clamping
        logits = torch.clamp(logits, min=-1e6, max=1e6)
        # 将潜在的 NaN 值替换为 0，并将正无穷、负无穷换为有限值，保证后续 log_softmax 不会出现 NaN
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return logits