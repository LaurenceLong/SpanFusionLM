# SpanFusionLM/modules/gate_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, g_max=8):
        super().__init__()
        self.g_max = g_max  # 最大迭代步数（仅用于传递配置）
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # 二分类输出：0 表示停止，1 表示继续
        )
        self.override_fn = None

    def override(self, override_fn):
        """允许在推理时覆盖GateNet的行为"""
        self.override_fn = override_fn

    def forward(self, entropy_feature: torch.Tensor, train: bool = True):
        """
        entropy_feature: (B, input_dim) – 通常为平均熵 (B,1)
        返回:
          decision: (B,) ，0 表示提前停止，1 表示继续迭代；
          logits: (B,2) 未归一化logits
        """
        if self.override_fn is not None:
            logits = self.override_fn(entropy_feature)
        else:
            logits = self.mlp(entropy_feature)

        logits = torch.clamp(logits, min=-10, max=10)

        if train:
            # 使用 Gumbel Softmax 实现带硬采样的梯度传递
            decision_one_hot = F.gumbel_softmax(logits, tau=0.5, hard=True)
        else:
            decisions = logits.argmax(dim=-1)
            decision_one_hot = F.one_hot(decisions, num_classes=2).float()

        # 如果概率最大的位置索引为 1，则输出 1 表示继续，否则为 0 表示停止
        decision = decision_one_hot.argmax(dim=-1)
        return decision, logits
