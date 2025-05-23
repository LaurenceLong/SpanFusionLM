# SpanFusionLM/modules/gate_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, g_max=8):
        super().__init__()
        self.g_max = g_max
        self.hidden_dim = hidden_dim

        # 更深的网络以更好地预测步数
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, g_max)
        )
        self.override_fn = None

    def override(self, override_fn):
        """允许在推理时覆盖GateNet的行为"""
        self.override_fn = override_fn

    def forward(self, entropy_feature: torch.Tensor, train: bool = True):
        """
        entropy_feature: (B, input_dim) - 通常是平均熵 (B, 1)
        返回: g_hat (B,), logits (B, g_max)
        """
        if self.override_fn is not None:
            logits = self.override_fn(entropy_feature)
        else:
            logits = self.mlp(entropy_feature)

        # 数值稳定性处理
        logits = torch.clamp(logits, min=-10, max=10)

        if train:
            # 改进的Gumbel-Softmax采样
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = torch.clamp(gumbels, min=-10, max=10)

            logits_with_gumbel = (logits + gumbels) / 1.0  # tau=1.0
            g_one_hot = F.softmax(logits_with_gumbel, dim=-1)

            # 使用straight-through estimator
            g_indices = g_one_hot.argmax(dim=-1)
            g_one_hot_hard = F.one_hot(g_indices, num_classes=self.g_max).float()
            g_one_hot = g_one_hot_hard - g_one_hot.detach() + g_one_hot
        else:
            # 推理时使用确定性选择
            g_indices = logits.argmax(dim=-1)
            g_one_hot = F.one_hot(g_indices, num_classes=self.g_max).float()

        # 计算实际步数 (1 to g_max)
        gate_values = torch.arange(1, self.g_max + 1, device=logits.device, dtype=torch.float)
        g_hat = (g_one_hot * gate_values).sum(dim=-1).long()

        return g_hat, logits
