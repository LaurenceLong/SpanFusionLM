import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    """
    GateNet 用于根据输入的 span 部分熵值动态决定 encoder refinement 的步数 g，
    为每个样本预测一个离散步数 ĝ ∈ {1, ..., g_max}。
    在训练时使用 Gumbel-Softmax 采样，在推理时使用 argmax。
    同时支持 override 以控制推理时 ĝ 的取值。
    """
    def __init__(self, hidden_dim=16, g_max=8):
        super().__init__()
        self.g_max = g_max
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, g_max)
        )
        self.override_fn = None

    def override(self, override_fn):
        self.override_fn = override_fn

    def forward(self, entropy, train=True):
        # entropy: (B, K)
        avg_entropy = entropy.mean(dim=1, keepdim=True)  # (B, 1)
        logits = self.mlp(avg_entropy)  # (B, g_max)
        if self.override_fn is not None:
            logits = self.override_fn(entropy)
        if train:
            g_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True)  # (B, g_max)
        else:
            indices = logits.argmax(dim=-1)
            g_one_hot = torch.nn.functional.one_hot(indices, num_classes=self.g_max).float()
        gate_values = torch.arange(1, self.g_max + 1, device=entropy.device).float()  # (g_max,)
        g_hat = (g_one_hot * gate_values).sum(dim=-1).long()  # (B,)
        return g_hat, logits
