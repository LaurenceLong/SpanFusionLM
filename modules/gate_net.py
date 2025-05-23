import torch
import torch.nn as nn

class GateNet(nn.Module):
    """
    GateNet 用于根据输入的 span 部分熵值动态决定 encoder refinement 的步数 g，值域在 [1, g_max]。
    该模块首先对每个样本计算平均熵，然后经过一个小型 MLP，经过 sigmoid 映射后再缩放到 [1, g_max]，并取整数。
    """
    def __init__(self, hidden_dim=16, output_range=(1, 8)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_range = output_range
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, entropy):
        # entropy: (B, K)
        # 对每个样本计算平均熵，shape: (B, 1)
        avg_entropy = entropy.mean(dim=1, keepdim=True)
        raw = self.mlp(avg_entropy)  # (B,1)
        gated = torch.sigmoid(raw)   # 取值范围 (0,1)
        min_val, max_val = self.output_range
        g_float = gated * (max_val - min_val) + min_val
        g_int = g_float.round().clamp(min=min_val, max=max_val).int()  # (B,1)
        # 简化起见，取 batch 内平均作为最终 g 值（实际应用可以每样本不同）
        g_scalar = int(g_int.float().mean().round().item())
        return g_scalar
