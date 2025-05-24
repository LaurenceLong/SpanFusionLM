# modules/gate_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import LlamaMLP

class GateNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            LlamaMLP(hidden_dim, intermediate_size=4 * hidden_dim),
            nn.Linear(hidden_dim, 2)  # 二分类输出：0 表示停止，1 表示继续
        )
        self._override_fn = None

    def override(self, override_fn):
        """
        在推理时，注入一个 override_fn(entropy_tensor) -> gate_logits
        """
        self._override_fn = override_fn

    def forward(self, entropy_feature: torch.Tensor, train: bool = True):
        # 如果设置了 override，就直接使用覆盖函数
        if self._override_fn is not None:
            gate_logits = self._override_fn(entropy_feature)
            decision = gate_logits.argmax(dim=-1)
            return decision, gate_logits

        logits = self.mlp(entropy_feature)

        if train:
            # 混合精度下，在 fp32 下计算 Gumbel-Softmax 以确保数值稳定性
            with torch.amp.autocast(enabled=False, device_type=logits.device.type):
                decision_one_hot = F.gumbel_softmax(logits.float(), tau=0.5, hard=True)
            decision_one_hot = decision_one_hot.to(logits.dtype)
        else:
            decisions = logits.argmax(dim=-1)
            decision_one_hot = F.one_hot(decisions, num_classes=2).float()

        decision = decision_one_hot.argmax(dim=-1)
        return decision, logits
