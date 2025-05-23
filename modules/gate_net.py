# SpanFusionLM/modules/gate_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, g_max=8): # input_dim could be K if taking full entropy
        super().__init__()
        self.g_max = g_max
        # Original pseudocode implies input is mean_entropy (B,1)
        # If input_dim is K, it would take entropy (B,K)
        # Let's assume input is (B, input_dim) where input_dim=1 for mean_entropy
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Takes mean_entropy (B,1) or pooled entropy
            nn.ReLU(),
            nn.Linear(hidden_dim, g_max) # Outputs logits for g_max choices
        )
        self.override_fn = None

    def override(self, override_fn):
        """
        Allows overriding the internal MLP logic to set specific g_hat values for inference.
        override_fn should take an entropy tensor (e.g., (B,K) or (B,1)) and return logits (B, g_max).
        """
        self.override_fn = override_fn

    def forward(self, entropy_feature: torch.Tensor, train: bool = True):
        # entropy_feature: (B, input_dim), e.g., (B,1) if mean_entropy, or (B,K) if processing full entropy
        
        if self.override_fn is not None:
            # The override_fn is responsible for producing appropriate logits
            # It might use the entropy_feature or ignore it.
            # The override_fn in infer.py is designed to take the original entropy (B,K)
            # but this GateNet is called with entropy_feature (e.g. mean_entropy (B,1))
            # For consistency, the override_fn should ideally operate on the same feature type
            # or be designed to accept the raw entropy and do its own feature extraction if needed.
            # Given current infer.py, override_fn expects to create logits (B, g_max)
            # and might ignore the passed entropy_feature if it's just for setting fixed g.
            # Let's assume override_fn can work with a dummy or the provided feature.
            logits = self.override_fn(entropy_feature) # Pass the feature it would normally get
        else:
            logits = self.mlp(entropy_feature)  # (B, g_max)
        
        if train:
            # Gumbel-Softmax for differentiable sampling during training
            g_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True)  # (B, g_max)
        else:
            # Argmax for deterministic prediction during inference/evaluation
            indices = logits.argmax(dim=-1) # (B,)
            g_one_hot = F.one_hot(indices, num_classes=self.g_max).float() # (B, g_max)
            
        # Calculate g_hat (actual number of steps, 1 to g_max)
        # gate_values are [1, 2, ..., g_max]
        gate_values = torch.arange(1, self.g_max + 1, device=logits.device, dtype=torch.float) # (g_max,)
        g_hat = (g_one_hot * gate_values).sum(dim=-1).long()  # (B,)
        
        return g_hat, logits # Return g_hat and the raw logits (for E[g] loss)
