# SpanFusionLM/model.py
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.decoder import SpanDecoder
from .modules.encoder import SpanEncoder
from .modules.gate_net import GateNet
from .modules.proj_head import ProjectionHead
from .modules.token_emb import TokenEmbedding
from .modules.tokenizer import build_tokenizer

def top_p_logits_processor(logits: torch.Tensor, top_p: float):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits

def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9):
    """改进的采样函数"""
    logits = logits / max(temperature, 1e-8)

    if top_p < 1.0:
        logits = top_p_logits_processor(logits, top_p=top_p)

    logits = torch.clamp(logits, min=-50, max=50)
    probs = torch.softmax(logits, dim=-1)

    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return torch.argmax(logits, dim=-1)

    try:
        if probs.ndim == 1:
            return torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze()
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    except:
        return torch.argmax(logits, dim=-1)

@dataclass
class SpanFusionLMConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_decoder_layers: int = 28
    num_encoder_layers: int = 8
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048 + 32
    rope_theta: float = 10000.0
    g_max: int = 8
    tokenizer_name: str = "gpt2"
    tokenizer: any = field(default=None, repr=False, compare=False)

    pred_token_id: int = None
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(self.tokenizer_name)

        if self.vocab_size is None or self.vocab_size == 32000:
            self.vocab_size = len(self.tokenizer)

        if hasattr(self.tokenizer, 'additional_special_tokens') and '[PRED]' in self.tokenizer.additional_special_tokens:
            self.pred_token_id = self.tokenizer.additional_special_tokens_ids[
                self.tokenizer.additional_special_tokens.index('[PRED]')
            ]

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def save_pretrained(self, save_directory: str):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config_file = path / "config.json"

        config_dict = self.__dict__.copy()
        if 'tokenizer' in config_dict:
            del config_dict['tokenizer']

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        config_file = Path(model_name_or_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file 'config.json' not found in {model_name_or_path}")

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        config_dict.update(kwargs)
        return cls(**config_dict)

class SpanFusionLM(nn.Module):
    def __init__(self, config: SpanFusionLMConfig):
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.decoder = SpanDecoder(config)
        self.encoder = SpanEncoder(config)
        self.proj_head = ProjectionHead(
            config.hidden_size,
            config.vocab_size,
            tied_embedding_weight=self.token_emb.embedding.weight
        )
        self.gate_net = GateNet(input_dim=1, hidden_dim=config.hidden_size // 16, g_max=config.g_max)

        # 用于损失计算的临时变量
        self.z_pred_for_loss = None
        self.z_teacher_for_loss = None
        self.logits_span_for_loss = None

    def _calc_entropy(self, logits: torch.Tensor):
        """计算熵"""
        logits = torch.clamp(logits, min=-50, max=50)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.clamp(entropy, min=0, max=20)
        return entropy

    def _select_top_m_positions(self, entropy: torch.Tensor, filled_mask: torch.Tensor, M_t: torch.Tensor):
        """选择优先级最高的M_t个位置"""
        B, K = entropy.shape
        pick_mask = torch.zeros_like(filled_mask, dtype=torch.bool)

        for b in range(B):
            m_b = int(M_t[b].item())
            if m_b <= 0:
                continue

            available_positions = (~filled_mask[b]).nonzero().flatten()
            if len(available_positions) == 0:
                continue

            available_entropy = entropy[b, available_positions]
            num_select = min(m_b, len(available_positions))

            if num_select > 0:
                _, selected_indices = torch.topk(available_entropy, num_select, largest=False)
                selected_positions = available_positions[selected_indices]
                pick_mask[b, selected_positions] = True

        return pick_mask

    def span_forward_pass(self,
                          seq_prompt: torch.Tensor,
                          K: int,
                          g: int = None,
                          temperature: float = 1.0,
                          top_p: float = 0.9,
                          is_teacher_path: bool = False):
        """简化版本的span前向传播，避免复杂的增量更新"""

        B, n_prompt = seq_prompt.shape
        device = seq_prompt.device

        if K == 0:
            return {'seq': seq_prompt, 'z_pred': None, 'logits_span': None, 'p_g': None}

        # 1. 构建初始序列
        pred_tokens = torch.full((B, K), self.config.pred_token_id, dtype=torch.long, device=device)
        seq = torch.cat([seq_prompt, pred_tokens], dim=1)

        # 2. 初始decoder前向
        embeddings = self.token_emb(seq)
        position_ids = torch.arange(n_prompt + K, device=device).unsqueeze(0).expand(B, -1)

        h_dec, _ = self.decoder(
            hidden_states=embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False  # 简化：不使用KV缓存
        )

        # 3. 计算初始熵
        h_pred_positions = h_dec[:, n_prompt:, :]
        logits_pred = self.proj_head(h_pred_positions)
        entropy = self._calc_entropy(logits_pred)

        # 4. GateNet预测步数
        if g is not None:
            g_hat = torch.full((B,), g, dtype=torch.long, device=device)
            gate_logits = None
        else:
            mean_entropy = entropy.mean(dim=-1, keepdim=True)
            g_hat, gate_logits = self.gate_net(mean_entropy, train=self.training and not is_teacher_path)

        g_loop = g_hat.max().item() if g_hat.numel() > 0 else 0

        # 5. 简化的refinement过程
        z = self.token_emb(pred_tokens)
        filled_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

        for t in range(g_loop):
            active_mask = g_hat > t
            if not active_mask.any():
                break

            # 计算当前步的填充数量
            g_active = g_hat[active_mask].float()
            M_t_active = torch.ceil((t + 1) * K / g_active) - torch.ceil(t * K / g_active)
            M_t = torch.zeros(B, device=device)
            M_t[active_mask] = M_t_active

            # 选择要填充的位置
            pick_mask = self._select_top_m_positions(entropy, filled_mask, M_t)

            # Encoder refinement
            z = self.encoder.step(z, h_pred_positions.detach(), n_prompt)

            # 投射并采样
            if pick_mask.any():
                z_picked = z[pick_mask]
                logits_picked = self.proj_head(z_picked) / temperature

                if is_teacher_path:
                    tokens_picked = torch.argmax(logits_picked, dim=-1)
                else:
                    tokens_picked = sample_from_logits(logits_picked, temperature, top_p)

                # 更新序列
                seq_new = seq.clone()
                seq_new[:, n_prompt:][pick_mask] = tokens_picked
                seq = seq_new
                filled_mask = filled_mask | pick_mask

            # 保存用于损失计算
            if t == g_loop - 1:
                if is_teacher_path:
                    self.z_teacher_for_loss = z.detach().clone()
                else:
                    self.z_pred_for_loss = z.clone()
                    self.logits_span_for_loss = self.proj_head(z)

        # 确保损失变量被设置
        if not is_teacher_path and self.z_pred_for_loss is None:
            self.z_pred_for_loss = z.clone()
            self.logits_span_for_loss = self.proj_head(z)

        return {
            'seq': seq,
            'z_pred': z if not is_teacher_path else None,
            'logits_span': self.proj_head(z) if not is_teacher_path else None,
            'p_g': F.softmax(gate_logits, dim=-1) if gate_logits is not None else None
        }

    def forward(self,
                seq_prompt: torch.Tensor,
                K: int,
                gold_span: torch.Tensor = None,
                temperature: float = 1.0,
                top_p: float = 0.9,
                compute_teacher_latent: bool = False):
        """模型前向传播"""

        # Student路径
        student_output = self.span_forward_pass(
            seq_prompt, K, g=None,
            temperature=temperature, top_p=top_p,
            is_teacher_path=False
        )

        # Teacher路径
        if compute_teacher_latent:
            with torch.no_grad():
                _ = self.span_forward_pass(
                    seq_prompt, K, g=self.config.g_max,
                    temperature=1.0, top_p=1.0,
                    is_teacher_path=True
                )

        return student_output
