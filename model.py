# model.py
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
        if hasattr(self.tokenizer, 'additional_special_tokens') and '<|PRED|>' in self.tokenizer.additional_special_tokens:
            self.pred_token_id = self.tokenizer.additional_special_tokens_ids[
                self.tokenizer.additional_special_tokens.index('<|PRED|>')
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
        # 使用基于二分类的 GateNet
        self.gate_net = GateNet(input_dim=1, hidden_dim=config.hidden_size // 16, g_max=config.g_max)

        # 用于损失计算的临时变量
        self.z_pred_for_loss = None
        self.z_teacher_for_loss = None
        self.logits_span_for_loss = None

    def _calc_entropy(self, logits: torch.Tensor):
        logits = torch.clamp(logits, min=-50, max=50)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.clamp(entropy, min=0, max=20)
        return entropy

    def _select_top_m_positions(self, entropy: torch.Tensor, filled_mask: torch.Tensor, M_t: torch.Tensor):
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
                          temperature: float = 1.0,
                          top_p: float = 0.9,
                          is_teacher_path: bool = False):
        """
        使用二分类GateNet实现动态迭代：每一步由GateNet根据当前平均熵决定是否继续迭代。
        """
        B, n_prompt = seq_prompt.shape
        device = seq_prompt.device

        # 重置 GateNet 的 override_fn
        if self.training:
            self.gate_net.override(None)

        # 构建初始序列
        pred_tokens = torch.full((B, K), self.config.pred_token_id, dtype=torch.long, device=device)
        seq = torch.cat([seq_prompt, pred_tokens], dim=1)

        # 初始化填充掩码，记录哪些位置已更新
        filled_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        # 初始化迭代计数（每个样本实际使用的迭代次数）
        iteration_count = torch.zeros(B, device=device, dtype=torch.long)
        final_z = None
        final_logits = None

        # active_mask 表示哪些样本当前仍在迭代中（开始时全部active）
        active_mask = torch.ones(B, dtype=torch.bool, device=device)
        max_iters = self.config.g_max
        # 固定每步更新的 token 数量（例如均分更新 K 个 token）
        M_step = (K + max_iters - 1) // max_iters  # 向上取整

        # 初始化隐状态 z 为预测 token 的嵌入
        z = self.token_emb(pred_tokens)

        for t in range(max_iters):
            if not active_mask.any():
                break

            # 全样本前向传播（后续更新只对active样本生效）
            embeddings = self.token_emb(seq)
            position_ids = torch.arange(seq.shape[1], device=device).unsqueeze(0).expand(B, -1)
            h_dec, _ = self.decoder(
                hidden_states=embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False
            )
            h_pred_positions = h_dec[:, n_prompt:, :]
            logits_pred = self.proj_head(h_pred_positions)
            entropy = self._calc_entropy(logits_pred)  # (B, K)

            # 对于 active 样本，每步计划更新 M_step 个 token，其余样本更新数为 0
            M_t = torch.where(active_mask, torch.full((B,), M_step, device=device), torch.zeros(B, device=device))
            pick_mask = self._select_top_m_positions(entropy, filled_mask, M_t)

            # Encoder refinement：更新隐状态 z
            z = self.encoder.step(z, h_pred_positions.detach(), n_prompt)

            if pick_mask.any():
                z_picked = z[pick_mask]
                logits_picked = self.proj_head(z_picked) / temperature
                if is_teacher_path:
                    tokens_picked = torch.argmax(logits_picked, dim=-1)
                else:
                    tokens_picked = sample_from_logits(logits_picked, temperature, top_p)
                # 更新对应位置的 token
                seq_updated = seq.clone()
                seq_span = seq_updated[:, n_prompt:]
                seq_span[pick_mask] = tokens_picked
                seq_updated[:, n_prompt:] = seq_span
                seq = seq_updated
                filled_mask = filled_mask | pick_mask

            # 保存最后一次迭代的隐状态和 logits（用于损失计算）
            if t == max_iters - 1:
                final_z = z.clone()
                final_logits = self.proj_head(z.clone())

            # 计算每个样本当前的平均熵，用于判断是否继续迭代
            avg_entropy = entropy.mean(dim=-1, keepdim=True)  # (B,1)
            decision, gate_logits = self.gate_net(avg_entropy, train=self.training and not is_teacher_path)
            # decision：1 表示继续，0 表示提前终止
            new_active = active_mask.clone()
            new_active[active_mask] = (decision[active_mask] == 1)
            # 对于还选择继续的样本，本轮迭代计数加 1
            iteration_count[active_mask] += 1
            active_mask = new_active

        # 保存用于损失计算的隐状态
        if is_teacher_path:
            self.z_teacher_for_loss = z.detach().clone()
        else:
            self.z_pred_for_loss = z.clone()
            self.logits_span_for_loss = self.proj_head(z)

        return {
            'seq': seq,
            'z_pred': final_z if not is_teacher_path else None,
            'logits_span': final_logits if not is_teacher_path else None,
            'p_g': F.softmax(gate_logits, dim=-1) if gate_logits is not None else None,
            'g_hat': iteration_count  # 每个样本的实际迭代步数
        }

    def forward(self,
                seq_prompt: torch.Tensor,
                K: int,
                gold_span: torch.Tensor = None,
                temperature: float = 1.0,
                top_p: float = 0.9,
                compute_teacher_latent: bool = False):
        # Student路径
        student_output = self.span_forward_pass(
            seq_prompt, K,
            temperature=temperature, top_p=top_p,
            is_teacher_path=False
        )
        # Teacher路径：使用detach的决策进行辅助
        if compute_teacher_latent:
            with torch.no_grad():
                _ = self.span_forward_pass(
                    seq_prompt, K,
                    temperature=1.0, top_p=1.0,
                    is_teacher_path=True
                )
        return student_output
