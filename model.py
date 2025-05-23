# SpanFusionLM/model.py
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.decoder import SpanDecoder, sample_from_logits
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
    """
    对 logits 进行温度缩放、top-p 筛选，并用 multinomial 采样 token。
    如果采样分布不合法，则回退到 argmax。
    """
    logits = logits / temperature
    if top_p < 1.0:
        logits = top_p_logits_processor(logits, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)

    # 检查 probs 是否存在 NaN、inf 或者概率分布总和为 0 的情况
    if torch.isnan(probs).any() or torch.isinf(probs).any() or torch.sum(probs) <= 0:
        # 回退到取最大值
        return torch.argmax(logits, dim=-1)

    # 如果 probs 为一维，则调整形状后采样
    if probs.ndim == 1:
        return torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze(-1).squeeze(0)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@dataclass
class SpanFusionLMConfig:
    vocab_size: int = 32000  # Default, will be overridden by tokenizer
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_decoder_layers: int = 28
    num_encoder_layers: int = 8
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048 + 32
    rope_theta: float = 10000.0
    g_max: int = 8
    tokenizer_name: str = "gpt2"  # Store tokenizer name for reloading
    tokenizer: any = field(default=None, repr=False, compare=False)  # Avoid serializing full tokenizer

    # Special token IDs - will be set in __post_init__
    pred_token_id: int = None
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(self.tokenizer_name)

        if self.vocab_size is None or self.vocab_size == 32000:
            self.vocab_size = len(self.tokenizer)

        if not hasattr(self.tokenizer, 'pred_token_id') or self.tokenizer.additional_special_tokens_ids is None:
            pass

        self.pred_token_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index('[PRED]')
        ]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        if self.tokenizer_name == "gpt2" and hasattr(self.tokenizer, 'name_or_path') and self.tokenizer.name_or_path:
            self.tokenizer_name = self.tokenizer.name_or_path

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
        # 修改这里：GateNet 的输入应为标量 (B,1) 而非 (B, g_max)
        self.gate_net = GateNet(input_dim=1, hidden_dim=config.hidden_size // 256, g_max=self.config.g_max)

        self.z_pred_for_loss = None
        self.z_teacher_for_loss = None
        self.logits_span_for_loss = None

    def _top_m_picker(self, priority_scores: torch.Tensor, filled_mask: torch.Tensor, M_t: torch.Tensor):
        B, K_dim = priority_scores.shape
        scores_masked = priority_scores.clone()
        scores_masked[filled_mask] = float('inf')

        pick_mask = torch.zeros_like(filled_mask, dtype=torch.bool)
        for b in range(B):
            m_b = int(M_t[b].item())
            if m_b == 0:
                continue
            unfilled_indices = (~filled_mask[b]).nonzero().flatten()
            if unfilled_indices.numel() == 0:
                continue
            scores_unfilled = scores_masked[b, unfilled_indices]
            num_to_pick = min(m_b, unfilled_indices.numel())
            if num_to_pick == 0:
                continue
            _, sorted_relative_indices = scores_unfilled.topk(num_to_pick, largest=False)
            actual_indices_to_pick = unfilled_indices[sorted_relative_indices]
            pick_mask[b, actual_indices_to_pick] = True

        return pick_mask

    def _calc_entropy(self, logits: torch.Tensor):
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0)
        return entropy

    def span_forward_pass(self,
                          seq_prompt: torch.Tensor,
                          K: int,
                          g: int = None,
                          temperature: float = 1.0,
                          top_p: float = 0.9,
                          is_teacher_path: bool = False):
        B, n_prompt = seq_prompt.shape
        device = seq_prompt.device

        if K == 0:
            output = {
                'seq': seq_prompt,
                'z_pred': None,
                'logits_span': None,
                'p_g': None
            }
            if is_teacher_path:
                self.z_teacher_for_loss = None
            else:
                self.z_pred_for_loss = None
                self.logits_span_for_loss = None
            return output

        pred_tokens = torch.full((B, K), self.config.pred_token_id, dtype=torch.long, device=device)
        seq = torch.cat([seq_prompt, pred_tokens], dim=1)
        full_seq_len = n_prompt + K

        current_embeddings = self.token_emb(seq)
        decoder_pos_ids = torch.arange(full_seq_len, device=device).unsqueeze(0).expand(B, -1)

        h_dec_full, kv_cache = self.decoder(
            hidden_states=current_embeddings,
            attention_mask=None,
            position_ids=decoder_pos_ids,
            past_key_values=None,
            use_cache=True
        )
        logits_K_preds = self.proj_head(h_dec_full[:, n_prompt:, :])
        entropy = self._calc_entropy(logits_K_preds)

        if g is not None:
            g_hat = torch.full((B,), g, dtype=torch.long, device=device)
            gate_logits_for_loss = None
        else:
            mean_entropy_for_gate = entropy.mean(dim=-1)
            g_hat, gate_logits_for_loss = self.gate_net(mean_entropy_for_gate.unsqueeze(-1),
                                                        train=self.training and not is_teacher_path)

        g_loop = g_hat.max().item() if g_hat.numel() > 0 else 0

        z = self.token_emb(pred_tokens)
        filled_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        h_dec_K_positions_detached = h_dec_full[:, n_prompt:, :].detach()

        for t in range(g_loop):
            active_samples_mask = g_hat > t
            if not active_samples_mask.any():
                break

            g_hat_active_float = g_hat[active_samples_mask].float()
            term1 = torch.ceil(((t + 1) * K) / g_hat_active_float)
            term2 = torch.ceil((t * K) / g_hat_active_float)
            M_t_active = term1 - term2
            M_t = torch.zeros(B, device=device, dtype=torch.float)
            M_t[active_samples_mask] = M_t_active

            current_pick_mask = torch.zeros_like(filled_mask)
            if active_samples_mask.any():
                current_pick_mask[active_samples_mask] = self._top_m_picker(
                    entropy[active_samples_mask],
                    filled_mask[active_samples_mask],
                    M_t[active_samples_mask]
                )

            z = self.encoder.step(
                latent_z=z,
                h_dec_k_positions=h_dec_K_positions_detached,
                prompt_len=n_prompt
            )

            if current_pick_mask.any():
                z_picked = z[current_pick_mask]
                if z_picked.numel() > 0:
                    logits_picked = self.proj_head(z_picked) / temperature
                    if is_teacher_path:
                        tok_hat = torch.argmax(logits_picked, dim=-1)
                    else:
                        tok_hat = sample_from_logits(logits_picked, temperature=temperature, top_p=top_p)

                    # 避免就地修改：克隆后拼接
                    seq_K_part = seq[:, n_prompt:].clone()
                    seq_K_part[current_pick_mask] = tok_hat
                    seq = torch.cat([seq[:, :n_prompt], seq_K_part], dim=1)
                    filled_mask[current_pick_mask] = True

            if t == g_loop - 1:
                if is_teacher_path:
                    # 增加数值处理，防止 NaN
                    self.z_teacher_for_loss = torch.nan_to_num(z.detach().clone(), nan=0.0, posinf=1e6, neginf=-1e6)
                else:
                    self.z_pred_for_loss = torch.nan_to_num(z.clone(), nan=0.0, posinf=1e6, neginf=-1e6)
                    self.logits_span_for_loss = self.proj_head(self.z_pred_for_loss)

            if t < g_loop - 1 and current_pick_mask.any() and not filled_mask.all():
                picked_indices_b_k = current_pick_mask.nonzero(as_tuple=False)
                if picked_indices_b_k.numel() == 0:
                    continue

                first_changed_k_idx_relative = picked_indices_b_k[:, 1].min().item()
                start_pos_incremental_global = n_prompt + first_changed_k_idx_relative
                tokens_for_incremental_pass = seq[:, start_pos_incremental_global: n_prompt + K]

                if tokens_for_incremental_pass.shape[1] > 0:
                    incremental_embeddings = self.token_emb(tokens_for_incremental_pass)
                    num_incremental_tokens = tokens_for_incremental_pass.shape[1]
                    incremental_pos_ids = torch.arange(
                        start_pos_incremental_global,
                        start_pos_incremental_global + num_incremental_tokens,
                        device=device
                    ).unsqueeze(0).expand(B, -1)
                    sliced_kv_cache_for_inc = []
                    if kv_cache:
                        for layer_kv in kv_cache:
                            k_c, v_c = layer_kv
                            sliced_k = k_c[:, :, :start_pos_incremental_global, :]
                            sliced_v = v_c[:, :, :start_pos_incremental_global, :]
                            sliced_kv_cache_for_inc.append((sliced_k, sliced_v))
                    past_kv_for_incremental = tuple(sliced_kv_cache_for_inc) if sliced_kv_cache_for_inc else None

                    h_dec_delta, updated_kv_cache_full = self.decoder(
                        hidden_states=incremental_embeddings,
                        attention_mask=None,
                        position_ids=incremental_pos_ids,
                        past_key_values=past_kv_for_incremental,
                        use_cache=True
                    )

                    kv_cache = updated_kv_cache_full
                    # 避免原地修改，克隆后更新
                    h_dec_full_updated = h_dec_full.clone()
                    h_dec_full_updated[:, start_pos_incremental_global: n_prompt + K, :] = h_dec_delta
                    h_dec_full = h_dec_full_updated

                    # 更新 encoder 使用的 decoder 隐状态
                    h_dec_K_positions_detached = h_dec_full[:, n_prompt:, :].detach()

                    current_h_dec_K = h_dec_full[:, n_prompt:, :]
                    logits_K_updated = self.proj_head(current_h_dec_K)
                    logits_K_preds = logits_K_updated
                    entropy = self._calc_entropy(logits_K_preds)

            if filled_mask.all():
                break

        if not is_teacher_path and self.z_pred_for_loss is None:
            self.z_pred_for_loss = torch.nan_to_num(z.clone(), nan=0.0, posinf=1e6, neginf=-1e6)
            self.logits_span_for_loss = self.proj_head(self.z_pred_for_loss)

        output = {
            'seq': seq,
            'z_pred': z if not is_teacher_path else None,
            'logits_span': self.proj_head(z) if not is_teacher_path else None,
            'p_g': F.softmax(gate_logits_for_loss, dim=-1) if gate_logits_for_loss is not None else None,
        }
        return output

    def forward(self,
                seq_prompt: torch.Tensor,
                K: int,
                gold_span: torch.Tensor = None,
                temperature: float = 1.0,
                top_p: float = 0.9,
                compute_teacher_latent: bool = False):

        student_output = self.span_forward_pass(
            seq_prompt, K, g=None,
            temperature=temperature, top_p=top_p,
            is_teacher_path=False
        )

        if compute_teacher_latent:
            with torch.no_grad():
                _ = self.span_forward_pass(
                    seq_prompt, K, g=self.config.g_max,
                    temperature=1.0, top_p=1.0,
                    is_teacher_path=True
                )

        return student_output