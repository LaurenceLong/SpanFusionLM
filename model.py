import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from dataclasses import dataclass, field

from .modules.token_emb import TokenEmbedding
from .modules.decoder import SpanDecoder
from .modules.encoder import SpanEncoder
from .modules.proj_head import ProjectionHead
from .modules.tokenizer import Gpt2Tokenizer

@dataclass
class SpanFusionLMConfig:
    vocab_size: int = 50257
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_decoder_layers: int = 28
    num_encoder_layers: int = 8
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048 + 32
    rope_theta: float = 10000.0
    pred_token_id: int = None
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None
    g_max: int = 8
    tokenizer: Gpt2Tokenizer = field(default_factory=Gpt2Tokenizer)

    def __post_init__(self):
        self.pred_token_id = self.tokenizer.pred_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

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
    logits = logits / temperature
    logits = top_p_logits_processor(logits, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if probs.ndim == 1:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampled_tokens

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

        self.z_pred_for_loss = None
        self.z_teacher_for_loss = None
        self.logits_span_for_loss = None

    def _top_m_picker(self, priority_scores: torch.Tensor, filled_mask: torch.Tensor, M_t: int):
        """根据优先级分数选择未填充的 top M_t 个位置"""
        B, K_dim = priority_scores.shape
        scores_masked = priority_scores.clone()
        scores_masked[filled_mask] = float('inf')
        pick_mask = torch.zeros_like(filled_mask, dtype=torch.bool)

        for i in range(B):
            row_scores = scores_masked[i]
            num_unfilled = (~filled_mask[i]).sum().item()
            actual_M = min(M_t, num_unfilled)
            if actual_M == 0:
                continue
            _, top_indices = torch.topk(row_scores, k=actual_M, largest=False, sorted=False)
            pick_mask[i, top_indices] = True
        return pick_mask

    def span_forward_pass(self,
                          seq_prompt: torch.Tensor,
                          K: int,
                          g: int,
                          temperature: float = 1.0,
                          top_p: float = 0.9,
                          is_teacher_path: bool = False):
        """实现伪代码中的 span_fwd 函数"""
        B, n = seq_prompt.shape
        device = seq_prompt.device

        # 1. 在序列尾部插入 [PRED]×K 占位符
        pred_tokens = torch.full((B, K), self.config.pred_token_id, dtype=torch.long, device=device)
        seq = torch.cat([seq_prompt, pred_tokens], dim=1)

        # 2. Decoder 全量前向
        full_seq_len = n + K
        decoder_pos_ids = torch.arange(full_seq_len, device=device).unsqueeze(0).expand(B, -1)
        current_embeddings = self.token_emb(seq)
        h_dec_full, kv_cache = self.decoder(
            hidden_states=current_embeddings,
            position_ids=decoder_pos_ids,
            past_key_values=None,
            use_cache=True
        )

        # 3. 计算占位符位置的 logits 和熵（优先级）
        logits_pred = self.proj_head(h_dec_full[:, -K:, :])  # (B, K, V)
        log_probs = F.log_softmax(logits_pred, dim=-1)
        probs = F.softmax(logits_pred, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, K)

        # 4. 初始化 latent z
        z = self.token_emb(pred_tokens)  # (B, K, d)
        filled = torch.zeros(B, K, dtype=torch.bool, device=device)
        # 用于 encoder cross-attention 的 h_dec 部分初始取最后 K 个位置（脱离计算图）
        h_dec_for_encoder = h_dec_full[:, -K:, :].detach()

        # 5. g 步迭代
        for t in range(g):
            # 5.1 计算本步批量大小 Mₜ（逐步增大）
            M_t = math.ceil((t + 1) * K / g) - math.ceil(t * K / g)
            pick = self._top_m_picker(entropy, filled, M_t)

            # 5.2 Encoder latent refinement（K 个位置全部更新）
            z = self.encoder.step(
                latent_z=z,
                h_dec_k_positions=h_dec_for_encoder,
                prompt_len=n
            )

            # 5.3 投射 pick 位置并写回
            if pick.any():
                z_picked = z[pick]  # (num_picked, d)
                logits_picked = self.proj_head(z_picked) / temperature

                if is_teacher_path:
                    # teacher 路径使用确定性选择
                    tok_hat = torch.argmax(logits_picked, dim=-1)
                else:
                    # student 路径使用随机采样
                    tok_hat = sample_from_logits(logits_picked, temperature=temperature, top_p=top_p)

                # 写回序列中 span 部分
                seq_k_part = seq[:, n:]
                seq_k_part[pick] = tok_hat
                seq[:, n:] = seq_k_part
                filled[pick] = True

            # 保存最终 latent 用于损失计算
            if t == g - 1:
                if is_teacher_path:
                    self.z_teacher_for_loss = z.detach()
                else:
                    self.z_pred_for_loss = z
                    self.logits_span_for_loss = self.proj_head(z)

            # 5.4 非最后一步：增量重算 Decoder & 更新 priority 信息
            if t != g - 1 and pick.any():
                picked_indices = pick.nonzero(as_tuple=False)
                first_changed_rel_idx = picked_indices[:, 1].min().item()
                start_pos_inc = n + first_changed_rel_idx

                tokens_inc = seq[:, start_pos_inc:]
                num_inc = tokens_inc.shape[1]
                incremental_embeddings = self.token_emb(tokens_inc)
                inc_pos_ids = torch.arange(start_pos_inc, start_pos_inc + num_inc, device=device).unsqueeze(0).expand(B, -1)

                sliced_kv_cache = []
                if kv_cache is not None:
                    for layer_kv in kv_cache:
                        k_cache, v_cache = layer_kv
                        sliced_k = k_cache[:, :, :start_pos_inc, :]
                        sliced_v = v_cache[:, :, :start_pos_inc, :]
                        sliced_kv_cache.append((sliced_k, sliced_v))
                sliced_kv_cache = tuple(sliced_kv_cache) if sliced_kv_cache else None

                h_dec_delta, updated_kv_cache = self.decoder(
                    hidden_states=incremental_embeddings,
                    position_ids=inc_pos_ids,
                    past_key_values=sliced_kv_cache,
                    use_cache=True
                )

                kv_cache = updated_kv_cache
                h_dec_full[:, start_pos_inc:, :] = h_dec_delta

                rel_change_idx = first_changed_rel_idx
                h_dec_updated = h_dec_full[:, n + rel_change_idx:n + K, :]
                if h_dec_updated.size(1) > 0:
                    logits_updated = self.proj_head(h_dec_updated)
                    logits_pred[:, rel_change_idx:, :] = logits_updated

                rem_mask = ~filled
                if rem_mask.any():
                    logits_rem = logits_pred[rem_mask]
                    log_probs_rem = F.log_softmax(logits_rem, dim=-1)
                    probs_rem = F.softmax(logits_rem, dim=-1)
                    entropy_rem = -(probs_rem * log_probs_rem).sum(dim=-1)
                    entropy[rem_mask] = entropy_rem

            # ★ 修改：增量更新后将 encoder 使用的 decoder 隐状态更新为最新值
            if t != g - 1:
                h_dec_for_encoder = h_dec_full[:, n:]

            # 如果全部填满，则提前退出
            if filled.all():
                break

        return seq

    def forward(self,
                seq_prompt: torch.Tensor,
                K: int,
                g: int = None,
                gold_span: torch.Tensor = None,
                temperature: float = 1.0,
                top_p: float = 0.9,
                compute_teacher_latent: bool = False):
        """主要的前向传播函数"""
        if g is None:
            g_runtime = random.randint(1, self.config.g_max)
        else:
            g_runtime = g

        # 学生路径
        seq_out = self.span_forward_pass(
            seq_prompt, K, g_runtime,
            temperature=temperature, top_p=top_p,
            is_teacher_path=False
        )

        # 如果需要，计算 teacher latent（采用最大步数）
        if compute_teacher_latent:
            with torch.no_grad():
                _ = self.span_forward_pass(
                    seq_prompt, K, self.config.g_max,
                    temperature=temperature, top_p=top_p,
                    is_teacher_path=True
                )

        return seq_out
