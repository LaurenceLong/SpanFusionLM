import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field

from .modules.token_emb import TokenEmbedding
from .modules.decoder import SpanDecoder
from .modules.encoder import SpanEncoder
from .modules.proj_head import ProjectionHead
from .modules.tokenizer import Gpt2Tokenizer
from .modules.gate_net import GateNet

# 如果没有单独定义 sample_from_logits ，可直接在此定义：
def top_p_logits_processor(logits: torch.Tensor, top_p: float):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0  # 始终保留最可能的 token
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
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

@dataclass
class SpanFusionLMConfig:
    vocab_size: int = 50257
    hidden_size: int = 4096
    intermediate_size: int = 11008  # Standard for Llama 7B-like models
    num_decoder_layers: int = 28
    num_encoder_layers: int = 8
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048 + 32  # 最大序列长度 + span 扩展
    rope_theta: float = 10000.0
    g_max: int = 8  # 最大 encoder refinement 步数
    # 默认使用 GPT2Tokenizer 封装类，内部会添加特殊 token
    tokenizer: Gpt2Tokenizer = field(default_factory=Gpt2Tokenizer)

    # 以下特殊 token ID 将在 __post_init__ 中设置
    pred_token_id: int = None
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None

    def __post_init__(self):
        self.pred_token_id = self.tokenizer.pred_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

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
        # 修改：新增 GateNet 实例（每个样本独立预测 ĝ），并传入 g_max
        self.gate_net = GateNet(hidden_dim=16, g_max=self.config.g_max)

        # 用于计算损失时保存中间变量
        self.z_pred_for_loss = None
        self.z_teacher_for_loss = None
        self.logits_span_for_loss = None

    def _top_m_picker(self, priority_scores: torch.Tensor, filled_mask: torch.Tensor, M_t: torch.Tensor):
        B, K_dim = priority_scores.shape
        scores_masked = priority_scores.clone()
        scores_masked[filled_mask] = float('inf')
        pick_mask = torch.zeros_like(filled_mask, dtype=torch.bool)
        for b in range(B):
            # 只处理需要更新的样本（若没有活跃则跳过）
            m = int(M_t[b].item())
            if m == 0:
                continue
            cand = (~filled_mask[b]).nonzero().flatten()
            if cand.numel() == 0:
                continue
            # 从候选槽位中选出 m 个熵值最低的
            sorted_indices = cand[priority_scores[b, cand].argsort()]
            selected = sorted_indices[:min(m, len(sorted_indices))]
            pick_mask[b, selected] = True
        return pick_mask

    def span_forward_pass(self,
                          seq_prompt: torch.Tensor,  # (B, n)
                          K: int,                  # span 长度
                          g: int = None,           # 若为 None，则由 GateNet 动态确定（每个样本可能不同）
                          temperature: float = 1.0,
                          top_p: float = 0.9,
                          is_teacher_path: bool = False):
        B, n = seq_prompt.shape
        device = seq_prompt.device
        # 在序列尾部添加 K 个 [PRED] 占位符
        pred_token = torch.full((B, K), self.config.pred_token_id, dtype=torch.long, device=device)
        seq = torch.cat([seq_prompt, pred_token], dim=1)  # (B, n+K)
        full_seq_len = n + K

        # 构造位置 id
        decoder_pos_ids = torch.arange(full_seq_len, device=device).unsqueeze(0).expand(B, -1)
        current_embeddings = self.token_emb(seq)  # (B, n+K, d)
        h_dec_full, kv_cache = self.decoder(
            hidden_states=current_embeddings,
            attention_mask=None,
            position_ids=decoder_pos_ids,
            past_key_values=None,
            use_cache=True
        )
        # 针对 span 部分得到 logits 与熵值，形状 (B, K, V) 和 (B, K)
        logits_all_K_preds = self.proj_head(h_dec_full[:, n:, :])
        log_probs_preds = F.log_softmax(logits_all_K_preds, dim=-1)
        probs_preds = F.softmax(logits_all_K_preds, dim=-1)
        entropy = -(probs_preds * log_probs_preds).sum(dim=-1)

        # GateNet：若未传入 g，则动态计算（返回每个样本的 ĝ 和 gate_net 的 logits，用于成本项计算）
        if g is None:
            g_hat, gate_logits = self.gate_net(entropy, train=not is_teacher_path)  # g_hat: (B,)，gate_logits: (B, g_max)
        else:
            g_hat = torch.full((B,), g, dtype=torch.long, device=device)
            gate_logits = None
        g_loop = g_hat.max().item()  # 外层循环以 batch 内最大 ĝ 为上限

        # 初始化 latent z 与已填充标记
        z = self.token_emb(pred_token)  # (B, K, d)
        filled_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        # h_dec_for_encoder 仅取 span 部分，用作 cross-attention（detach防止梯度流回 decoder）
        h_dec_for_encoder = h_dec_full[:, n:, :].detach()

        for t in range(g_loop):
            # 对每个样本判断其是否仍需进行更新（ĝ > t）
            active = g_hat > t  # (B,)
            if not active.any():
                break
            # 计算每个样本本步所需更新槽位数量 M_t[b] = ceil((t+1)*K/ĝ[b]) - ceil(t*K/ĝ[b])
            M_t = torch.ceil((t + 1) * K / g_hat.float()) - torch.ceil(t * K / g_hat.float())  # (B,)
            pick_mask_batch = self._top_m_picker(entropy, filled_mask, M_t)
            # Encoder latent refinement：对全部 K 个槽位更新 latent
            z = self.encoder.step(
                latent_z=z,
                h_dec_k_positions=h_dec_for_encoder,
                prompt_len=n
            )
            if pick_mask_batch.any():
                z_picked = z[pick_mask_batch]
                logits_picked = self.proj_head(z_picked) / temperature
                if is_teacher_path:
                    tok_hat = torch.argmax(logits_picked, dim=-1)
                else:
                    tok_hat = sample_from_logits(logits_picked, temperature=temperature, top_p=top_p)
                # 写回对应 [PRED] 位置（仅更新被选中的位置）
                seq_K_part = seq[:, n:]
                seq_K_part[pick_mask_batch] = tok_hat
                seq[:, n:] = seq_K_part
                filled_mask[pick_mask_batch] = True

            # 收集 teacher latent 或 student latent
            if t == g_loop - 1:
                if is_teacher_path:
                    self.z_teacher_for_loss = z.detach()
                else:
                    self.z_pred_for_loss = z
                    self.logits_span_for_loss = self.proj_head(z)
            # 非最后一步时进行增量重新计算 Decoder（RoPE 缓存更新）
            if t != g_loop - 1 and pick_mask_batch.any():
                picked_indices = pick_mask_batch.nonzero(as_tuple=False)
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
                h_dec_delta, updated_kv_cache_delta = self.decoder(
                    hidden_states=incremental_embeddings,
                    attention_mask=None,
                    position_ids=inc_pos_ids,
                    past_key_values=sliced_kv_cache,
                    use_cache=True
                )
                kv_cache = updated_kv_cache_delta
                h_dec_full[:, start_pos_inc:, :] = h_dec_delta
                rel_change_idx = first_changed_rel_idx
                h_dec_updated = h_dec_full[:, n + rel_change_idx:n + K, :]
                if h_dec_updated.size(1) > 0:
                    logits_updated = self.proj_head(h_dec_updated)
                    logits_all_K_preds[:, rel_change_idx:, :] = logits_updated
                rem_mask = ~filled_mask
                if rem_mask.any():
                    logits_rem = logits_all_K_preds[rem_mask]
                    log_probs_rem = F.log_softmax(logits_rem, dim=-1)
                    probs_rem = F.softmax(logits_rem, dim=-1)
                    entropy_rem = -(probs_rem * log_probs_rem).sum(dim=-1)
                    entropy[rem_mask] = entropy_rem
            if filled_mask.all():
                break

        output = {
            'seq': seq,
            'z_pred': z,
            'logits_span': self.proj_head(z),
            'p_g': F.softmax(gate_logits, dim=-1) if gate_logits is not None else None
        }
        return output

    def forward(self,
                seq_prompt: torch.Tensor,
                K: int,
                g: int = None,
                gold_span: torch.Tensor = None,
                temperature: float = 1.0,
                top_p: float = 0.9,
                compute_teacher_latent: bool = False):
        seq_out = self.span_forward_pass(
            seq_prompt, K, g,
            temperature=temperature, top_p=top_p,
            is_teacher_path=False
        )
        if compute_teacher_latent:
            with torch.no_grad():
                _ = self.span_forward_pass(
                    seq_prompt, K, g=self.config.g_max,
                    temperature=temperature, top_p=top_p,
                    is_teacher_path=True
                )
        return seq_out
