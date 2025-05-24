# model.py
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.decoder import SpanDecoder
from .modules.encoder import SpanEncoder
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
    if logits.dim() == 2:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    elif logits.dim() == 3:
        B, L, _ = logits.shape
        logits_flat = logits.view(-1, logits.size(-1))
        tokens = torch.multinomial(torch.softmax(logits_flat, dim=-1), num_samples=1)
        return tokens.view(B, L)
    else:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


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
    max_seq_length: int = 512  # 新增最大序列长度，用于计算固定 span 长度
    tokenizer_name: str = "gpt2"
    tokenizer: any = field(default=None, repr=False, compare=False)

    pred_token_id: int = None
    tbd_token_id: int = None  # 用以记录 <|TBD|> token 的 ID
    pad_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None
    fixed_span_length: int = None  # 将在 __post_init__ 中计算

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(self.tokenizer_name)
        if self.vocab_size is None or self.vocab_size == 32000:
            self.vocab_size = len(self.tokenizer)
        # 确保特殊 token <|PRED|> 和 <|TBD|> 已经被添加
        if hasattr(self.tokenizer, 'additional_special_tokens'):
            if '<|PRED|>' not in self.tokenizer.additional_special_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|PRED|>']})
            self.pred_token_id = self.tokenizer.additional_special_tokens_ids[
                self.tokenizer.additional_special_tokens.index('<|PRED|>')
            ]
            if '<|TBD|>' not in self.tokenizer.additional_special_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|TBD|>']})
            self.tbd_token_id = self.tokenizer.additional_special_tokens_ids[
                self.tokenizer.additional_special_tokens.index('<|TBD|>')
            ]
        else:
            self.pred_token_id = self.pad_token_id  # fallback
            self.tbd_token_id = self.pad_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        # 计算固定 span 长度：ceil(sqrt(max_seq_length))
        self.fixed_span_length = math.ceil(math.sqrt(self.max_seq_length))

    def save_pretrained(self, save_directory: str):
        config_dict = asdict(self)
        config_dict.pop('tokenizer', None)
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config_file = path / "config.json"
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
        # GateNet 已内化，不再单独使用

    def _calc_entropy(self, logits: torch.Tensor):
        logits = torch.clamp(logits, min=-50, max=50)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.clamp(entropy, min=0, max=20)
        return entropy

    def forward(self,
                  seq_prompt: torch.Tensor,
                  temperature: float = 1.0,
                  top_p: float = 1.0):
        """
        优化后的 span forward pass：
          - 固定使用 fixed_span_length = ceil(sqrt(max_seq_length))
          - 训练数据的 span 为 fixed_span_length tokens（前 fixed_span_length-1 为真实 token，最后一个固定为 <|TBD|>）
          - 每一步先使用 Decoder 预测 span，再利用 Encoder + Proj Head 对 span 最后一个位置进行预测，
            当预测为 <|TBD|> 或达到 g_max 迭代步数时停止。
        """
        B, n_prompt = seq_prompt.shape
        device = seq_prompt.device
        fixed_span_length = self.config.fixed_span_length

        # 构造初始序列：在 prompt 后追加 fixed_span_length 个 [PRED] 占位符
        pred_tokens = torch.full((B, fixed_span_length), self.config.pred_token_id, dtype=torch.long, device=device)
        seq = torch.cat([seq_prompt, pred_tokens], dim=1)

        # 初始化迭代计数和活动 mask（记录哪些样本仍处于更新中）
        iteration_count = torch.zeros(B, device=device, dtype=torch.long)
        active_mask = torch.ones(B, dtype=torch.bool, device=device)

        # 初始 Encoder 隐状态，根据占位符生成
        z = self.token_emb(pred_tokens)

        for t in range(self.config.g_max):
            if not active_mask.any():
                break

            embeddings = self.token_emb(seq)
            position_ids = torch.arange(seq.shape[1], device=device).unsqueeze(0).expand(B, -1)
            padding_mask = (seq != self.config.pad_token_id).bool()

            h_dec, _ = self.decoder(
                hidden_states=embeddings,
                attention_mask=None,
                position_ids=position_ids,
                padding_mask=padding_mask,
                past_key_values=None,
                use_cache=False
            )
            h_pred_positions = h_dec[:, n_prompt:, :]  # (B, fixed_span_length, hidden)

            # 使用 Proj Head 对 span 最后一个 token 进行预测，判断是否出现 <|TBD|>
            logits_last = self.proj_head(h_pred_positions[:, -1, :]) / temperature
            tokens_last = sample_from_logits(logits_last, temperature, top_p)  # (B,)
            finished = tokens_last == self.config.tbd_token_id
            # 更新活动 mask：未结束的仍为 active
            active_mask = active_mask & (~finished)
            iteration_count[active_mask] += 1

            # 更新整个 span 的预测（仅对活跃样本更新）
            logits_span = self.proj_head(h_pred_positions) / temperature  # (B, fixed_span_length, vocab)
            B_, L, V = logits_span.size()
            logits_span_flat = logits_span.view(-1, V)
            tokens_flat = sample_from_logits(logits_span_flat, temperature, top_p)
            tokens_span = tokens_flat.view(B_, L)
            seq_span = seq[:, n_prompt:]
            seq_span[active_mask] = tokens_span[active_mask]
            seq[:, n_prompt:] = seq_span

            # Encoder 根据当前预测进行隐状态更新
            z = self.encoder.step(self.token_emb(seq[:, n_prompt:]), h_pred_positions.detach(), n_prompt)

        final_logits = self.proj_head(z)
        return {
            'seq': seq,
            'logits_span': final_logits,
            'g_hat': iteration_count  # 每个样本的实际迭代步数
        }
