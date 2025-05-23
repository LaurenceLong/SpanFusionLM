import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math

from spanfusionlm.model import SpanFusionLM, SpanFusionLMConfig
from spanfusionlm.modules.tokenizer import Gpt2Tokenizer

def build_batch(raw_data_sample, tokenizer, K, seq_len=2048, device='cpu'):
    """
    构造一个 dummy batch，返回：
      seq_prompt, gold_span, K, prompt_len（实际 prompt 长度）
    保证：prompt_len + K < seq_len
    """
    prompt_len = random.randint(max(1, seq_len - K - 50), seq_len - K - 1)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    seq_prompt = torch.randint(4, tokenizer.vocab_size, (1, prompt_len), device=device)
    seq_prompt = torch.cat([
        torch.full((1, 1), bos, device=device, dtype=torch.long),
        seq_prompt,
        torch.full((1, 1), eos, device=device, dtype=torch.long)
    ], dim=1)
    gold_span = torch.randint(4, tokenizer.vocab_size, (1, K), device=device)
    return seq_prompt, gold_span, K, seq_prompt.shape[1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = SpanFusionLMConfig(
        vocab_size=32000,
        hidden_size=512,  # 示例使用较小模型
        intermediate_size=512 * 2,
        num_decoder_layers=4,
        num_encoder_layers=2,
        num_attention_heads=8,
        max_position_embeddings=512,
        g_max=8,
        tokenizer=Gpt2Tokenizer()
    )
    model = SpanFusionLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    K_choices = [8, 16, 24, 32]
    beta = 0.02  # 成本系数

    total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model initialized with ~{total_params_m:.2f}M params.")

    for step in range(10):
        model.train()
        K = random.choice(K_choices)
        # 当外部不传入 g 时，由 GateNet 根据 entropy 动态确定
        g_train = None
        seq_prompt, gold_span, current_K, prompt_len = build_batch(
            None, config.tokenizer, K, seq_len=config.max_position_embeddings - K - 1, device=device
        )
        # 学生路径 forward，同时计算 teacher latent（compute_teacher_latent=True）
        out = model(
            seq_prompt,
            current_K,
            g=g_train,
            gold_span=gold_span,
            compute_teacher_latent=True
        )
        # L_AR：对完整 gold 序列做自回归交叉熵计算
        gold_full_sequence = torch.cat([seq_prompt, gold_span], dim=1)
        gold_input_ids = gold_full_sequence[:, :-1]
        gold_target_ids = gold_full_sequence[:, 1:]
        gold_embeddings = model.token_emb(gold_input_ids)
        ar_pos_ids = torch.arange(gold_input_ids.shape[1], device=device).unsqueeze(0).expand(gold_input_ids.shape[0], -1)
        h_gold_ar, _ = model.decoder(
            hidden_states=gold_embeddings,
            attention_mask=None,
            position_ids=ar_pos_ids,
            past_key_values=None,
            use_cache=False
        )
        logits_ar = model.proj_head(h_gold_ar)
        loss_ar = F.cross_entropy(
            logits_ar.reshape(-1, config.vocab_size),
            gold_target_ids.reshape(-1),
            ignore_index=config.pad_token_id
        )
        # L_latent：学生与 teacher latent 的 L2 loss（K 位置平均）
        if (model.z_pred_for_loss is not None) and (model.z_teacher_for_loss is not None):
            loss_lat = F.mse_loss(model.z_pred_for_loss, model.z_teacher_for_loss)
        else:
            loss_lat = torch.tensor(0.0, device=device)
        # L_token：通过学生在 span 上计算 CE loss
        if model.logits_span_for_loss is not None:
            loss_tok = F.cross_entropy(
                model.logits_span_for_loss.reshape(-1, config.vocab_size),
                gold_span.reshape(-1),
                ignore_index=config.pad_token_id
            )
        else:
            loss_tok = torch.tensor(0.0, device=device)
        # 计算期望步数成本：E[g] = (p_g * [1,2,…,g_max]).sum(-1).mean()
        if out['p_g'] is not None:
            g_range = torch.arange(1, config.g_max + 1, device=seq_prompt.device).float()
            exp_g = (out['p_g'] * g_range).sum(dim=-1).mean()
        else:
            exp_g = 0.0
        loss = loss_ar + 0.5 * loss_lat + 0.5 * loss_tok + beta * exp_g

        print(f"Step {step}: K={current_K}, g={'dynamic' if g_train is None else g_train}, Total Loss: {loss.item():.4f} "
              f"(AR: {loss_ar.item():.4f}, Latent: {loss_lat.item():.4f}, Token: {loss_tok.item():.4f}, exp_g: {exp_g if isinstance(exp_g, float) else exp_g.item():.4f})")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.z_pred_for_loss = None
        model.z_teacher_for_loss = None
        model.logits_span_for_loss = None

    print("Training loop example finished.")

if __name__ == "__main__":
    main()
