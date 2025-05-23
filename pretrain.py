import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math

from spanfusionlm.model import SpanFusionLM, SpanFusionLMConfig

def build_batch(raw_data_sample, tokenizer, K, seq_len=2048, device='cpu'):
    """构造训练批次数据"""
    prompt_len = random.randint(max(1, seq_len - K - 50), seq_len - K - 1)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

    # 生成随机prompt
    seq_prompt = torch.randint(4, tokenizer.vocab_size - 100, (1, prompt_len), device=device)
    seq_prompt = torch.cat([
        torch.full((1, 1), bos, device=device, dtype=torch.long),
        seq_prompt,
        torch.full((1, 1), eos, device=device, dtype=torch.long)
    ], dim=1)

    # 生成金标准span
    gold_span = torch.randint(4, tokenizer.vocab_size - 100, (1, K), device=device)

    return seq_prompt, gold_span, K, seq_prompt.shape[1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 配置模型（使用较小配置用于演示）
    config = SpanFusionLMConfig(
        vocab_size=50257,
        hidden_size=512,
        intermediate_size=512 * 2,
        num_decoder_layers=4,
        num_encoder_layers=2,
        num_attention_heads=8,
        max_position_embeddings=512,
    )

    model = SpanFusionLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    K_choices = [8, 16, 24, 32]
    total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model initialized with ~{total_params_m:.2f}M params.")

    # 训练循环
    for step in range(10):
        model.train()

        # 随机选择span长度和迭代步数
        K = random.choice(K_choices)
        g_train = random.randint(1, config.g_max)

        # 构造批次数据
        seq_prompt, gold_span, current_K, prompt_len = build_batch(
            None, config.tokenizer, K,
            seq_len=config.max_position_embeddings - K - 1,
            device=device
        )

        # 前向传播（同时计算teacher和student）
        seq_out = model(
            seq_prompt,
            current_K,
            g=g_train,
            gold_span=gold_span,
            compute_teacher_latent=True
        )

        # === 计算三个损失 ===

        # 1. L_AR：自回归损失
        gold_full_sequence = torch.cat([seq_prompt, gold_span], dim=1)
        gold_input_ids = gold_full_sequence[:, :-1]
        gold_target_ids = gold_full_sequence[:, 1:]

        gold_embeddings = model.token_emb(gold_input_ids)
        ar_pos_ids = torch.arange(gold_input_ids.shape[1], device=device).unsqueeze(0).expand(gold_input_ids.shape[0], -1)
        h_gold_ar, _ = model.decoder(
            hidden_states=gold_embeddings,
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

        # 2. L_latent：潜空间蒸馏损失
        if (model.z_pred_for_loss is not None) and (model.z_teacher_for_loss is not None):
            loss_lat = F.mse_loss(model.z_pred_for_loss, model.z_teacher_for_loss)
        else:
            loss_lat = torch.tensor(0.0, device=device)

        # 3. L_token：span预测损失
        if model.logits_span_for_loss is not None:
            loss_tok = F.cross_entropy(
                model.logits_span_for_loss.reshape(-1, config.vocab_size),
                gold_span.reshape(-1),
                ignore_index=config.pad_token_id
            )
        else:
            loss_tok = torch.tensor(0.0, device=device)

        # 总损失
        loss = loss_ar + 0.5 * loss_lat + 0.5 * loss_tok

        print(f"Step {step}: K={current_K}, g={g_train}, Total Loss: {loss.item():.4f} "
              f"(AR: {loss_ar.item():.4f}, Latent: {loss_lat.item():.4f}, Token: {loss_tok.item():.4f})")

        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # 清理中间变量
        model.z_pred_for_loss = None
        model.z_teacher_for_loss = None
        model.logits_span_for_loss = None

    print("Training loop completed successfully!")

if __name__ == "__main__":
    main()
