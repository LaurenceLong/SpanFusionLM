# SpanFusionLM — 端到端预训练技术方案

**核心思想**: 通过一个标准的自回归 Transformer (Decoder) 生成初步序列，并引入一个迭代式的细化过程。此过程包含一个 Encoder 对特定片段 (span) 的隐状态进行多步优化，以及一个 GateNet 动态决定每个样本的细化迭代次数。填充策略基于对数几率的熵，优先填充模型最不确定的 token 位置。

---

## 1. 总览

1.  **初始序列构建**: 给定一个 `seq_prompt`，在其后拼接 `K` 个特殊的 `[PRED]` (预测占位符) token，形成初始输入序列 `seq`。
2.  **迭代式细化 (`span_forward_pass`)**:
    *   **最大迭代次数**: 预设一个最大迭代步数 `g_max` (来自配置 `config.g_max`)。
    *   **Decoder 前向**: 在每次迭代 `t` 开始时，对当前整个序列 `seq` (包含已部分填充的 span) 进行完整的 **Span-Decoder** 前向传播，得到 `K` 个预测位置的隐状态 `h_pred_positions` 和对应的 `logits_pred`。
    *   **熵计算与位置选择**: 根据 `logits_pred` 计算每个预测位置的熵。对于当前仍在活跃处理 (`active_mask`) 的样本，选择 `M_step = ceil(K / g_max)` 个熵最高 (最不确定) 且尚未被填充的位置 (`pick_mask`) 进行更新。
    *   **Encoder 细化**: **Span-Encoder** 的 `step` 方法接收上一轮的 span 隐状态 `z` 和当前 Decoder 输出的 `h_pred_positions`，对 *所有* `K` 个位置的 `z` 进行细化。
    *   **Token 采样与回写**: 从细化后的 `z` 中，根据 `pick_mask` 选出对应位置的隐状态，通过 **ProjectionHead** 得到 logits，然后进行采样 (训练时教师路径用 `argmax`，学生路径用 `sample_from_logits`) 得到 `tokens_picked`。将这些新 token 写回到 `seq` 中的相应位置，并更新 `filled_mask`。
    *   **GateNet 决策**:
        *   计算当前 `K` 个预测位置的平均熵 `avg_entropy`。
        *   **GateNet** (一个小型 MLP) 以 `avg_entropy` 为输入，输出一个二分类的 logits (继续迭代 vs. 停止迭代)。
        *   根据 GateNet 的决策 (训练时用 Gumbel-Softmax，评估/教师路径用 `argmax`) 更新 `active_mask`。如果一个样本的决策是停止，则它在后续迭代中不再更新。
        *   记录每个样本实际的迭代次数 `iteration_count` (即 `g_hat`)。
    *   **循环终止**: 当所有样本都停止迭代 (`active_mask` 全为 false) 或达到 `g_max` 步时，循环结束。
3.  **输出**: 返回最终生成的序列 `seq`，学生路径下最后一步的隐状态 `final_z` 和 `final_logits`，GateNet 的概率 `p_g` (二分类的 softmax)，以及每个样本的实际迭代步数 `g_hat`。
4.  **教师路径与学生路径**:
    *   **学生路径 (`is_teacher_path=False`)**: 正常执行上述流程，使用 `sample_from_logits` 进行采样，GateNet 在训练时使用 Gumbel-Softmax。其最终的 `z` (`z_pred_for_loss`) 和 `logits` (`logits_span_for_loss`) 用于损失计算。
    *   **教师路径 (`is_teacher_path=True`)**: 在 `torch.no_grad()` 上下文中执行，采样时使用 `argmax`，GateNet 使用 `argmax` 决策。其最终的 `z` (`z_teacher_for_loss`) 被 `detach()` 并保存，用作学生路径潜空间损失的监督信号。

---

## 2. 网络组件

| 模块 | 结构 | 关键张量/行为 |
|------|------|----------|
| TokenEmbedding | `nn.Embedding`，权重可与 ProjHead 共享 | `(vocab_size, hidden_size)` |
| Span-Decoder | `Ld` 层因果 Transformer (GPT-like)，使用 RoPE | 输入 `(B, L, d)`，输出 `h_dec (B, L, d)`。在 `span_forward_pass` 的每次迭代中，对整个当前序列进行完整前向。 |
| Span-Encoder | `Le` 层双向 Transformer，包含自注意力和交叉注意力 (Q=z, K/V=h_dec 来自 Decoder 的对应位置)。其 `step` 方法用于迭代更新 span 的隐状态 `z`。 | 输入 `z (B, K, d)`, `h_pred_positions (B, K, d)`，输出更新后的 `z (B, K, d)`。 |
| ProjectionHead | `Linear(hidden_size, vocab_size)`，权重可与 TokenEmbedding 共享 | 输入 `(..., d)`，输出 `logits (..., vocab_size)` |
| GateNet | 小型 MLP (Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → ReLU → Dropout → Linear)，输出 2 个 logits (停止/继续)。 | 输入 `avg_entropy (B, 1)`，输出 `decision (B,)` (0=停止, 1=继续) 和 `gate_logits (B, 2)`。训练时使用 Gumbel-Softmax 采样决策，评估时用 `argmax`。可通过 `override` 方法在推理时改变其行为。 |

---

## 3. 前向核心逻辑 (`span_forward_pass` 简化伪码 - 贴合实现)

```python
# config: SpanFusionLMConfig
# self: SpanFusionLM instance

def span_forward_pass(seq_prompt, K, temperature, top_p, is_teacher_path):
    B, n_prompt = seq_prompt.shape
    device = seq_prompt.device
    PRED_ID = self.config.pred_token_id
    g_max = self.config.g_max

    # 初始化 GateNet override (训练时设为 None)
    if self.training: self.gate_net.override(None)

    # 初始序列: prompt + K 个 [PRED]
    pred_tokens = torch.full((B, K), PRED_ID, dtype=torch.long, device=device)
    seq = torch.cat([seq_prompt, pred_tokens], dim=1)

    filled_mask = torch.zeros(B, K, dtype=torch.bool, device=device) # 记录已填充位置
    iteration_count = torch.zeros(B, device=device, dtype=torch.long) # 实际迭代次数 g_hat
    active_mask = torch.ones(B, dtype=torch.bool, device=device) # 哪些样本在迭代

    # 每步固定更新 M_step 个 token (向上取整)
    M_step = (K + g_max - 1) // g_max

    # 初始化隐状态 z 为 [PRED] token 的嵌入
    z = self.token_emb(pred_tokens)
    final_z, final_logits, gate_logits_output = None, None, None

    for t in range(g_max):
        if not active_mask.any(): break

        # 1. Decoder 全量前向 (对当前整个 seq)
        embeddings = self.token_emb(seq)
        pos_ids = torch.arange(seq.shape[1], device=device).unsqueeze(0)
        h_dec, _ = self.decoder(embeddings, position_ids=pos_ids, use_cache=False)
        h_pred_positions = h_dec[:, n_prompt:, :] # 取 K 个 PRED 位置的隐状态
        logits_pred = self.proj_head(h_pred_positions)
        entropy = self._calc_entropy(logits_pred) # (B, K)

        # 2. 确定本轮要更新的 token 数量 M_t (active 样本为 M_step, inactive 为 0)
        M_t_per_sample = torch.where(active_mask, M_step, 0)
        pick_mask = self._select_top_m_positions(entropy, filled_mask, M_t_per_sample) # (B,K) bool

        # 3. Encoder latent refinement (对所有 K 个位置的 z)
        #    encoder.step 内部处理 RoPE 等细节
        z = self.encoder.step(z, h_pred_positions.detach(), n_prompt)

        # 4. 投射 pick 位置并写回序列
        if pick_mask.any():
            z_picked = z[pick_mask] # (num_picked, d)
            logits_picked = self.proj_head(z_picked)
            if is_teacher_path:
                tokens_picked = torch.argmax(logits_picked, dim=-1)
            else:
                tokens_picked = sample_from_logits(logits_picked, temperature, top_p)

            # 更新 seq 和 filled_mask
            seq_span_view = seq[:, n_prompt:]
            seq_span_view[pick_mask] = tokens_picked
            seq[:, n_prompt:] = seq_span_view
            filled_mask = filled_mask | pick_mask

        # 保存最后一次迭代的 z 和 logits (用于可能的损失计算或最终输出)
        if t == g_max - 1:
            final_z = z.clone()
            final_logits = self.proj_head(z.clone()) # 使用更新后的 z

        # 5. GateNet 决策是否继续迭代
        avg_entropy = entropy.mean(dim=-1, keepdim=True) # (B, 1)
        # GateNet 训练时: train=self.training AND not is_teacher_path
        # GateNet 教师路径/推理时: train=False
        decision, gate_logits = self.gate_net(avg_entropy,
                                              train=(self.training and not is_teacher_path))
        gate_logits_output = gate_logits # 保存用于返回

        new_active = active_mask.clone()
        new_active[active_mask] = (decision[active_mask] == 1) # 1=继续, 0=停止
        iteration_count[active_mask] += 1 # 仍在迭代的样本，计数+1
        active_mask = new_active

    # 保存用于损失计算的特定隐状态
    if is_teacher_path:
        self.z_teacher_for_loss = z.detach().clone() # 使用循环结束时的 z
    else:
        self.z_pred_for_loss = z.clone() # 使用循环结束时的 z
        self.logits_span_for_loss = self.proj_head(z) # 使用循环结束时的 z 计算 logits

    return {
        'seq': seq,
        'z_pred': final_z if not is_teacher_path else None, # 学生路径下最后一步的 z
        'logits_span': final_logits if not is_teacher_path else None, # 学生路径下最后一步的 logits
        'p_g': F.softmax(gate_logits_output, dim=-1) if gate_logits_output is not None else None,
        'g_hat': iteration_count # 每个样本的实际迭代步数
    }
```

---

## 4. 损失定义 (`compute_losses` 函数)

```
L_total = 0.4·L_AR + 0.2·L_latent + 0.4·L_token + β_cost_g·E[g_hat]
```

1.  **L_AR (Autoregressive Loss)**:
    *   构建金标准完整序列 `gold_full_seq = torch.cat([seq_prompt, gold_span], dim=1)`.
    *   对 `gold_full_seq` (取 `gold_full_seq[:, :-1]` 为输入, `gold_full_seq[:, 1:]` 为目标) 计算标准的交叉熵损失。
    *   通过 `unwrapped_model.token_emb`, `unwrapped_model.decoder`, `unwrapped_model.proj_head` 计算。
    *   梯度流向: `TokenEmbedding`, `Span-Decoder`, `ProjectionHead`.

2.  **L_latent (Latent Space Consistency Loss)**:
    *   学生路径 (`span_forward_pass` with `is_teacher_path=False`) 得到 `z_pred_for_loss`.
    *   教师路径 (`span_forward_pass` with `is_teacher_path=True`, `no_grad()`, argmax sampling) 得到 `z_teacher_for_loss`.
    *   `z_s = LayerNorm(z_pred_for_loss)`, `z_t = LayerNorm(z_teacher_for_loss)`.
    *   `L_latent = 0.5 * (1 - F.cosine_similarity(z_s, z_t, dim=-1)).mean()`.
    *   梯度流向: `TokenEmbedding` (间接通过 `z_pred_for_loss` 的初始值), `Span-Encoder` (主要目标).

3.  **L_token (Token Prediction Loss for Span)**:
    *   使用学生路径在 `span_forward_pass` 结束时保存的 `logits_span_for_loss` (即 `proj_head(z_pred_for_loss)`).
    *   `L_token = CE(logits_span_for_loss.reshape(-1, vocab_size), gold_span.reshape(-1))`.
    *   梯度流向: `TokenEmbedding`, `Span-Encoder`, `ProjectionHead`.

4.  **E[g_hat] (Expected Iteration Cost)**:
    *   `g_hat` 是学生路径 `span_forward_pass` 返回的 `iteration_count` (每个样本的实际迭代次数)。
    *   `E[g_hat] = student_out['g_hat'].float().mean()`.
    *   `β_cost_g` 是一个动态调整的系数 (curriculum: `min(0.05, completed_steps / 30000 * 0.05)`).
    *   梯度流向: `GateNet` (通过 Gumbel-Softmax 的 `hard=True` 和 `logits` 传递梯度), `TokenEmbedding`, `Span-Decoder`, `ProjectionHead` (间接通过影响平均熵输入到 GateNet).

---

## 5. 预训练配置 (基于 `pretrain.py` 的 `args`)

| 项 | 值 (来自 `pretrain.py` 默认值) | 备注 |
|----|----|---|
| **模型参数** | | |
| `tokenizer_name` | "gpt2" |  |
| `hidden_size` | 768 | GPT-2 small 级别 |
| `intermediate_size` | 3072 | GPT-2 small 级别 |
| `num_decoder_layers` | 12 | GPT-2 small 级别 |
| `num_encoder_layers` | 6 | 约为 Decoder 层数一半 |
| `num_attention_heads` | 12 | GPT-2 small 级别 |
| `rope_theta` | 10000.0 |  |
| `g_max` | 4 | 最大 GateNet 迭代步数 |
| `vocab_size` | 根据 `tokenizer` 动态确定 |  |
| `max_position_embeddings` | `args.max_seq_length + max(args.span_lengths)` | 例如 512 + 16 = 528 |
| **数据参数** | | |
| `dataset_name` | "wikitext" |  |
| `dataset_config_name` | "wikitext-103-raw-v1" |  |
| `max_seq_length` | 512 | Prompt + Span 的最大长度，不含 K |
| `span_lengths_str` | "8,16" | 训练时随机选择 K 值 |
| **训练参数** | | |
| `learning_rate` | 5e-5 |  |
| `weight_decay` | 0.01 |  |
| `adam_beta1` | 0.9 |  |
| `adam_beta2` | 0.99 |  |
| `grad_clip` | 1.0 |  |
| `beta_cost_g` | 0.01 (初始值) | 逐步增大的成本系数，见损失定义 |
| `mixed_precision` | "fp16" |  |
| `batch_size_per_device` | 16 |  |
| `gradient_accumulation_steps` | 4 |  |
| `num_train_epochs` | 1 |  |
| `max_train_steps` | 1000 |  |
| `lr_scheduler_type` | "linear" |  |
| `warmup_steps` | 100 |  |

---

## 6. 训练循环关键步骤 (`pretrain.py main` 函数)

1.  **初始化**: Accelerator, logging, W&B, tokenizer, model config, model, optimizer, dataset, DataLoader, LR scheduler.
2.  **数据准备 (`collate_fn`)**:
    *   从样本中取文本，编码。
    *   随机选择一个 `current_K` from `args.span_lengths`.
    *   随机切分 `prompt_len`，构造 `seq_prompt_toks` 和 `gold_span_toks` (长度为 `current_K`).
    *   Padding 后返回 `padded_seq_prompts`, `padded_gold_spans`.
3.  **训练迭代**:
    *   For each batch (`seq_prompt`, `gold_span`):
        *   `current_K = gold_span.shape`.
        *   `compute_losses(model, seq_prompt, gold_span, current_K, model_config, accelerator)`:
            *   内部调用 `model.forward(seq_prompt, current_K, gold_span=gold_span, compute_teacher_latent=True)`.
            *   `model.forward` 调用 `span_forward_pass` 一次作为学生路径。
            *   如果 `compute_teacher_latent=True`，则在 `torch.no_grad()` 下再次调用 `span_forward_pass` 作为教师路径。
            *   从 `unwrapped_model` 中获取 `z_pred_for_loss`, `z_teacher_for_loss`, `logits_span_for_loss`。
            *   计算 `L_AR`, `L_latent`, `L_token`, `E[g_hat]`。
        *   计算 `total_loss` (加权和，`beta_cost_g` 动态调整)。
        *   `accelerator.backward(total_loss)`.
        *   梯度同步、裁剪、优化器步进、学习率调度、梯度清零。
        *   Logging, saving checkpoints.

---

## 7. RoPE 实现要点 (根据代码结构推断)

1.  **绝对位置**: RoPE 编码基于 token 在完整序列中的绝对位置。`position_ids` 在 `SpanDecoder` 和 `SpanEncoder` (内部) 的前向传播中生成并传递给注意力模块，RoPE 在注意力模块内部应用。
2.  **Decoder**: 在 `span_forward_pass` 的每次迭代 `t` 中，`SpanDecoder` 对整个当前序列 `seq` (长度 `n_prompt + K`) 进行操作。`position_ids` 会从 `0`到 `seq.shape-1`。由于 `use_cache=False`，每次都是从头计算，不涉及 KV 缓存的增量更新。
3.  **Encoder**: `SpanEncoder` 的 `step(z, h_pred_positions, n_prompt)` 方法接收 `h_pred_positions` (来自 Decoder 在 `n_prompt` 到 `n_prompt+K` 位置的输出) 和当前的 `z`。`Encoder` 内部应用 RoPE 时，其 `position_ids` 应该是相对于 span 在完整序列中的全局位置 (即 `n_prompt` 到 `n_prompt+K-1`)，以确保与 `h_pred_positions` 的 RoPE 应用方式对齐。

---

## 8. 推理接口 (`infer.py` - 示意，与方案对齐)

```python
# infer.py (概念对齐，实际代码可能更复杂)
# from .model import SpanFusionLM, SpanFusionLMConfig
# from .modules.tokenizer import build_tokenizer

# def generate(prompt_text, max_new_tokens, K_value, model, tokenizer, config,
#              temperature=0.8, top_p=0.9, gate_override_mode=None):
#     seq_prompt = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
#     generated_sequence = seq_prompt.clone()
#
#     # GateNet override logic based on gate_override_mode
#     # Example: if gate_override_mode == 'fast':
#     # model.gate_net.override(lambda entropy_feature: torch.tensor([[0.0, 1.0]] * entropy_feature.shape[0], device=model.device)) # Force continue (effectively g=1 if M_step fills K)
#     # elif gate_override_mode == 'accurate':
#     # model.gate_net.override(lambda entropy_feature: torch.tensor([[1.0, 0.0]] * entropy_feature.shape[0], device=model.device)) # Force stop after 1st (effectively g=g_max if M_step is small)
#     # else: model.gate_net.override(None) # Use learned GateNet
#
#     current_len = seq_prompt.shape[1]
#     target_len = current_len + max_new_tokens
#
#     while current_len < target_len:
#          K_to_generate = min(K_value, target_len - current_len)
#          if K_to_generate == 0: break
#
#         # Call the model's forward or span_forward_pass directly for inference
#         # Note: In actual infer.py, it would call model.generate or a similar high-level method
#         # which internally calls span_forward_pass.
#         # For this alignment, assume span_forward_pass is used:
#         output_dict = model.span_forward_pass(
#             generated_sequence,
#             K_to_generate,
#             temperature=temperature,
#             top_p=top_p,
#             is_teacher_path=False # Inference is like student path but with argmax/sampling
#         )
#         # The 'seq' from output_dict contains the prompt + K generated tokens
#         # We need to handle how 'seq' is formed if K_to_generate < K capacity of span_forward_pass
#         # Assuming output_dict['seq'] is [B, L_prompt + K_generated]
#         generated_sequence = output_dict['seq']
#         current_len = generated_sequence.shape[1]
#
#     return tokenizer.decode(generated_sequence[0])

```
**注**: `infer.py` 的实际 `generate` 函数会更复杂，处理 token 拼接、EOS 等。上述伪码仅为对齐 `span_forward_pass` 和 `GateNet.override` 的概念。代码中 `GateNet.override(None)` 会在 `span_forward_pass` 开始时被调用（如果 `self.training`），推理时需要手动设置。

---

## 9. 关键代码点与方案对应

1.  **动态迭代次数**: `model.py` 的 `span_forward_pass` 中的 `active_mask` 和 `iteration_count`，以及 `gate_net.py` 的二分类输出 (`decision`) 完全体现了动态迭代。`g_hat` (即 `iteration_count`) 用于计算 `E[g_hat]` 损失。
2.  **固定数量 M_step 更新**: `M_step = (K + max_iters - 1) // max_iters` 在 `span_forward_pass` 中定义，`_select_top_m_positions` 使用 `M_t` (基于 `M_step` 和 `active_mask`) 来选择位置。
3.  **Decoder 全量计算**: `span_forward_pass` 循环内，`self.decoder(...)` 每次都对完整 `seq` 操作，`use_cache=False`。
4.  **损失函数权重与公式**: `pretrain.py` 的 `compute_losses` 中的权重 (0.4 AR, 0.2 Latent, 0.4 Token) 和 Latent Loss 的余弦相似度公式 (`0.5 * (1 - F.cosine_similarity(z_s, z_t, dim=-1)).mean()`) 已对齐。
5.  **教师路径行为**: `span_forward_pass` 中 `is_teacher_path=True` 时，采样为 `argmax`，GateNet 的 `train` 参数为 `False` (导致 `argmax` 决策)。`z_teacher_for_loss` 被 `detach()`。
6.  **配置参数**: 方案中的配置值已更新为 `pretrain.py` 中的 `argparse` 默认值。

---

### 结语 (修正后)

SpanFusionLM 通过在标准的自回归 Decoder 之上叠加一个迭代细化模块 (Span-Encoder 和 GateNet)，实现了对预测片段的深度处理。GateNet 基于当前预测的平均不确定性（熵）动态地为每个样本决定迭代细化的次数，使得模型可以将更多计算资源用于困难的预测部分。通过熵指导的优先级填充策略，模型优先更新最不确定的 token。损失函数结合了自回归、片段 token 预测、潜空间一致性以及迭代成本控制，旨在平衡生成质量与计算效率。该方案已根据提供的工程实践代码进行了精确对齐。
export HF_ENDPOINT=https://hf-mirror.com  # hf mirror 
python -m SpanFusionLM.pretrain   # 训练示例  
python -m SpanFusionLM.infer      # 推理示例