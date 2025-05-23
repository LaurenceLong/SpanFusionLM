# SpanFusionLM — 端到端预训练技术方案  

片段级自回归 + 潜空间多步深思 + 动态优先级填充 + RoPE 增量回写  

---

## 1. 总览  

1. 整段文本由 **Span-Decoder**（因果 Transformer）自回归保证时序因果。  
2. 当需要预测接下来 K 个 token 时，先在序列尾部插入 `[PRED]×K` 占位符。  
3. Decoder 对这些占位符给出单步 logits，按“熵低→易”排序得到优先级 τ。  
4. **Span-Encoder** 在隐空间迭代 g 步 (`1≤g≤8`)：  
   • 每步 pick Mₜ 个优先级最高且尚未填充的位置；  
   • 对全部 K 个位置同时 Refinement；  
   • 对 pick 位置一次性投射为真实 token 并写回序列；  
   • 从本轮最左写回点起增量重算 Decoder（RoPE 缓存）以获得下一轮新熵。  
5. 重复步骤 3–4 直至 K 个位置全部填满，外层 while-loop 继续预测下一块。  

---

## 2. 网络组件  

| 模块 | 结构 | 关键张量 |
|------|------|----------|
| TokenEmbedding | weight tying，维度 d | `E` |
| Span-Decoder | Ld 层 causal Transformer，RoPE | logits `(B,L,V)`，hidden `h_dec`，KV 缓存 |
| Span-Encoder | Le 层双向 Transformer，RoPE，自注意+cross-attn(Q=z,K/V=h_dec) | latent `z⁰…zᵍ` `(B,K,d)` |
| ProjHead | `LN → Wᵀ` (W 与 Embedding 共享) | logits_span `(B,K,V)` |
| GateNet | 预训练阶段固定 `g∼Uniform[1,8]` | — |

---

## 3. 前向伪码（单 batch）

```python
def span_fwd(seq_prompt, K, g, theta=10000, temp=1.0):
    B, n = seq_prompt.shape
    PRED = tokenizer.pred_id
    seq = torch.cat([seq_prompt,
                     torch.full((B, K), PRED, device=seq_prompt.device)], dim=1)

    # ── 1. Decoder 全量前向 ───────────────────────
    logits, h_dec, kv = decoder(seq)         # RoPE 内部使用全局 index
    # 熵 = 难度分，priority 越小越容易
    entropy = (-F.softmax(logits[:, -K:], -1)
               *F.log_softmax(logits[:, -K:], -1)).sum(-1)   # (B,K)
    priority = entropy.argsort(dim=-1)

    # ── 2. 初始化 latent z ───────────────────────
    z = tok_emb(seq[:, -K:])                  # (B,K,d) RoPE 将在 enc_step 中应用
    filled = torch.zeros(B, K, dtype=torch.bool, device=seq.device)

    # ── 3. g 步迭代 ─────────────────────────────
    for t in range(g):
        # 3-1 计算本步批量大小 Mₜ（逐步增大）
        M_t = math.ceil((t+1)*K/g) - math.ceil(t*K/g)
        pick = topM(priority, filled, M_t)    # Bool mask (B,K)

        # 3-2 Encoder latent refinement（K 个位置全部更新）
        z = encoder.step(z, h_dec, t)         # RoPE 内置 (prefix_len + idx)

        # 3-3 投射 pick 位置并写回
        logits_pick = proj(z[pick]) / temp
        tok_hat     = top_p_sample(logits_pick, 0.9)         # (Tot_pick,)
        seq[:, -K:][pick] = tok_hat
        filled[pick] = True

        # 3-4 非最后一步：增量重算 Decoder & priority
        if t != g-1:
            # pick 中最左全局下标
            first = (pick.nonzero()[:,1].reshape(B,-1).min(-1).values).min().item()
            logits_delta, h_dec, kv = decoder(
                    seq,
                    kv_cache=kv,
                    start_pos=seq.size(1)-K+first)          # RoPE 重新旋转
            logits[:, -K+first:] = logits_delta
            rem_mask = ~filled
            entropy[rem_mask] = (-F.softmax(
                  logits_delta[rem_mask],-1)
                  *F.log_softmax(logits_delta[rem_mask],-1)).sum(-1)
            priority = entropy.argsort(dim=-1)

    return seq
```

要点  
• `encoder.step` 内部对 **全部 K 位置** 执行 Self-Attn / Cross-Attn，因此早选槽位会经历更多 Refinement。  
• 每次 Decoder 重算的区间为 `[first_changed, end)`，随着步骤推进单调变短。  

---

## 4. 损失定义  

```
L_total = L_AR + 0.5 · L_latent + 0.5 · L_token
```

1. L_AR：对完整金序列做自回归交叉熵。  
2. L_latent：  
   • 复制 batch，使用 g_max=8 得到 teacher z★  
   • `L₂ = ‖zᵍ − stop_grad(z★)‖²`（对所有 K 位置平均）  
3. L_token：`CE(Softmax(proj(zᵍ)), gold_span)`  

梯度流向  
• L_AR → TokenEmbedding + Decoder  
• L_latent → Encoder  
• L_token → Encoder + ProjHead + TokenEmbedding  

---

## 5. 预训练配置  

| 项 | 值 |
|----|----|
| 模型 | d=4096, n_head=32, Ld=28, Le=8 (~7B) |
| 词表 | 32 k SentencePiece + byte fallback |
| Span 长度 K | 8,16,24,32 均匀采样 |
| g 步 | 1…8 均匀采样 |
| 序列长 | 2 048 |
| 数据 | 2 T 混合文本（代码 20 %） |
| Optim | AdamW β=(0.9,0.95), lr=3e-4 → cosine decay |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| AMP | bfloat16 |
| Batch | 4 096 tok/GPU × 1 024 GPU |
| 训练步 | 300 k (~14 B token) |

---

## 6. 训练循环示例

```python
for step, raw in enumerate(loader):
    seq_prompt, gold_span, K = build_batch(raw)
    g = random.randint(1,8)

    seq_out = span_fwd(seq_prompt, K, g)

    # losses
    loss_ar  = ce(seq_out[:, :-K], seq_prompt[:, 1:])
    loss_lat = ((model.z_pred - model.z_teacher.detach())**2).mean()
    loss_tok = ce(model.logits_span, gold_span)

    loss = loss_ar + 0.5*loss_lat + 0.5*loss_tok
    loss.backward()
    opt.step(); opt.zero_grad()
```

---

## 7. RoPE 实现要点  

1. 绝对 index = 已生成长度 + span 内 offset。  
2. Decoder 重算时再取 `rope_cache[start_pos:]`，左侧 KV 复用。  
3. Encoder 每步重新 `apply_rope(z, pos_global)`，保证与最新 h_dec 对齐。  

---

## 8. 推理接口

```python
def generate(prompt, max_new=1024,
             K_fn=lambda rest: 16,
             g=4, temperature=0.8, top_p=0.9):
    seq = tokenizer(prompt).unsqueeze(0).to(device)
    while len(seq[0]) - prompt_len < max_new:
        K = K_fn(max_new - (len(seq[0]) - prompt_len))
        seq = span_fwd(seq, K, g, temp)           # 单批
    return detokenize(seq[0])
```

---

## 9. 正确性与效率检查清单  

1. g=1, K=8 时输出与纯 GPT 一致。  
2. 对同一输入，g=4 时每步 `first_changed` 单调递增。  
3. 写回后重复覆盖同 index（先错后对）→ 最终 logits 与直接写正确 token 相同。  
4. A100-80G, 7B, g=4, K=16：吞吐 ≥110 tok/s。  

---

## 10. 工程目录

```
SpanFusionLM/
├─ modules/
│   ├─ token_emb.py
│   ├─ rope.py
│   ├─ decoder.py
│   ├─ encoder.py
│   └─ proj_head.py
├─ model.py          # glue SpanFusionLM
├─ pretrain.py       # main script
├─ data/
└─ infer.py
```

---

### 结语  

SpanFusionLM 在保持自回归因果的同时，通过“熵排序 + 潜空间多步 Refinement + 增量 RoPE 解码”实现块内高质量补全；重要 token 被多次思考、简单 token 快速确定，计算成本仅比同规模 GPT 增长约 20 %，适合直接落地大规模预训练。

python -m spanfusionlm.pretrain # For training example
python -m spanfusionlm.infer    # For inference example