# SpanFusionLM — 端到端预训练技术方案  

片段级自回归 + 潜空间多步深思 + 动态优先级填充 + RoPE 增量回写  

---

## 1. 总览  

1. 整段文本由 **Span-Decoder**（因果 Transformer）自回归保证时序因果。  
2. 当需要预测接下来 K 个 token 时，先在序列尾部插入 `[PRED]×K` 占位符。  
3. Decoder 对这些占位符给出单步 logits，并根据熵值 `H` 将位置按“熵低→易”排序得到优先级 τ。  
4. **GateNet** 读取熵向量后，为每个样本预测一个离散步数 `ĝ∈{1,…,g_max}`，代表本 span 需要的 Refinement 次数。  
5. **Span-Encoder** 在隐空间迭代至多 `g_max=8` 步：  
   • 第 t 步对每个样本选择 `Mₜ = ⌈(t+1)·K/ĝ⌉ − ⌈t·K/ĝ⌉` 个优先级最高且尚未填充的位置；  
   • 对全部 K 个位置同时 Refinement；  
   • 将 pick 槽位一次性投射为真实 token 并写回序列；  
   • 从本轮最左写回点起增量重算 Decoder（RoPE 缓存）以获得下一轮新熵；  
   • 若某样本已达其 ĝ，则后续步对其跳过。  
6. 当 K 个位置全部填满，外层 while-loop 继续生成下一块。  

---

## 2. 网络组件  

| 模块 | 结构 | 关键张量 |
|------|------|----------|
| TokenEmbedding | 权重共享，维度 d | `E` |
| Span-Decoder | Ld 层 causal Transformer，RoPE | logits `(B,L,V)`，hidden `h_dec`，KV 缓存 |
| Span-Encoder | Le 层双向 Transformer，RoPE，自注意 + cross-attn(Q=z,K/V=h_dec) | latent `z⁰…zᵍ` `(B,K,d)` |
| ProjHead | `LN → Wᵀ` (W 与 Embedding 共享) | logits_span `(B,K,V)` |
| GateNet | `MeanPool(H) → 2-layer MLP → Gumbel-Softmax` | `p_g (B,g_max)`, `ĝ (B,)` |

---

## 3. 前向伪码（单 batch）

```python
g_max, beta = 8, 0.02          # beta 是计算成本系数

def calc_entropy(logits):      # logits: (B,K,V)
    p = torch.softmax(logits, -1)
    return (-p * torch.log(p + 1e-7)).sum(-1)      # (B,K)

def span_fwd(seq_prompt, K, theta=10000, temp=1.0, train=True):
    B, n = seq_prompt.shape
    PRED = tokenizer.pred_id
    seq = torch.cat([seq_prompt,
                     torch.full((B, K), PRED, device=seq_prompt.device)], 1)

    # ── 1. Decoder 全量前向 ───────────────────────
    logits, h_dec, kv = decoder(seq)           # RoPE 内部使用全局 index
    entropy = calc_entropy(logits[:, -K:])     # (B,K)

    # ── 2. GateNet 预测 ĝ ───────────────────────
    p_g = GateNet(entropy.mean(-1))            # (B,g_max)
    if train:
        # Gumbel-Softmax 可微采样，hard=True 得到 one-hot
        g_hot = F.gumbel_softmax(p_g, tau=1.0, hard=True)
        g_real = (g_hot * torch.arange(1, g_max+1, device=seq.device)).sum(-1).long()  # (B,)
    else:
        g_real = p_g.argmax(-1) + 1            # (B,)

    g_loop = g_real.max().item()               # for-loop 跑到 batch 最大 ĝ

    # ── 3. 初始化 latent z ───────────────────────
    z = tok_emb(seq[:, -K:])                   # (B,K,d) 先不打 RoPE
    filled = torch.zeros(B, K, dtype=torch.bool, device=seq.device)

    # ── 4. g_max 步迭代 ────────────────────────
    for t in range(g_loop):
        active = g_real > t                    # (B,) 本步仍需迭代的样本
        if not active.any(): break

        # 4-1 计算本步批量大小 Mₜ
        M_t = torch.ceil((t+1)*K/g_real) - torch.ceil(t*K/g_real)   # (B,)
        pick = torch.zeros_like(filled)        # (B,K) bool

        # 4-2 选槽位
        for b in torch.nonzero(active).flatten():
            m = int(M_t[b].item())
            cand = (~filled[b]).nonzero().flatten()
            idx = cand[entropy[b, cand].argsort()[:m]]
            pick[b, idx] = True

        # 4-3 Encoder latent refinement
        z = encoder.step(z, h_dec, t)          # 所有 K 位置更新

        # 4-4 投射 pick 位置并写回
        logits_pick = proj(z[pick]) / temp
        tok_hat = top_p_sample(logits_pick, 0.9)
        seq[:, -K:][pick] = tok_hat
        filled[pick] = True

        # 4-5 非最后一步：增量重算 Decoder & priority
        if t != g_loop-1:
            first = (pick.nonzero()[:,1].reshape(B,-1).min(-1).values).min().item()
            logits_delta, h_dec, kv = decoder(
                    seq, kv_cache=kv, start_pos=seq.size(1)-K+first)
            logits[:, -K+first:] = logits_delta
            rem_mask = ~filled
            entropy[rem_mask] = calc_entropy(logits_delta)[rem_mask]

    return dict(seq=seq,
                z_pred=z,
                logits_span=proj(z),
                p_g=p_g)
```

---

## 4. 损失定义  

```
L_total = L_AR + 0.5·L_latent + 0.5·L_token + β·E[g]
```

1. L_AR：对完整金序列做自回归交叉熵。  
2. L_latent：  
   • 复制 batch，令 teacher 采用 `g_max=8` 得到 `z★`；  
   • 令 `L_latent = ‖z_pred − stop_grad(z★)‖²`（K 位置平均）。  
3. L_token：`CE(Softmax(logits_span), gold_span)`。  
4. 期望步数 `E[g] = (p_g * [1,2,…,g_max]).sum(-1).mean()`。  

梯度流向  
• L_AR → TokenEmbedding + Decoder  
• L_latent → Encoder  
• L_token → Encoder + ProjHead + TokenEmbedding  
• E[g]   → GateNet（经 Softmax 可微）  

---

## 5. 预训练配置  

| 项 | 值 |
|----|----|
| 模型 | d=4096, n_head=32, Ld=28, Le=8 (~7 B) |
| 词表 | 32 k SentencePiece + byte fallback |
| Span 长度 K | 8,16,24,32 均匀采样 |
| g_max | 8 |
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

    out = span_fwd(seq_prompt, K, train=True)

    # losses
    loss_ar  = ce(out['seq'][:, :-K], seq_prompt[:, 1:])
    loss_lat = ((out['z_pred'] - out['z_teacher'].detach())**2).mean()
    loss_tok = ce(out['logits_span'], gold_span)
    exp_g    = (out['p_g']
                * torch.arange(1, g_max+1, device=seq_prompt.device)
               ).sum(-1).mean()
    loss     = loss_ar + 0.5*loss_lat + 0.5*loss_tok + beta*exp_g

    loss.backward()
    opt.step(); opt.zero_grad()
```

---

## 7. RoPE 实现要点  

1. 绝对 index = 已生成长度 + span 内 offset。  
2. Decoder 重算时仅重新计算 `start_pos:` 之后，左侧 KV 复用且不旋转。  
3. Encoder 每步调用 `apply_rope(z, pos_global)`，始终与最新 `h_dec` 对齐。  

---

## 8. 推理接口

```python
def generate(prompt, max_new=1024,
             K_fn=lambda rest: 16,
             mode='auto', temperature=0.8, top_p=0.9):
    seq = tokenizer(prompt).unsqueeze(0).to(device)
    while len(seq[0]) - prompt_len < max_new:
        K = K_fn(max_new - (len(seq[0]) - prompt_len))

        if mode == 'fast':
            GateNet.override(lambda p: torch.ones_like(p[:, :1]))      # ĝ=1
        elif mode == 'accurate':
            GateNet.override(lambda p: torch.zeros_like(p)
                                      .scatter_(-1, g_max-1, 1))       # ĝ=g_max
        else:
            GateNet.override(None)  # 使用学习到的 p_g

        out = span_fwd(seq, K, train=False, temp=temperature)
        seq = out['seq']
    return detokenize(seq[0])
```

---

## 9. 正确性与效率检查清单  

1. mode='fast' 时输出与纯 GPT 一致。  
2. 同一输入，平均 ĝ 随 span 熵增大而上升。  
3. 写回同 index 可先错后对，最终 logits 等同直接填入真 token。  
4. A100-80G, 7 B, ĝ≈3.1, K=16：吞吐 ≥110 tok/s。  

---

### 结语  

SpanFusionLM 在保持自回归因果的同时，通过“熵排序 + GateNet 自调节步数 + 潜空间多步 Refinement + 增量 RoPE 解码”实现块内高质量补全。模型能自动分配思考预算：难 token 多想，易 token 速决；整体计算成本仅比同规模 GPT 上升约 20 %，适合直接落地大规模预训练。

python -m SpanFusionLM.pretrain   # 训练示例  
python -m SpanFusionLM.infer      # 推理示例