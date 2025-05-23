# SpanFusionLM/model.py 的关键修复部分

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
        self.gate_net = GateNet(input_dim=1, hidden_dim=config.hidden_size // 16, g_max=config.g_max)

        # 用于损失计算的临时变量
        self.z_pred_for_loss = None
        self.z_teacher_for_loss = None
        self.logits_span_for_loss = None

    def _calc_entropy(self, logits: torch.Tensor):
        """计算熵 - 增强数值稳定性"""
        # 更严格的数值稳定性处理
        logits = torch.clamp(logits, min=-20, max=20)  # 更严格的范围限制

        # 使用数值稳定的方式计算熵
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # 确保概率不为0或过小
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        log_probs = torch.clamp(log_probs, min=-20, max=0)

        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.clamp(entropy, min=0, max=10)  # 限制熵的最大值

        # 检查并修复NaN/Inf
        entropy = torch.where(torch.isnan(entropy) | torch.isinf(entropy),
                             torch.ones_like(entropy), entropy)

        return entropy

    def span_forward_pass(self,
                          seq_prompt: torch.Tensor,
                          K: int,
                          g: int = None,
                          temperature: float = 1.0,
                          top_p: float = 0.9,
                          is_teacher_path: bool = False):
        """增强数值稳定性的span前向传播"""

        B, n_prompt = seq_prompt.shape
        device = seq_prompt.device

        if K == 0:
            return {'seq': seq_prompt, 'z_pred': None, 'logits_span': None, 'p_g': None}

        try:
            # 1. 构建初始序列
            pred_tokens = torch.full((B, K), self.config.pred_token_id, dtype=torch.long, device=device)
            seq = torch.cat([seq_prompt, pred_tokens], dim=1)

            # 2. 初始decoder前向
            embeddings = self.token_emb(seq)
            # 添加小的噪声来增强数值稳定性
            if self.training:
                embeddings = embeddings + torch.randn_like(embeddings) * 0.01

            position_ids = torch.arange(n_prompt + K, device=device).unsqueeze(0).expand(B, -1)

            h_dec, _ = self.decoder(
                hidden_states=embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False
            )

            # 3. 计算初始熵
            h_pred_positions = h_dec[:, n_prompt:, :]

            # 添加层归一化来稳定输出
            h_pred_positions = F.layer_norm(h_pred_positions, h_pred_positions.shape[-1:])

            logits_pred = self.proj_head(h_pred_positions)
            entropy = self._calc_entropy(logits_pred)

            # 4. GateNet预测步数
            if g is not None:
                g_hat = torch.full((B,), g, dtype=torch.long, device=device)
                gate_logits = None
            else:
                mean_entropy = entropy.mean(dim=-1, keepdim=True)
                # 确保mean_entropy在合理范围内
                mean_entropy = torch.clamp(mean_entropy, min=0, max=5)
                g_hat, gate_logits = self.gate_net(mean_entropy, train=self.training and not is_teacher_path)

            g_loop = min(g_hat.max().item() if g_hat.numel() > 0 else 0, self.config.g_max)  # 限制最大循环次数

            # 5. 简化的refinement过程
            z = self.token_emb(pred_tokens)
            if self.training:
                z = z + torch.randn_like(z) * 0.005  # 添加小噪声

            filled_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

            for t in range(g_loop):
                active_mask = g_hat > t
                if not active_mask.any():
                    break

                # 计算当前步的填充数量
                g_active = g_hat[active_mask].float()
                M_t_active = torch.ceil((t + 1) * K / g_active) - torch.ceil(t * K / g_active)
                M_t = torch.zeros(B, device=device)
                M_t[active_mask] = M_t_active

                # 选择要填充的位置
                pick_mask = self._select_top_m_positions(entropy, filled_mask, M_t)

                # Encoder refinement
                try:
                    z = self.encoder.step(z, h_pred_positions.detach(), n_prompt)
                    # 添加层归一化
                    z = F.layer_norm(z, z.shape[-1:])
                except Exception as e:
                    logger.warning(f"Encoder step failed: {e}")
                    break

                # 投射并采样
                if pick_mask.any():
                    z_picked = z[pick_mask]
                    # 确保温度不为0
                    safe_temp = max(temperature, 0.1)
                    logits_picked = self.proj_head(z_picked) / safe_temp

                    if is_teacher_path:
                        tokens_picked = torch.argmax(logits_picked, dim=-1)
                    else:
                        tokens_picked = sample_from_logits(logits_picked, safe_temp, top_p)

                    # 更新序列
                    seq_new = seq.clone()
                    seq_new[:, n_prompt:][pick_mask] = tokens_picked
                    seq = seq_new
                    filled_mask = filled_mask | pick_mask

                # 保存用于损失计算
                if t == g_loop - 1:
                    if is_teacher_path:
                        self.z_teacher_for_loss = z.detach().clone()
                    else:
                        self.z_pred_for_loss = z.clone()
                        self.logits_span_for_loss = self.proj_head(z)

            # 确保损失变量被设置
            if not is_teacher_path and self.z_pred_for_loss is None:
                self.z_pred_for_loss = z.clone()
                self.logits_span_for_loss = self.proj_head(z)

            return {
                'seq': seq,
                'z_pred': z if not is_teacher_path else None,
                'logits_span': self.proj_head(z) if not is_teacher_path else None,
                'p_g': F.softmax(gate_logits, dim=-1) if gate_logits is not None else None
            }

        except Exception as e:
            logger.error(f"Error in span_forward_pass: {e}")
            # 返回一个安全的默认输出
            return {
                'seq': torch.cat([seq_prompt, torch.zeros(B, K, dtype=torch.long, device=device)], dim=1),
                'z_pred': None,
                'logits_span': None,
                'p_g': None
            }
