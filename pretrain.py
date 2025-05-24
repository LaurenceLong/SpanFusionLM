# pretrain.py
import argparse
import logging
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from .model import SpanFusionLM, SpanFusionLMConfig
from .modules.tokenizer import build_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SpanFusionLM model.")

    # Data args
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1", help="Dataset config")
    parser.add_argument("--train_split", type=str, default="train", help="Train split")
    parser.add_argument("--eval_split", type=str, default="test", help="Eval split")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")

    # Model args (以 GPT-2 small 为例)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size (GPT-2 small configuration)")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Intermediate size (GPT-2 small configuration)")
    parser.add_argument("--num_decoder_layers", type=int, default=8, help="Decoder layers (GPT-2 small configuration)")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Encoder layers (set to about half of decoder layers)")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Attention heads (GPT-2 small configuration)")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
    parser.add_argument("--g_max", type=int, default=4, help="Max GateNet iterations")
    parser.add_argument("--span_lengths_str", type=str, default="8,16", help="Comma-separated span lengths")

    # Training args
    parser.add_argument("--output_dir", type=str, default="./spanfusionlm_checkpoint", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--beta_cost_g", type=float, default=0.05, help="Initial β·E[g] cost coefficient")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision")

    # 注意：现在 collate_fn 只加载文本并转为数字，因此可以用更大的 batch_size
    parser.add_argument("--batch_size_per_device", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_steps", type=int, default=200, help="Eval steps")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for DataLoader")
    parser.add_argument("--wandb_project", type=str, default="SpanFusionLM", help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()
    args.span_lengths = [int(k) for k in args.span_lengths_str.split(',')]
    return args


def simple_collate_fn(batch_samples, tokenizer, max_seq_len, pad_token_id):
    """
    简单地将 batch 中的文本加载并转换为数字（token id），不进行 prompt–span 划分。
    每个样本返回一条 token 序列（未 pad），后续训练中会针对每条数据进行 next_span 切分。
    """
    token_seqs = []
    for sample in batch_samples:
        text = sample.get("text", "")
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=max_seq_len,
            truncation=True
        )
        if len(tokens) == 0:
            continue
        token_seqs.append(torch.tensor(tokens, dtype=torch.long))
    return token_seqs


def compute_losses(model, seq_prompt, gold_span, current_K, model_config, accelerator):
    output = model(seq_prompt, current_K, gold_span=gold_span)
    unwrapped_model = accelerator.unwrap_model(model)
    losses = {}

    gold_full_seq = torch.cat([seq_prompt, gold_span], dim=1)
    if gold_full_seq.shape[1] > model_config.max_position_embeddings:
        gold_full_seq = gold_full_seq[:, :model_config.max_position_embeddings]

    gold_input_ids = gold_full_seq[:, :-1]
    gold_target_ids = gold_full_seq[:, 1:]

    if gold_input_ids.shape[1] > 0:
        embeddings = unwrapped_model.token_emb(gold_input_ids)
        batch_size_ar, ar_seq_len = gold_input_ids.shape
        ar_pos_ids = torch.arange(ar_seq_len, device=accelerator.device).unsqueeze(0).expand(batch_size_ar, -1)
        padding_mask = (gold_input_ids != model_config.pad_token_id).bool()

        h_gold_ar, _ = unwrapped_model.decoder(
            hidden_states=embeddings,
            attention_mask=None,
            position_ids=ar_pos_ids,
            padding_mask=padding_mask,
            past_key_values=None,
            use_cache=False
        )
        logits_ar = unwrapped_model.proj_head(h_gold_ar)
        losses['ar'] = F.cross_entropy(
            logits_ar.reshape(-1, model_config.vocab_size),
            gold_target_ids.reshape(-1),
            ignore_index=model_config.pad_token_id
        )
    else:
        losses['ar'] = torch.tensor(0.0, device=accelerator.device)

    # 使用 output 中的 logits_span 计算 token 预测 loss
    losses['token'] = F.cross_entropy(
        output['logits_span'].reshape(-1, model_config.vocab_size),
        gold_span.reshape(-1),
        ignore_index=model_config.pad_token_id
    )

    return losses, output


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        device_placement=True
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if accelerator.is_main_process and not args.disable_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    tokenizer = build_tokenizer(args.tokenizer_name)

    # 这里模型配置里的 max_position_embeddings 取 max_seq_length + max(span_lengths)
    max_k_val = max(args.span_lengths)
    config_max_pos_embeddings = args.max_seq_length + max_k_val

    model_config = SpanFusionLMConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=config_max_pos_embeddings,
        rope_theta=args.rope_theta,
        g_max=args.g_max,
        tokenizer_name=args.tokenizer_name,
        tokenizer=tokenizer
    )

    logger.info(f"Model config: {model_config}")
    model = SpanFusionLM(model_config)
    logger.info(f"模型结构: {model}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数总数: {total_params}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    logger.info(f"Loading dataset: {args.dataset_name}")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    train_dataset = raw_datasets[args.train_split]

    # 使用简单的 collate_fn 仅加载 token 数字序列
    def collate_wrapper(batch_samples):
        return simple_collate_fn(batch_samples, tokenizer, args.max_seq_length, model_config.pad_token_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=True,
        collate_fn=collate_wrapper,
        drop_last=True,
        num_workers=args.num_workers
    )

    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
    completed_steps = 0
    skip_count = 0

    model.train()

    for epoch in range(args.num_train_epochs):
        for step, batch_data in enumerate(train_dataloader):
            # batch_data 为一个列表，每个元素是一条 token 序列（1D tensor）
            if not batch_data or len(batch_data) == 0:
                continue

            # 随机选择一个当前的 span 长度，用于本个 batch 内所有样本
            current_K = random.choice(args.span_lengths)
            all_prompts = []
            all_spans = []

            for seq in batch_data:
                T = seq.size(0)
                # 如果序列长度不足以生成 prompt 和 span，则略过
                if T < current_K + 1:
                    continue

                # 计算可选的最大初始 prompt 长度，保证后续有足够空间划分 span，
                # 此处取三个值中的最小值：文中剩余长度、允许的最大长度、以及 fixed_K 自身
                max_initial = min(T - current_K, args.max_seq_length - current_K, current_K)
                if max_initial < 1:
                    continue

                prompt_len = random.randint(1, max_initial)
                idx = prompt_len
                # 对当前序列多次生成 prompt–span 对
                while idx + current_K <= T and idx + current_K <= args.max_seq_length:
                    prompt_tokens = seq[:idx]
                    span_tokens = seq[idx: idx + current_K]
                    all_prompts.append(prompt_tokens)
                    all_spans.append(span_tokens)
                    idx += current_K

            if len(all_prompts) == 0:
                continue

            # 对所有生成的 prompt 序列进行 pad，span 序列固定长度直接 stack
            padded_seq_prompts = pad_sequence(
                all_prompts,
                batch_first=True,
                padding_value=model_config.pad_token_id
            )
            padded_gold_spans = torch.stack(all_spans, dim=0)

            with accelerator.accumulate(model):
                losses, output = compute_losses(
                    model, padded_seq_prompts, padded_gold_spans, current_K, model_config, accelerator
                )

                g_loss = args.beta_cost_g * output["g_hat"].float().mean()
                total_loss = 0.5 * losses['ar'] + 0.5 * losses['token'] + g_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    skip_count += 1
                    logger.warning(f"Invalid loss detected: {total_loss}, skipping batch")
                    continue

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    if args.grad_clip > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    completed_steps += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

                    if completed_steps % args.logging_steps == 0:
                        g_avg = output["g_hat"].float().mean().item()
                        loss_log = {
                            "train_loss": total_loss.item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "loss_ar": losses['ar'].item(),
                            "loss_token": losses['token'].item(),
                            "loss_g": g_loss,
                            "g_avg": g_avg,
                            "step": completed_steps,
                            "skip_count": skip_count,
                        }

                        if accelerator.is_main_process and not args.disable_wandb:
                            wandb.log(loss_log, step=completed_steps)
                        logger.info(f"Step {completed_steps}: Loss={total_loss.item():.4f}, AR_loss={losses['ar'].item():.4f}, Token_loss={losses['token'].item():.4f}, G_loss={g_loss:.4f}, G_avg={g_avg:.4f}")

                    if completed_steps % args.save_steps == 0:
                        if accelerator.is_main_process:
                            save_path = Path(args.output_dir) / f"checkpoint-{completed_steps}"
                            save_path.mkdir(parents=True, exist_ok=True)

                            unwrapped_model_to_save = accelerator.unwrap_model(model)
                            torch.save(unwrapped_model_to_save.state_dict(), save_path / "model.pt")
                            tokenizer.save_pretrained(str(save_path))
                            unwrapped_model_to_save.config.save_pretrained(str(save_path))
                            logger.info(f"Checkpoint saved to {save_path}")

            if completed_steps >= args.max_train_steps:
                break

    if accelerator.is_main_process:
        final_save_path = Path(args.output_dir) / "final_checkpoint"
        final_save_path.mkdir(parents=True, exist_ok=True)
        unwrapped_model_final = accelerator.unwrap_model(model)
        torch.save(unwrapped_model_final.state_dict(), final_save_path / "model.pt")
        tokenizer.save_pretrained(str(final_save_path))
        unwrapped_model_final.config.save_pretrained(str(final_save_path))
        logger.info(f"Final model saved to {final_save_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not args.disable_wandb:
        wandb.finish()
    logger.info(f"Training complete. Total skipped batches: {skip_count}")


if __name__ == "__main__":
    main()