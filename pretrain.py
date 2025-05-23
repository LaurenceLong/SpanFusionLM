# SpanFusionLM/pretrain.py
import argparse
import logging
import math
import random
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb  # Assuming wandb is installed and configured
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler  # For learning rate scheduling

from .model import SpanFusionLM, SpanFusionLMConfig
from .modules.tokenizer import build_tokenizer

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SpanFusionLM model.")
    # Data args
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Name of the dataset from Hugging Face datasets.")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1", help="Configuration name for the dataset.")
    parser.add_argument("--train_split", type=str, default="train", help="Dataset split for training.")
    parser.add_argument("--eval_split", type=str, default="test", help="Dataset split for evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=992, help="Maximum sequence length for model input.")
    
    # Model args
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name or path.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Model hidden size.")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="MLP intermediate size.")
    parser.add_argument("--num_decoder_layers", type=int, default=8, help="Number of decoder layers.")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers.")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads.")
    # max_position_embeddings will be max_seq_length + max_K from span_lengths
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta value.")
    parser.add_argument("--g_max", type=int, default=8, help="Maximum GateNet steps.")
    parser.add_argument("--span_lengths_str", type=str, default="8,16,24,32", help="Comma-separated string of possible span lengths K.")

    # Training args
    parser.add_argument("--output_dir", type=str, default="./spanfusionlm_checkpoint", help="Output directory to save model and tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--beta_cost_g", type=float, default=0.02, help="Coefficient for E[g] cost in loss.")
    
    parser.add_argument("--batch_size_per_device", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps. Overrides num_train_epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type (e.g., linear, cosine).")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for lr scheduler.")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every N training steps.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log training info every N steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps.")

    # W&B args
    parser.add_argument("--wandb_project", type=str, default="SpanFusionLM", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")

    args = parser.parse_args()
    args.span_lengths = [int(k) for k in args.span_lengths_str.split(',')]
    return args

def collate_fn(batch_samples, tokenizer, fixed_K_value, max_seq_len, pad_token_id):
    # batch_samples is a list of dataset items, e.g., [{'text': "..."}]
    seq_prompts_list, gold_spans_list = [], []

    for sample in batch_samples:
        text = sample.get("text", "") # Make sure 'text' field exists
        if not isinstance(text, str) or not text.strip():
            continue

        # Tokenize without special tokens initially, up to a certain max to avoid excessive memory
        # Max length for initial tokenization: max_seq_len + fixed_K_value + buffer
        buffer = 50 
        tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_seq_len + fixed_K_value + buffer, truncation=True)

        if len(tokens) < fixed_K_value + 1: # Need at least 1 token for prompt and K for span
            continue
        
        # Max prompt length such that prompt + K <= max_seq_len
        # Prompt itself should not exceed max_seq_len - K.
        # And prompt should not exceed len(tokens) - K
        effective_max_prompt_len = min(max_seq_len - fixed_K_value, len(tokens) - fixed_K_value)
        
        if effective_max_prompt_len < 1: # Not enough tokens for a valid prompt
            continue
            
        prompt_len = random.randint(1, effective_max_prompt_len)
        
        seq_prompt_toks = tokens[:prompt_len]
        gold_span_toks = tokens[prompt_len : prompt_len + fixed_K_value]

        # Ensure gold_span actually has K tokens, might be less if text was short
        if len(gold_span_toks) != fixed_K_value:
            # This can happen if len(tokens) was just slightly more than fixed_K_value
            # and prompt_len was chosen such that remaining tokens < fixed_K_value
            # This should be rare if effective_max_prompt_len logic is correct
            continue # Skip this sample

        seq_prompts_list.append(torch.tensor(seq_prompt_toks, dtype=torch.long))
        gold_spans_list.append(torch.tensor(gold_span_toks, dtype=torch.long))

    if not seq_prompts_list: # If no valid samples were processed
        return None

    # Pad sequences
    # seq_prompt is (B, n_prompt), gold_span is (B, K)
    padded_seq_prompts = torch.nn.utils.rnn.pad_sequence(
        seq_prompts_list, batch_first=True, padding_value=pad_token_id
    )
    padded_gold_spans = torch.nn.utils.rnn.pad_sequence(
        gold_spans_list, batch_first=True, padding_value=pad_token_id
    )
    
    return padded_seq_prompts, padded_gold_spans


@torch.no_grad()
def evaluate(model, dataloader, accelerator, config, args):
    model.eval()
    total_ar_loss = 0.0
    total_samples = 0
    
    fixed_K_for_eval = random.choice(args.span_lengths) # Or use a fixed one, e.g., args.span_lengths[0]

    for step, batch_samples in enumerate(dataloader):
        collated_data = collate_fn(batch_samples, config.tokenizer, fixed_K_for_eval, args.max_seq_length, config.pad_token_id)
        if collated_data is None:
            continue
        
        seq_prompt, gold_span = collated_data
        seq_prompt = seq_prompt.to(accelerator.device)
        gold_span = gold_span.to(accelerator.device)

        # Prepare for L_AR calculation
        # L_AR: Autoregressive CE on the complete gold sequence.
        gold_full_seq = torch.cat([seq_prompt, gold_span], dim=1)
        
        # Ensure gold_full_seq is not longer than max_position_embeddings
        # This should be handled by collate_fn ensuring prompt+K <= max_seq_len
        # And max_seq_len itself should be <= config.max_position_embeddings
        if gold_full_seq.shape[1] > config.max_position_embeddings:
             gold_full_seq = gold_full_seq[:, :config.max_position_embeddings]


        gold_input_ids = gold_full_seq[:, :-1]
        gold_target_ids = gold_full_seq[:, 1:]
        
        if gold_input_ids.shape[1] == 0: # Skip if sequence becomes too short
            continue

        # Direct pass through decoder for L_AR
        unwrapped_model = accelerator.unwrap_model(model)
        embeddings = unwrapped_model.token_emb(gold_input_ids)
        
        # Create position_ids for the AR pass
        batch_size, ar_seq_len = gold_input_ids.shape
        ar_position_ids = torch.arange(ar_seq_len, device=accelerator.device).unsqueeze(0).expand(batch_size, -1)

        # Decoder pass (no KV cache needed for AR loss on full sequence from scratch)
        h_gold_ar, _ = unwrapped_model.decoder(
            hidden_states=embeddings,
            attention_mask=None, # Causal mask applied internally
            position_ids=ar_position_ids,
            past_key_values=None,
            use_cache=False 
        )
        logits_ar = unwrapped_model.proj_head(h_gold_ar) # (B, ar_seq_len, V)

        loss_ar = F.cross_entropy(
            logits_ar.reshape(-1, config.vocab_size),
            gold_target_ids.reshape(-1),
            ignore_index=config.pad_token_id
        )
        
        total_ar_loss += loss_ar.item() * gold_input_ids.shape[0] # loss_ar is mean, scale by batch size
        total_samples += gold_input_ids.shape[0]

    avg_ar_loss = total_ar_loss / total_samples if total_samples > 0 else 0.0
    perplexity = math.exp(avg_ar_loss) if avg_ar_loss > 0 else float('inf')
    
    model.train()
    return avg_ar_loss, perplexity


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        # transformers.utils.logging.set_verbosity_info() # If using transformers logging
    else:
        datasets.utils.logging.set_verbosity_error()
        # transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and not args.disable_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    # Load tokenizer
    tokenizer = build_tokenizer(args.tokenizer_name)

    # Create Model Config
    # max_pos_emb needs to accommodate max_seq_len + max_K
    max_k_val = max(args.span_lengths) if args.span_lengths else 32
    config_max_pos_embeddings = args.max_seq_length + max_k_val # Ensure this is used by RoPE

    model_config = SpanFusionLMConfig(
        vocab_size=len(tokenizer), # Will be set by tokenizer
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=config_max_pos_embeddings,
        rope_theta=args.rope_theta,
        g_max=args.g_max,
        tokenizer_name=args.tokenizer_name, # Save for reloading
        tokenizer=tokenizer # Pass tokenizer object
    )
    
    logger.info(f"Initializing model with config: {model_config}")
    model = SpanFusionLM(model_config)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.weight_decay
    )

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name})")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    
    # For IterableDataset, we might not know the length beforehand for max_train_steps
    # For now, assume map-style dataset for simplicity in length calculation
    train_dataset = raw_datasets[args.train_split]
    eval_dataset = raw_datasets[args.eval_split]

    # DataLoaders
    # We need to pass a randomly chosen K for each batch to collate_fn
    # This makes the DataLoader a bit more complex if K changes per batch.
    # Simpler: collate_fn itself picks K for its batch.
    
    def train_collate_wrapper(batch_samples):
        # For each call to collate_fn (i.e., for each batch), pick a K.
        current_K = random.choice(args.span_lengths)
        return collate_fn(batch_samples, tokenizer, current_K, args.max_seq_length, model_config.pad_token_id)

    def eval_collate_wrapper(batch_samples):
        current_K = random.choice(args.span_lengths) # Or a fixed K for eval consistency
        return collate_fn(batch_samples, tokenizer, current_K, args.max_seq_length, model_config.pad_token_id)


    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size_per_device, 
        shuffle=True, # Shuffle for map-style dataset
        collate_fn=train_collate_wrapper,
        drop_last=True # Important for consistent batch shapes
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size_per_device, 
        collate_fn=eval_collate_wrapper
    )

    # Scheduler
    # Calculate total training steps if not provided
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / (len(train_dataloader) / args.gradient_accumulation_steps))


    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * args.gradient_accumulation_steps, # Scheduler steps by grad updates
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with Accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size_per_device}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.batch_size_per_device * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Training Steps")
    completed_steps = 0
    
    # Training Loop
    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch_data in enumerate(train_dataloader):
            if completed_steps >= args.max_train_steps:
                break
            
            if batch_data is None: # Skip if collate_fn returned None
                logger.warning(f"Skipping batch at epoch {epoch}, step {step} due to empty collated data.")
                continue
            
            seq_prompt, gold_span = batch_data # K is implicit in gold_span.shape[1]
            current_K_for_batch = gold_span.shape[1]

            seq_prompt = seq_prompt.to(accelerator.device)
            gold_span = gold_span.to(accelerator.device)

            with accelerator.accumulate(model):
                # Forward pass - model.forward will call span_forward_pass for student and teacher
                # It stores z_pred_for_loss, z_teacher_for_loss, logits_span_for_loss internally
                # And returns student_output which includes p_g for E[g] loss
                student_out = model(
                    seq_prompt, 
                    current_K_for_batch, 
                    gold_span=gold_span, # Not directly used by model.forward but good for clarity
                    compute_teacher_latent=True # Enable teacher path for L_latent
                )
                
                unwrapped_model = accelerator.unwrap_model(model)

                # 1. L_AR: Autoregressive CE on the complete gold sequence.
                gold_full_seq = torch.cat([seq_prompt, gold_span], dim=1)
                # Truncate if exceeds model capacity (collate_fn should ideally prevent this for prompt+K)
                if gold_full_seq.shape[1] > unwrapped_model.config.max_position_embeddings:
                    gold_full_seq = gold_full_seq[:, :unwrapped_model.config.max_position_embeddings]

                gold_input_ids = gold_full_seq[:, :-1]
                gold_target_ids = gold_full_seq[:, 1:]

                loss_ar = torch.tensor(0.0, device=accelerator.device)
                if gold_input_ids.shape[1] > 0:
                    embeddings = unwrapped_model.token_emb(gold_input_ids)
                    batch_size_ar, ar_seq_len = gold_input_ids.shape
                    ar_pos_ids = torch.arange(ar_seq_len, device=accelerator.device).unsqueeze(0).expand(batch_size_ar, -1)
                    
                    h_gold_ar, _ = unwrapped_model.decoder(
                        embeddings, 
                        attention_mask=None, 
                        position_ids=ar_pos_ids, 
                        past_key_values=None, 
                        use_cache=False
                    )
                    logits_ar = unwrapped_model.proj_head(h_gold_ar)
                    loss_ar = F.cross_entropy(
                        logits_ar.reshape(-1, model_config.vocab_size),
                        gold_target_ids.reshape(-1),
                        ignore_index=model_config.pad_token_id
                    )

                # 2. L_latent: MSE between z_pred and stop_grad(z_teacher)
                loss_lat = torch.tensor(0.0, device=accelerator.device)
                if unwrapped_model.z_pred_for_loss is not None and unwrapped_model.z_teacher_for_loss is not None:
                    loss_lat = F.mse_loss(unwrapped_model.z_pred_for_loss, unwrapped_model.z_teacher_for_loss)
                
                # 3. L_token: CE(Softmax(logits_span), gold_span)
                loss_tok = torch.tensor(0.0, device=accelerator.device)
                if unwrapped_model.logits_span_for_loss is not None:
                    loss_tok = F.cross_entropy(
                        unwrapped_model.logits_span_for_loss.reshape(-1, model_config.vocab_size),
                        gold_span.reshape(-1), # gold_span is (B, K)
                        ignore_index=model_config.pad_token_id
                    )
                
                # 4. E[g]: Expected steps from GateNet
                exp_g = torch.tensor(0.0, device=accelerator.device)
                if student_out['p_g'] is not None: # p_g comes from student_out
                    g_range = torch.arange(1, model_config.g_max + 1, device=accelerator.device, dtype=torch.float)
                    exp_g = (student_out['p_g'] * g_range).sum(dim=-1).mean()
                
                # Total Loss
                total_loss = loss_ar + 0.5 * loss_lat + 0.5 * loss_tok + args.beta_cost_g * exp_g
                
                accelerator.backward(total_loss)
                
                if args.grad_clip is not None and args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step under the hood
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % args.logging_steps == 0:
                    loss_log = {
                        "train_loss": total_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "loss_ar": loss_ar.item() if isinstance(loss_ar, torch.Tensor) else loss_ar,
                        "loss_lat": loss_lat.item() if isinstance(loss_lat, torch.Tensor) else loss_lat,
                        "loss_tok": loss_tok.item() if isinstance(loss_tok, torch.Tensor) else loss_tok,
                        "exp_g": exp_g.item() if isinstance(exp_g, torch.Tensor) else exp_g,
                        "epoch": epoch,
                        "step": completed_steps,
                    }
                    accelerator.log(loss_log, step=completed_steps)
                    if accelerator.is_main_process and not args.disable_wandb:
                         wandb.log(loss_log)
                    logger.info(f"Epoch {epoch}, Step {completed_steps}: {loss_log}")


                if completed_steps % args.eval_steps == 0:
                    logger.info(f"Running evaluation at step {completed_steps}...")
                    avg_ar_loss, perplexity = evaluate(model, eval_dataloader, accelerator, model_config, args)
                    eval_log = {
                        "eval_ar_loss": avg_ar_loss,
                        "eval_perplexity": perplexity,
                    }
                    accelerator.log(eval_log, step=completed_steps)
                    if accelerator.is_main_process and not args.disable_wandb:
                        wandb.log(eval_log)
                    logger.info(f"Evaluation results at step {completed_steps}: {eval_log}")
                    model.train() # Set back to train mode

                if completed_steps % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = Path(args.output_dir) / f"checkpoint-{completed_steps}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        
                        unwrapped_model_to_save = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model_to_save.state_dict(), save_path / "model.pt")
                        tokenizer.save_pretrained(str(save_path))
                        # Save config
                        unwrapped_model_to_save.config.save_pretrained(str(save_path))
                        logger.info(f"Checkpoint saved to {save_path}")
            
            if completed_steps >= args.max_train_steps:
                break
    
    # Final save
    if accelerator.is_main_process:
        final_save_path = Path(args.output_dir) / "final_checkpoint"
        final_save_path.mkdir(parents=True, exist_ok=True)
        unwrapped_model_final = accelerator.unwrap_model(model)
        torch.save(unwrapped_model_final.state_dict(), final_save_path / "model.pt")
        tokenizer.save_pretrained(str(final_save_path))
        unwrapped_model_final.config.save_pretrained(str(final_save_path))
        logger.info(f"Final model checkpoint saved to {final_save_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not args.disable_wandb:
        wandb.finish()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
