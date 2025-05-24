# infer.py
import torch
import argparse
import json
from pathlib import Path

from .model import SpanFusionLM, SpanFusionLMConfig
from .modules.tokenizer import build_tokenizer

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load tokenizer
    tokenizer = build_tokenizer(args.model_name_or_path)  # Assumes tokenizer is saved with model

    # Load config
    config_path = Path(args.model_name_or_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Please ensure pretrain.py saves it.")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config_dict.pop("tokenizer_name", None)
    config_dict.pop("tokenizer", None)

    config = SpanFusionLMConfig(vocab_size=len(tokenizer), tokenizer=tokenizer, **config_dict)

    # Initialize model
    model = SpanFusionLM(config).to(device)

    # Load model weights
    model_weights_path = Path(args.model_name_or_path) / "model.pt"
    if not model_weights_path.exists():
        model_weights_path = Path(args.model_name_or_path) / "pytorch_model.bin"
        if not model_weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {args.model_name_or_path}/model.pt or pytorch_model.bin")

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    print(f"Model and tokenizer loaded. Using device: {device}")

    seq = torch.tensor(tokenizer.encode(args.prompt, add_special_tokens=False), dtype=torch.long).unsqueeze(0).to(device)
    prompt_len = seq.shape[1]
    generated_tokens = 0

    print(f"Starting generation with max_new_tokens={args.max_new_tokens}...")
    with torch.no_grad():
        while generated_tokens < args.max_new_tokens:
            current_seq_len = seq.shape[1]
            # 使用模型配置中的固定 span 长度
            K = model.config.fixed_span_length
            if current_seq_len + K > config.max_position_embeddings:
                print(f"Warning: Sequence length {current_seq_len + K} might exceed max_position_embeddings {config.max_position_embeddings}. Truncating K.")
                K = config.max_position_embeddings - current_seq_len
                if K <= 0:
                    print("Cannot generate further due to max length.")
                    break

            prompt_for_span = seq
            out = model.span_forward_pass(
                prompt_for_span,
                K,
                temperature=args.temperature,
                top_p=args.top_p
            )

            newly_generated_part = out['seq'][:, current_seq_len : current_seq_len + K]
            seq = out['seq']
            generated_tokens += newly_generated_part.shape[1]

            # 如果生成的最后一个 token 为 <|TBD|> 则提前停止
            if tokenizer.tbd_token_id is not None and (tokenizer.tbd_token_id in newly_generated_part[0]):
                print("TBD token generated. Early stopping.")
                break

            if (tokenizer.eos_token_id is not None) and (tokenizer.eos_token_id in newly_generated_part[0]):
                print("EOS token generated.")
                eos_indices = (newly_generated_part[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    first_eos_idx_in_new = eos_indices[0].item()
                    seq = seq[:, : current_seq_len + first_eos_idx_in_new + 1]
                    generated_tokens = seq.shape[1] - prompt_len
                break

    final_output_ids = seq[0, prompt_len:].tolist()
    if config.pred_token_id is not None:
        final_output_ids = [tok_id for tok_id in final_output_ids if tok_id != config.pred_token_id]

    return tokenizer.decode(final_output_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using SpanFusionLM.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Initial prompt for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for generation.")

    args = parser.parse_args()

    output_text = generate(args)
    print("\n--- Generated Output ---")
    print(output_text)
    print("--- End of Output ---")
