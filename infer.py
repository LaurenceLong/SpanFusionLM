import torch
from spanfusionlm.model import SpanFusionLM, SpanFusionLMConfig

def generate(prompt, max_new=1024,
             K_fn=lambda rest: 16,
             g=4, temperature=0.8, top_p=0.9):
    """推理接口，实现伪代码中的generate函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置模型（使用较小配置用于演示）
    config = SpanFusionLMConfig(
        vocab_size=50257,
        hidden_size=512,
        intermediate_size=512 * 2,
        num_decoder_layers=4,
        num_encoder_layers=2,
        num_attention_heads=8,
        max_position_embeddings=512
    )

    model = SpanFusionLM(config).to(device)
    model.eval()

    # token化输入
    seq = config.tokenizer(prompt).unsqueeze(0).to(device)
    prompt_len = seq.shape[1]

    # 生成循环
    with torch.no_grad():
        while seq.shape[1] - prompt_len < max_new:
            rest = max_new - (seq.shape[1] - prompt_len)
            K = K_fn(rest)
            K = min(K, rest)  # 确保不超过剩余长度

            if K <= 0:
                break

            seq = model.span_forward_pass(
                seq, K, g,
                temperature=temperature,
                top_p=top_p,
                is_teacher_path=False
            )

    return config.tokenizer.detokenize(seq[0])

if __name__ == "__main__":
    prompt = "Hello world"
    output = generate(prompt, max_new=50, K_fn=lambda rest: min(8, rest), g=2)
    print("Generated output:")
    print(output)
