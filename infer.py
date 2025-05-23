import torch
from spanfusionlm.model import SpanFusionLM, SpanFusionLMConfig

def generate(prompt, max_new=1024,
             K_fn=lambda rest: 16,
             mode='auto', temperature=0.8, top_p=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SpanFusionLMConfig(
        vocab_size=50257,   # 可与 GPT-2 默认一致
        hidden_size=512,    # 小模型示例
        intermediate_size=1024,
        num_decoder_layers=4,
        num_encoder_layers=2,
        num_attention_heads=8,
        max_position_embeddings=512,
        g_max=8
    )
    model = SpanFusionLM(config).to(device)
    model.eval()
    # 使用 tokenizer 进行 token 化
    seq = config.tokenizer(prompt).unsqueeze(0).to(device)
    prompt_len = seq.shape[1]
    while seq.shape[1] - prompt_len < max_new:
        K = K_fn(max_new - (seq.shape[1] - prompt_len))
        if mode == 'fast':
            # ĝ 固定为1，即只做一步 Refinement
            model.gate_net.override(lambda p: torch.ones(p.size(0), 1, device=p.device))
        elif mode == 'accurate':
            # ĝ 固定为 g_max
            model.gate_net.override(lambda p: torch.ones(p.size(0), 1, device=p.device) * config.g_max)
        else:
            model.gate_net.override(None)
        out = model.span_forward_pass(seq, K, g=None,
                                      temperature=temperature,
                                      top_p=top_p,
                                      is_teacher_path=False)
        seq = out['seq']
    return config.tokenizer.detokenize(seq[0])

if __name__ == "__main__":
    prompt = "Hello world"
    output = generate(prompt, mode='auto')
    print("Generated output:")
    print(output)
