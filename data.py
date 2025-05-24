# data.py
import math
import torch
from torch.nn.utils.rnn import pad_sequence

def training_collate_fn(batch, pad_token_id, tbd_token_id):
    """
    对 batch 中的每个样本：
      - 对 seq_prompt 和 gold_span 进行 pad；
      - 将 gold_span 的最后一个 token 固定设置为 <|TBD|> token。
    """
    # 假设 batch 中每个元素为 (seq_prompt, gold_span, current_K)
    seq_prompts, gold_spans, current_K = zip(*batch)
    seq_prompts_padded = pad_sequence(seq_prompts, batch_first=True, padding_value=pad_token_id)
    gold_spans_padded = pad_sequence(gold_spans, batch_first=True, padding_value=pad_token_id)
    # 设置每个 gold_span 最后一个 token 为 <|TBD|> token（使用 tbd_token_id）
    gold_spans_padded[:, -1] = tbd_token_id
    return seq_prompts_padded, gold_spans_padded, torch.tensor(current_K)


def load_preprocessed_dataset(args, tokenizer):
    """
    模拟加载预处理后的数据集。
    实际实现中可对原始数据进行 tokenize、分割后缓存。
    此处为演示，返回若干样本，每个样本包含：
      - seq_prompt: torch.tensor (token id 序列)
      - gold_span: torch.tensor (token id 序列)，采样为 fixed_span_length tokens，
                   其中前 fixed_span_length-1 token 来自原始序列，最后一个 token 固定为 <|TBD|> token
      - current_K: 数值（初始为 0）
    """
    dummy_data = []
    fixed_span_length = math.ceil(math.sqrt(args.max_seq_length))
    for i in range(100):
        # 随机生成一些 token id 序列，仅供演示
        seq_prompt = torch.randint(0, len(tokenizer), (args.max_seq_length // 2,))
        # 采样固定长度 gold_span：取前 fixed_span_length-1 个 token，然后追加 <|TBD|> token
        if fixed_span_length - 1 > 0:
            gold_tokens = torch.randint(0, len(tokenizer), (fixed_span_length - 1,))
        else:
            gold_tokens = torch.tensor([], dtype=torch.long)
        tbd_token = torch.tensor([tokenizer.tbd_token_id], dtype=torch.long)
        gold_span = torch.cat([gold_tokens, tbd_token])
        current_K = 0
        dummy_data.append((seq_prompt, gold_span, current_K))
    return dummy_data
