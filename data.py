# data.py
import os
import glob
import random
import math
import logging
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_and_save_shards(args, tokenizer, chunk_size=5000, data_dir=os.getcwd()):
    """
    按照固定 span 生成逻辑预处理原始数据，并分片存储到 data_dir 目录下。
    每个 shard 存储不超过 chunk_size 条生成的训练实例，实例格式为字典：
         {
            "prompt": list_of_token_ids,  # 初始 prompt（长度随机在 1 到 fixed_span_length 间）
            "span": list_of_token_ids,    # 从文本中截取固定长度 span_length-1 个真实 token，
                                          # 最后一个 token 固定设置为 <|TBD|> token
            "K": fixed_K_value           # 固定 span 长度，即 ceil(sqrt(max_seq_length))
         }
    同一文本可能产生多个实例，每次生成时将 span 追加到 prompt 后（即 prompt += span）。
    返回的 index_data 为字典：
         {"shard_files": [shard_file_0, shard_file_1, ...],
          "shard_lengths": [num0, num1, ...]}
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    index_filename = f"{data_dir}/cached_index_{args.dataset_name}_{args.train_split}_maxlen{args.max_seq_length}.pt"

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    raw_dataset = raw_datasets[args.train_split]
    shard_files = []
    shard_lengths = []

    current_shard = []
    shard_count = 0

    logger.info("Start processing raw data for shard building ...")
    # 固定 span 长度，根据 max_seq_length 计算（即 ceil(sqrt(max_seq_length))）
    fixed_K = math.ceil(math.sqrt(args.max_seq_length))

    # 使用 tqdm 展示预处理进度
    for sample in tqdm(raw_dataset, desc="Processing raw data", unit="sample"):
        text = sample.get("text", "")
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue

        # tokenize 并截断到最大序列长度
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=args.max_seq_length,
            truncation=True
        )
        seq_token_cnt = len(tokens)
        # 若文本 token 数不足以生成一个 prompt 和 span，则跳过
        if seq_token_cnt < fixed_K + 1:
            continue

        # prompt 的初始长度取值范围为：1 至 min(seq_token_cnt - fixed_K, fixed_K)
        max_initial = min(seq_token_cnt - fixed_K, fixed_K)
        if max_initial < 1:
            continue
        prompt_len = random.randint(1, max_initial)
        idx = prompt_len

        # 这里采用“prompt += span”的方式，即以固定步长不断扩充 prompt
        while idx + fixed_K <= seq_token_cnt and idx + fixed_K <= args.max_seq_length:
            # 训练实例中的 prompt 为文本中前 idx 个 token
            instance = {"prompt": tokens[:idx]}
            # span 的采样为从 idx 开始的固定长度：取前 fixed_K 个 token
            span_tokens = tokens[idx: idx + fixed_K]
            instance["span"] = span_tokens
            instance["K"] = fixed_K

            current_shard.append(instance)
            if len(current_shard) >= chunk_size:
                shard_filename = f"{data_dir}/cached_{args.dataset_name}_{args.train_split}_maxlen{args.max_seq_length}_shard_{shard_count}.pt"
                torch.save(current_shard, shard_filename)
                logger.info("Saved shard %d with %d instances to %s", shard_count, len(current_shard), shard_filename)
                shard_files.append(shard_filename)
                shard_lengths.append(len(current_shard))
                current_shard = []
                shard_count += 1
            idx += fixed_K  # prompt 增加一个 span 的长度

    # 保存剩余不足 chunk_size 数量的 shard
    if len(current_shard) > 0:
        shard_filename = f"{data_dir}/cached_{args.dataset_name}_{args.train_split}_maxlen{args.max_seq_length}_shard_{shard_count}.pt"
        torch.save(current_shard, shard_filename)
        logger.info("Saved shard %d with %d instances to %s", shard_count, len(current_shard), shard_filename)
        shard_files.append(shard_filename)
        shard_lengths.append(len(current_shard))

    index_data = {"shard_files": shard_files, "shard_lengths": shard_lengths}
    torch.save(index_data, index_filename)
    logger.info("Saved index file to %s with %d shards", index_filename, len(shard_files))
    return index_data


class PreprocessedShardedDataset(Dataset):
    """
    利用 shards index 实现惰性加载预处理数据。
    __getitem__ 根据全局索引确定对应 shard 及在 shard 内的索引，加载对应 shard（采用简单缓存）。
    """

    def __init__(self, index_data):
        self.shard_files = index_data["shard_files"]
        self.shard_lengths = index_data["shard_lengths"]
        self.cumulative_lengths = []
        total = 0
        for length in self.shard_lengths:
            total += length
            self.cumulative_lengths.append(total)
        self.total_length = total
        # 简单缓存：记录最后加载的 shard 索引及其内容
        self._cached_shard_index = None
        self._cached_shard = None

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} is out of range")
        # 找到对应的 shard 索引
        shard_idx = 0
        while idx >= self.cumulative_lengths[shard_idx]:
            shard_idx += 1
        # 计算在该 shard 内的相对索引
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_lengths[shard_idx - 1]
        # 如果缓存的不是当前 shard，则加载对应 shard 文件
        if self._cached_shard_index != shard_idx:
            shard_file = self.shard_files[shard_idx]
            self._cached_shard = torch.load(shard_file, map_location="cpu")
            self._cached_shard_index = shard_idx
        return self._cached_shard[local_idx]


def load_preprocessed_dataset(args, tokenizer, chunk_size=5000, data_dir=os.getcwd()):
    """
    检查 data_dir 目录下是否存在符合命名规则的分片文件：
      如果存在则构造 index 信息并创建 PreprocessedShardedDataset；
      否则调用 process_and_save_shards 对原始数据进行分片预处理。
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 查找符合命名规则的分片文件
    pattern = f"cached_{args.dataset_name}_{args.train_split}_maxlen{args.max_seq_length}_shard_*.pt"
    shard_files = sorted(glob.glob(pattern, root_dir=data_dir))
    if shard_files:
        shard_lengths = []
        logger.info("Found %d shard files.", len(shard_files))
        # 对每个分片文件加载后统计实例数量
        for sf in shard_files:
            try:
                data_chunk = torch.load(os.path.join(data_dir, sf), map_location="cpu")
                shard_lengths.append(len(data_chunk))
            except Exception as e:
                logger.error("Error loading shard file %s: %s", sf, e)
                shard_lengths.append(0)
        total_instances = sum(shard_lengths)
        logger.info("Total instances found in shards: %d", total_instances)
        index_data = {"shard_files": [os.path.join(data_dir, sf) for sf in shard_files], "shard_lengths": shard_lengths}
    else:
        logger.info("No cached shard files found. Processing raw data and building shards ...")
        index_data = process_and_save_shards(args, tokenizer, chunk_size, data_dir)
    return PreprocessedShardedDataset(index_data)


def training_collate_fn(batch, pad_token_id):
    """
    对 batch 内的实例进行 collate。
    每个实例为 dict {"prompt": list, "span": list, "K": fixed_K_value}。
    对 prompt 序列进行 pad（span 长度固定，可直接 stack）。
    为确保进入模型时 span 长度一致，以第一个实例的 K 为准，仅保留 K 值匹配的样本，
    并输出 warning 提示过滤掉的样本数。
    返回：(padded_prompts, padded_spans, current_K)
    """
    if not batch:
        return None

    # 以第一个样本的 K 为基准（这里所有实例均应使用相同的 fixed_K）
    current_K = batch[0]["K"]
    filtered_batch = [item for item in batch if item["K"] == current_K]
    num_dropped = len(batch) - len(filtered_batch)
    if num_dropped > 0:
        logger.warning("Dropped %d samples with mismatched K (expected %d) in current batch.", num_dropped, current_K)
    prompts = [torch.tensor(item["prompt"], dtype=torch.long) for item in filtered_batch]
    spans = [torch.tensor(item["span"], dtype=torch.long) for item in filtered_batch]
    padded_prompts = pad_sequence(prompts, batch_first=True, padding_value=pad_token_id)
    padded_spans = torch.stack(spans)  # 此处所有 span 长度均应为 current_K
    return padded_prompts, padded_spans, current_K