from transformers import GPT2Tokenizer as HF_GPT2Tokenizer
import torch

class Gpt2Tokenizer:
    def __init__(self, pretrained_model_name='gpt2'):
        self.tokenizer = HF_GPT2Tokenizer.from_pretrained(pretrained_model_name)
        special_tokens_dict = {
            'bos_token': '<BOS>',
            'eos_token': '<EOS>',
            'pad_token': '<PAD>',
            'additional_special_tokens': ['[PRED]']
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pred_token_id = self.tokenizer.additional_special_tokens_ids[0] if self.tokenizer.additional_special_tokens_ids else None

    def __call__(self, text, **kwargs):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    @property
    def vocab_size(self):
        return len(self.tokenizer)
