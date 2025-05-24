# SpanFusionLM/modules/tokenizer.py
from transformers import AutoTokenizer, GPT2Tokenizer

def build_tokenizer(tokenizer_name_or_path="gpt2", **kwargs):
    """
    Loads a tokenizer, preferably GPT2Tokenizer or AutoTokenizer,
    and adds special tokens required for SpanFusionLM.
    """
    try:
        # Try GPT2Tokenizer first if it's a known GPT-2 variant for specific behavior
        if "gpt2" in tokenizer_name_or_path.lower() or tokenizer_name_or_path == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)
    except Exception as e:
        print(f"Failed to load tokenizer {tokenizer_name_or_path} with specific class, trying AutoTokenizer. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)

    special_tokens_to_add = {
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
        'pad_token': '<PAD>',
        'additional_special_tokens': ['<|PRED|>'] # For span filling
    }
    
    # Check existing special tokens and add if not present or different
    current_special_tokens = {}
    if tokenizer.bos_token: current_special_tokens['bos_token'] = tokenizer.bos_token
    if tokenizer.eos_token: current_special_tokens['eos_token'] = tokenizer.eos_token
    if tokenizer.pad_token: current_special_tokens['pad_token'] = tokenizer.pad_token
    
    num_added_toks = tokenizer.add_special_tokens(special_tokens_to_add)
    
    if num_added_toks > 0:
        print(f"Added {num_added_toks} special tokens to tokenizer: {special_tokens_to_add}")

    # Ensure pad_token is set if it was None (common for GPT-2)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print(f"Warning: tokenizer.pad_token was None. Setting it to eos_token: {tokenizer.eos_token}")
            tokenizer.pad_token = tokenizer.eos_token
        else: # Fallback if EOS is also None, though unlikely after adding
            # This case needs careful handling, as padding is crucial.
            # Using a newly added special token if available, or erroring.
            # For now, assume <PAD> gets added and has an ID.
            if special_tokens_to_add['pad_token'] in tokenizer.get_vocab():
                 tokenizer.pad_token = special_tokens_to_add['pad_token']
            else: # Should not happen if add_special_tokens worked
                 raise ValueError("pad_token is None and could not be set. Please check tokenizer.")


    # Make len(tokenizer) return vocab_size
    if not hasattr(tokenizer, "__len__"):
        setattr(tokenizer, "__len__", lambda: tokenizer.vocab_size)
        
    # Store PRED token ID for convenience if needed, though config should handle it
    # pred_token_str = '<|PRED|>'
    # if pred_token_str in tokenizer.additional_special_tokens:
    #     tokenizer.pred_token_id = tokenizer.convert_tokens_to_ids(pred_token_str)
    # else:
    #     # This case means <|PRED|> wasn't added or found, which is an issue.
    #     # The config's __post_init__ will handle getting the ID.
    #     pass

    return tokenizer
