"""Model input contracts, shared by the app and offline evaluation."""


class ContextLimitError(ValueError):
    def __init__(self, tokens, limit):
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"Input has {tokens} model tokens; this engine supports {limit}. Nothing was scored.")


def bert_window(tokenizer, content_ids):
    """DistilBERT requires CLS/SEP on every window, including the last one."""
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("BERT tokenizer is missing CLS/SEP tokens")
    return {"input_ids": [tokenizer.cls_token_id, *content_ids, tokenizer.sep_token_id],
            "attention_mask": [1] * (len(content_ids) + 2)}


def bounded_inputs(tokenizer, text, limit, padding=False):
    """Reject unsupported lengths before inference, never silently truncate."""
    inputs = tokenizer(text, truncation=False, return_tensors="pt", return_attention_mask=True)
    count = inputs["input_ids"].shape[-1]
    if count > limit:
        raise ContextLimitError(count, limit)
    if padding:
        inputs = tokenizer.pad(inputs, padding="max_length", max_length=limit, return_tensors="pt")
    return inputs


def qwen_readout_inputs(tokenizer, torch, text, max_text_tokens):
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) > max_text_tokens:
        raise ContextLimitError(len(ids) + 1, max_text_tokens + 1)
    if tokenizer.eos_token_id is None:
        raise ValueError("Qwen tokenizer is missing its readout token")
    ids.append(tokenizer.eos_token_id)
    return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
