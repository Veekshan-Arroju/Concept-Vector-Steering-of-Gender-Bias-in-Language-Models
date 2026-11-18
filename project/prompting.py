"""
Single source of truth for turning a raw user message into the exact string the
model sees. Activation extraction (training data) and steering (generation) BOTH
call format_chat_prompt, so the two can never drift apart or apply the chat
template twice.

Position being read
-------------------
apply_chat_template(..., add_generation_prompt=True) ends with the assistant-turn
header. `.strip()` removes trailing whitespace/newlines after it, so the final token
(the one whose hidden state is extracted, rep_token=-1) is the last token of that
header, e.g. <|end_header_id|> for Llama-3.1. This is the same convention as the
authors' code.
"""


_WARNED_NO_TEMPLATE = set()


def has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def format_chat_prompt(tokenizer, user_message, add_generation_prompt=True):
    """
    Wrap a raw user message in the instruction-tuned model's chat template.

    Args:
        tokenizer             : HuggingFace tokenizer with apply_chat_template support (without a template the
                                plain prompt is used, with a warning).
        user_message          : Plain text from the user (no template applied yet).
        add_generation_prompt : Add the model's turn-start header (True for generation
                                and for activation extraction).

    Returns:
        Formatted prompt string, stripped of leading/trailing whitespace.
    """
    if not has_chat_template(tokenizer):
        # Base models (and some tokenizers) ship without a chat template. Fall back to the plain prompt, with
        # the BOS token if the tokenizer defines one (extraction tokenises with add_special_tokens=False).
        key = getattr(tokenizer, "name_or_path", "tokenizer")
        if key not in _WARNED_NO_TEMPLATE:
            _WARNED_NO_TEMPLATE.add(key)
            print(f"[format_chat_prompt] WARNING: {key} has no chat template; using the plain prompt. The token read for "
                  f"the hidden state is then the last token of the statement, not an assistant header, so results are "
                  f"not comparable with chat-template runs.")
        bos = getattr(tokenizer, "bos_token", None) or ""
        return bos + user_message.strip()

    chat = [{"role": "user", "content": user_message}]
    formatted = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return formatted.strip()


def describe_extraction_position(tokenizer, user_message, n_tokens=6):
    """
    Return the last `n_tokens` tokens of the formatted prompt, as strings.
    Printed once per extraction run so you can confirm which token is being read
    (and catch a chat template whose header ends somewhere unexpected, e.g. a
    reasoning model that appends '<think>').
    """
    formatted = format_chat_prompt(tokenizer, user_message)
    ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode([i]) for i in ids[-n_tokens:]]
