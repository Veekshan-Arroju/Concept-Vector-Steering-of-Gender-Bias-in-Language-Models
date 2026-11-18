"""Tiny random models + a toy tokenizer with a Llama-3-style chat template (no downloads, CPU)."""
import warnings
warnings.filterwarnings("ignore")
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                          LlamaForCausalLM, PreTrainedTokenizerFast)

LLAMA3_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


def make_tokenizer(texts, with_template=True):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<unk>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    trainer = trainers.BpeTrainer(vocab_size=700, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(list(texts), trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                   unk_token="<unk>", pad_token="<|eot_id|>")
    fast.padding_side = "left"
    if with_template:
        fast.chat_template = LLAMA3_TEMPLATE
    return fast


def make_llama(tok, seed=0, layers=4):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    return LlamaForCausalLM(cfg).eval()


def make_neox(tok, seed=0, layers=4):
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                        num_attention_heads=4, max_position_embeddings=1024,
                        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    return GPTNeoXForCausalLM(cfg).eval()


def make_gpt2(tok, seed=0, layers=4):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=len(tok), n_embd=32, n_layer=layers, n_head=4, n_positions=1024,
                     bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    return GPT2LMHeadModel(cfg).eval()
