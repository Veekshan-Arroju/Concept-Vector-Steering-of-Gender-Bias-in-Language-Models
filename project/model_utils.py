"""
Model-agnostic helpers: loading, layer lookup, sizes.

Nothing else in the pipeline assumes a particular architecture, so a new model (Llama, Mistral, Gemma, Qwen,
GPT-2/NeoX style, Sarvam, other Indian-language LLMs, ...) needs no code changes as long as
  * it loads with AutoModelForCausalLM (pass --trust-remote-code if it ships custom code),
  * its decoder blocks can be found (get_decoder_layers tries known layouts, then scans for the block list).
If a model still fails, the error message lists what was tried and how to fix it.
"""

import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpu_utils import describe_device, warn_if_multi_gpu

# Attribute paths (relative to the model object) of the ModuleList that holds the decoder blocks.
_CANDIDATE_PATHS = [
    "model.layers",                  # Llama / Mistral / Gemma / Qwen / Phi
    "transformer.h",                 # GPT-2 / GPT-J / Falcon / BLOOM
    "gpt_neox.layers",               # GPT-NeoX / Pythia
    "transformer.layers",            # some custom architectures
    "model.decoder.layers",          # OPT / BART-style decoders
    "backbone.layers",               # some MoE / state-space models
    "model.language_model.layers",   # multimodal wrappers
    "language_model.model.layers",   # multimodal wrappers
]

_NUM_LAYER_ATTRS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
_HIDDEN_ATTRS = ("hidden_size", "n_embd", "d_model", "dim")


def _config_candidates(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return []
    out = [cfg]
    for sub in ("text_config", "language_config", "llm_config"):
        if getattr(cfg, sub, None) is not None:
            out.append(getattr(cfg, sub))
    return out


def _config_value(model, attrs):
    for cfg in _config_candidates(model):
        for a in attrs:
            v = getattr(cfg, a, None)
            if isinstance(v, int) and v > 0:
                return v
    return None


def get_decoder_layers(model):
    """Return the ModuleList of decoder blocks of `model`."""
    for path in _CANDIDATE_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, (torch.nn.ModuleList, list)) and len(obj) > 0:
            return obj

    # Fallback: scan for a ModuleList whose length equals the configured number of layers.
    n = _config_value(model, _NUM_LAYER_ATTRS)
    cands = [(name, m) for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) > 0 and (n is None or len(m) == n)]
    if cands:
        preferred = [c for c in cands if any(k in c[0].lower() for k in ("layer", ".h", "block"))]
        name, mod = (preferred or cands)[0]
        print(f"[get_decoder_layers] Found decoder blocks by scanning: model.{name} ({len(mod)} blocks)")
        return mod

    raise AttributeError(
        "Could not locate the decoder blocks on this model. Tried: "
        + ", ".join(f"model.{p}" for p in _CANDIDATE_PATHS)
        + ". Run print(model) to find the attribute path of the block list and add it to _CANDIDATE_PATHS in model_utils.py."
    )


def get_num_layers(model) -> int:
    return len(get_decoder_layers(model))


def get_hidden_size(model) -> int:
    v = _config_value(model, _HIDDEN_ATTRS)
    if v is not None:
        return v
    return int(model.get_input_embeddings().weight.shape[1])


_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def parse_dtype(name):
    """'float16' | 'bfloat16' | 'float32' | 'auto' | None -> torch dtype (None/'auto' = float16 on GPU, float32 on CPU)."""
    if isinstance(name, torch.dtype):
        return name
    if name is None or str(name).lower() == "auto":
        return torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
    key = str(name).lower()
    if key not in _DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; use one of {sorted(_DTYPES)} or 'auto'")
    return _DTYPES[key]


def build_max_memory(value: str = None, n_gpus: int = None):
    """
    The environment variable MAX_GPU_GIB caps how much of EACH visible GPU the model may take (for shared GPUs):
        MAX_GPU_GIB=9 CUDA_VISIBLE_DEVICES=0,1 python steering_eval.py ...
    puts a model of about 15 GiB across two GPUs at about 7.5 GiB each, never more than 9 GiB on either.
    Returns {gpu_index: "9GiB", ...} for transformers' max_memory, or None when the variable is unset or no GPU is visible.
    It is read here, in the one loader every script uses, so no script can silently ignore it.
    """
    value = os.getenv("MAX_GPU_GIB") if value is None else value
    if not value or not str(value).strip():
        return None
    try:
        gib = float(value)
    except ValueError:
        raise ValueError(f"MAX_GPU_GIB must be a number of GiB (for example 9), got {value!r}")
    if gib <= 0:
        raise ValueError(f"MAX_GPU_GIB must be positive, got {gib}")
    n = torch.cuda.device_count() if n_gpus is None else n_gpus
    if n == 0:
        return None
    return {i: f"{gib:g}GiB" for i in range(n)}


def load_causal_lm(model_id: str, dtype=None, device_map: str = "auto", trust_remote_code: bool = False):
    """
    Load a HuggingFace causal LM + tokenizer for inference (one loader for every model).

    Choices that mirror the paper's utils.load_model:
      * padding_side="left"  so [:, -1, :] is always the real final token, even in padded batches
      * pad token = EOS when the tokenizer has none
      * model.eval()         inference only

    Args:
        dtype             : 'float16' | 'bfloat16' | 'float32' | 'auto' (default). Use bfloat16 for MoE models such as Sarvam.
        trust_remote_code : needed for models that ship custom modelling code (e.g. sarvamai/sarvam-30b).
    """
    torch_dtype = parse_dtype(dtype)
    if device_map == "auto":
        warn_if_multi_gpu("device_map='auto'")
    print(f"[load_causal_lm] Device: {describe_device()}")
    print(f"[load_causal_lm] Loading tokenizer: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    architectures = getattr(config, "architectures", []) or []
    use_fast = "LlamaForCausalLM" not in architectures   # mirrors the paper's logic

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side="left", legacy=False, use_fast=use_fast, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  -> pad_token_id set to eos_token_id ({tokenizer.pad_token_id})")

    print(f"[load_causal_lm] Loading model weights: {model_id}  (dtype {torch_dtype})")
    max_memory = build_max_memory() if device_map == "auto" else None
    extra = {}
    if max_memory:
        extra["max_memory"] = max_memory
        print(f"[load_causal_lm] GPU memory cap per visible GPU (MAX_GPU_GIB): {sorted(set(max_memory.values()))[0]}; "
              f"the model is spread over {len(max_memory)} GPU(s) within it")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, device_map=device_map, trust_remote_code=trust_remote_code, **extra,
    )
    model.eval()
    print(f"  -> {get_num_layers(model)} transformer layers | hidden_dim = {get_hidden_size(model)} | dtype = {torch_dtype}")
    report_placement(model)
    return model, tokenizer


def report_placement(model) -> dict:
    """
    Print where the model's weights ended up and warn if device_map="auto" offloaded part of it to the CPU or disk
    (it does that silently when the GPU has too little free memory, which makes extraction extremely slow).
    Returns {device: number_of_modules} ({} if the model has no device map, e.g. a plain CPU model).
    """
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return {}
    counts = {}
    for dev in dm.values():
        counts[str(dev)] = counts.get(str(dev), 0) + 1
    print(f"[placement] modules per device: {counts}")
    offloaded = [d for d in counts if d in ("cpu", "disk")]
    if offloaded and torch.cuda.is_available():
        print(f"\n[placement] WARNING: part of the model is on {offloaded} because the visible GPU(s) did not have enough free "
              f"memory. Extraction will be MUCH slower. Stop (Ctrl+C) and wait for a GPU with at least ~19 GB free "
              f"(check nvidia-smi), or use a smaller --batch-size only if the weights alone fit.\n")
    return counts


def add_model_args(parser) -> None:
    """--dtype and --trust-remote-code, shared by every script that loads a model."""
    parser.add_argument("--dtype", default="auto", help="float16 | bfloat16 | float32 | auto (default: float16 on GPU, float32 on CPU)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="allow custom modelling code from the model repo (needed for e.g. sarvamai/sarvam-30b)")
