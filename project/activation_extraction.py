"""
Activation extraction (v4).

Runs every prompt in the REGISTRY (prompts.csv, written by dataset_builder.py) through the
frozen model exactly once and stores the final-token hidden state of every block, one file
per layer (see activation_store.py). Training sets and the paired test are just lists of
prompt_ids into that registry, so nothing is extracted twice.

What changed vs. v3
-------------------
* The chat template is applied HERE, per prompt, through prompting.format_chat_prompt
  (the same function the steering scripts use). v3 fed raw text, so the hidden state it
  read was the last character of the statement, not the "about to answer" position the
  authors read. The template already contains the BOS token, so tokenising with
  add_special_tokens=False (kept from v3) is correct.
* The dataset files are model-agnostic raw prompts; only this step knows the model.
* Output is keyed by prompt_id, not by train/test split.

The model loader, layer selection and extract_hidden_states() are unchanged from v3.
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import experiment
import activation_store as store
from dataset_builder import SEED
from model_utils import add_model_args, get_hidden_size, get_num_layers, load_causal_lm   # one loader / layer lookup for every model
from dataset_io import load_manifest, load_registry
from activation_store import ACTIVATIONS_DIR, save_prompt_activations
from prompting import describe_extraction_position, format_chat_prompt

load_dotenv()

torch.manual_seed(SEED)                       # deterministic CPU ops
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)          # deterministic CUDA ops

# Forward-pass batch size - balance between speed and GPU VRAM usage. Matches paper's hyperparams['forward_batch_size'] default of 16.
DEFAULT_BATCH_SIZE: int = 16

# Prompts per call to extract_hidden_states(). Prompts are padded to the longest one in a
# chunk, so smaller chunks waste less compute; this is also the memory bound for the
# padded token tensor. Does not change the results beyond fp16 rounding.
DEFAULT_CHUNK_SIZE: int = 256


# -----------------------------------------------------------------------
#  1. Model Loading
#  load_causal_lm() now lives in model_utils.py (imported above): one loader for every model, with
#  --dtype and --trust-remote-code options (needed e.g. for sarvamai/sarvam-30b).
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
#  2. Layer Selection
#  Reproduces NeuralController.__init__() in neural_controllers.py: self.hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
# -----------------------------------------------------------------------
def select_layers(model) -> list:
    """
    Build the list of transformer block indices to extract activations from,
    matching the paper's per-block scheme: A_l(X) for l ∈ {2, ..., L}.

    The paper skips:
    - The embedding output (block 1 / A_1): always excluded (paper Eq. A1).
    - The very first transformer block (block 2): excluded by the range formula.
      Only blocks 3 through L are covered (negative indices -1 to -(L-2)).

    Negative index convention (matching the paper's code):
        -1  → last transformer block  (block L)
        -2  → second-to-last          (block L-1)
        -(num_hidden_layers - 1) → third transformer block (block 3)

    The positive equivalent is computed inside extract_hidden_states via:
        pos_idx = num_layers + layer_idx   (e.g. -1 → num_layers - 1)

    Args:
        model : Loaded AutoModelForCausalLM

    Returns:
        List of negative int indices, e.g. [-1, -2, ..., -(L-1)]
    """
    num_layers = get_num_layers(model)
    # range(-1, -num_layers, -1) → [-1, -2, ..., -(num_layers - 1)]
    # Covers (num_layers - 1) transformer blocks in reverse order.
    hidden_layers = list(range(-1, -num_layers, -1))

    print(
        f"[select_layers] Tracking {len(hidden_layers)} layers: "
        f"{hidden_layers[:3]} ... {hidden_layers[-3:]}"
    )
    return hidden_layers


# -----------------------------------------------------------------------
#  3. Core Activation Extraction
#  Direct adaptation of direction_utils.get_hidden_states() for text-only causal LMs (no multimodal / seq2seq branches needed for this task).
# -----------------------------------------------------------------------
def extract_hidden_states(
    prompts: list,
    model,
    tokenizer,
    hidden_layers: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rep_token: int = -1,
    all_positions: bool = False,
) -> dict:
    """
    Extract per-block activations a_l^(i) for each prompt and each layer l.

    Paper definition (Section 3.1):
        a_l^(i) = last row of A_l(X^(i)) ∈ R^k
        ↕ (in transformer terms)
        a_l^(i) = hidden_state_l[:, -1, :]   (final token position)

    This function is the workhorse of the pipeline. It batches prompts through
    the frozen model and accumulates the final-token hidden state from each
    selected transformer block, building the data matrix Z_l for every layer.

    ---- Information-leakage safeguards (caller-enforced) ----
    NEVER pass a mix of train and test prompts to a single call. The caller
    (main()) is responsible for invoking this function separately for each
    split. Additional safeguards within this function:
    • torch.no_grad()  : no gradient information is stored anywhere
    • shuffle=False    : prompts stay in the same order as their labels
    • model.eval()     : already set by load_causal_lm()

    Args:
        prompts       : List[str] — texts for ONE split (train OR test only)
        model         : Loaded causal LM in eval() mode
        tokenizer     : Matching left-padding tokenizer
        hidden_layers : Layer indices from select_layers() (negative ints)
        batch_size    : Prompts per forward pass (tune to fit GPU VRAM)
        rep_token     : Token position to read from; -1 = final token (paper default)
        all_positions : If True, return full sequence of shape (N, T, D).
                        If False (default), return final-token only (N, D),
                        matching the paper's a_l^(i) definition.

    Returns:
        Dict[int → Tensor] keyed by the ORIGINAL (negative) layer indices,
        each Tensor shaped (N, hidden_dim)   when all_positions=False
                       or (N, seq_len, hidden_dim) when all_positions=True,
        where N = len(prompts).
    """
    # ---- Tokenise all prompts with left-padding ----
    # The paper does NOT add special tokens here (add_special_tokens=False) for causal LMs — prompts are already formatted as needed.
    # Attention mask is cast to float16 to match direction_utils.py exactly.
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,               # pad sequences to same length within batch
        add_special_tokens=False,   # paper excludes extra special tokens for CausalLM
    )
    # Cast attention mask to fp16 — mirrors paper's direction_utils.py line exactly
    encoded["attention_mask"] = encoded["attention_mask"].half()

    # ---- Build DataLoader — shuffle MUST be False ----
    # Preserving order guarantees that activations[i] corresponds to labels[i]. Shuffling here would silently corrupt the prompt-to-label mapping.
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ---- Internal storage keyed by POSITIVE index ----
    # The paper converts negative → positive indices before storing, then converts back to negative indices for the returned dict. 
    # This prevents dict key collisions (e.g., -1 and num_layers-1 both pointing to last block).
    num_layers = get_num_layers(model)
    pos_to_batches: dict = {}  # positive_index → list of per-batch tensors
    for layer_idx in hidden_layers:
        if isinstance(layer_idx, int):
            # Convert negative index to positive 0-based block index
            # e.g., -1 → num_layers - 1  (last transformer block)
            pos_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
            if 0 <= pos_idx < num_layers:   # validate range before storing
                pos_to_batches[pos_idx] = []

    # ---- Forward passes through the frozen model ----
    with torch.no_grad():   # pure inference — no gradient computation anywhere
        for batch_ids, batch_mask in tqdm(dataloader, desc="Extracting activations"):
            # Move batch to the model's device (CPU, single GPU, or sharded)
            batch_ids  = batch_ids.to(model.device)
            batch_mask = batch_mask.to(model.device)

            # Forward pass with all hidden states requested.
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors:
            #   index 0 → embedding output A_1(X)    [excluded — paper Eq. A1]
            #   index 1 → block 2 output  A_2(X)     [first transformer block]
            #   ...
            #   index L-1 → block L output A_L(X)   [last transformer block]
            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                output_hidden_states=True,   # request all intermediate activations
                return_dict=True,
            )

            # Skip index 0 (embedding A_1) — paper always excludes it. Mirrors paper: out_hidden_states = list(out_hidden_states)[1:]
            transformer_block_outputs = list(outputs.hidden_states)[1:]

            # Iterate over transformer blocks; enumerate gives 0-based positive index
            for pos_idx, hidden_state in enumerate(transformer_block_outputs):
                if pos_idx not in pos_to_batches:
                    # This block index was not selected — skip it
                    continue

                if all_positions:
                    # Store the full sequence: shape (batch, seq_len, hidden_dim), Needed for positional analyses; not the paper default.
                    pos_to_batches[pos_idx].append(hidden_state.detach().cpu())
                else:
                    # Paper default: extract a_l^(i) = A_l(X^(i))[-1, :] i.e., the hidden state at the FINAL TOKEN position.
                    # hidden_state shape: (batch, seq_len, hidden_dim)
                    # [:, rep_token, :] → (batch, hidden_dim)  (rep_token = -1)
                    final_tok_rep = hidden_state[:, rep_token, :].detach().cpu()
                    pos_to_batches[pos_idx].append(final_tok_rep)

    # ---- Concatenate all batches and re-key to ORIGINAL (negative) indices ----
    # torch.cat along dim=0 stacks (batch, hidden_dim) tensors into (N, hidden_dim).
    # The returned dict uses the same negative indices as the input hidden_layers, matching the convention throughout the paper's codebase.
    hidden_states_by_layer: dict = {}
    for orig_idx in hidden_layers:
        if isinstance(orig_idx, int):
            pos_idx = orig_idx if orig_idx >= 0 else num_layers + orig_idx
            # Concatenate list of per-batch tensors → (N, hidden_dim) or (N, T, D)
            hidden_states_by_layer[orig_idx] = torch.cat(pos_to_batches[pos_idx], dim=0)

    return hidden_states_by_layer




# -----------------------------------------------------------------------
#  4. Registry extraction (replaces the v3 train/test-split pipeline)
# -----------------------------------------------------------------------
def _dry_run_indices(chat_prompts: list, tokenizer, limit: int) -> list:
    """`limit` representative prompts: evenly spaced through the registry, plus the LONGEST one, so that the
    time and memory estimates of a dry run are not optimistic. limit >= len(prompts) returns every index."""
    n = len(chat_prompts)
    if limit >= n:
        return list(range(n))
    idx = set(np.linspace(0, n - 1, max(1, limit - 1)).astype(int).tolist())
    lengths = [len(x) for x in tokenizer(chat_prompts, add_special_tokens=False)["input_ids"]]
    idx.add(int(np.argmax(lengths)))
    return sorted(idx)


def _extract_chunks(chat_prompts: list, model, tokenizer, hidden_layers: list, batch_size: int, chunk_size: int) -> tuple:
    """Forward passes in chunks with timing; returns ({layer -> Tensor(n, d)}, elapsed seconds)."""
    per_layer_chunks = {l: [] for l in hidden_layers}
    n = len(chat_prompts)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    for start in range(0, n, chunk_size):
        chunk = chat_prompts[start:start + chunk_size]
        t0 = time.perf_counter()
        out = extract_hidden_states(chunk, model, tokenizer, hidden_layers, batch_size=batch_size)
        for l in hidden_layers:
            per_layer_chunks[l].append(out[l])
        done = start + len(chunk)
        dt, elapsed = time.perf_counter() - t0, time.perf_counter() - t_all
        print(f"[extract] prompts {start}-{done - 1} of {n}: {dt:.1f} s ({dt / len(chunk):.3f} s per prompt) | "
              f"elapsed {elapsed / 60:.1f} min | remaining about {(n - done) * elapsed / done / 60:.1f} min")
    per_layer = {l: torch.cat(chunks, dim=0) for l, chunks in per_layer_chunks.items()}
    elapsed = time.perf_counter() - t_all
    if torch.cuda.is_available():
        print(f"[extract] Peak GPU memory of this process: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved (nvidia-smi shows about 0.5 GiB more: CUDA context)")
    return per_layer, elapsed


def extract_registry_activations(
    model,
    tokenizer,
    model_id: str,
    hidden_layers: list = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int = None,
) -> dict:
    """
    Extract final-token hidden states for every registry prompt and save them.

    Args:
        model, tokenizer : loaded causal LM (eval mode) and its LEFT-padding tokenizer
        model_id         : used for the storage namespace (must match MODEL_ID downstream)
        hidden_layers    : negative layer indices; default = select_layers(model)
        limit            : DRY RUN: extract only `limit` representative prompts (evenly spaced + the longest one), report
                           speed, a projection for the full run and peak GPU memory, and save NOTHING

    Returns:
        {layer_idx -> Tensor(n_prompts, d)}  (row i == prompt_id i)
    """
    registry = load_registry()
    manifest = load_manifest()
    n_registry = len(registry)
    hidden_layers = hidden_layers if hidden_layers is not None else select_layers(model)

    # ---- apply the chat template (same function as steering) ----
    chat_prompts = [format_chat_prompt(tokenizer, p) for p in registry["user_prompt"].tolist()]
    if limit is not None:
        chat_prompts = [chat_prompts[i] for i in _dry_run_indices(chat_prompts, tokenizer, limit)]

    # ---- show exactly which token will be read, for one prompt of each kind ----
    print("\n[extract] Extraction position check (last tokens of the formatted prompt):")
    for kind in ("neutral", "subtype"):
        first = registry[registry["kind"] == kind].iloc[0]["user_prompt"]
        print(f"  {kind:<8} -> {describe_extraction_position(tokenizer, first)}")
    print(f"  example formatted prompt:\n{chat_prompts[0]!r}\n")

    # ---- forward passes, chunk by chunk ----
    n = len(chat_prompts)
    per_layer, elapsed = _extract_chunks(chat_prompts, model, tokenizer, hidden_layers, batch_size, chunk_size)
    if limit is not None:
        print(f"[extract] DRY RUN: {n} prompts took {elapsed:.0f} s = {elapsed / n:.3f} s per prompt. The full run "
              f"({n_registry} prompts) would take about {elapsed / n * n_registry / 60:.0f} min at this speed "
              f"(the first chunk includes warm-up, so this is slightly pessimistic). Nothing was saved.")

    sample = per_layer[hidden_layers[0]]
    assert sample.shape[0] == n, f"row mismatch: {sample.shape[0]} activations for {n} prompts"
    assert sample.shape[1] == get_hidden_size(model), "hidden dimension does not match the model config"

    if limit is None:
        last_tokens = {k: describe_extraction_position(tokenizer, registry[registry["kind"] == k].iloc[0]["user_prompt"])
                       for k in ("neutral", "subtype")}
        save_prompt_activations(
            per_layer, model_id, manifest["registry_sha1"],
            extra_meta={"last_tokens_example": last_tokens, "chunk_size": chunk_size,
                        "experiment": manifest.get("experiment"), "language": manifest.get("language")},
        )
    return per_layer


def extend_registry_activations(model, tokenizer, model_id: str, source_experiment: str,
                                batch_size: int = DEFAULT_BATCH_SIZE, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
    """
    Build the activation store of the CURRENT experiment by REUSING the stored rows of `source_experiment` and extracting
    only the prompts that are new (about 420 for the matched / stance experiments, instead of 6,562).

    It only does so when it is provably safe:
      * the source store is fresh for its own dataset (registry hash matches),
      * the source registry is an exact PREFIX of this experiment's registry (every prompt identical, in the same order),
      * the model and dtype are the same as the source extraction (same hidden size, same dtype).
    The new store holds the source rows unchanged, followed by the newly extracted rows, and records where it came from
    (meta.json: extended_from, n_reused, n_extracted).
    """
    dest = experiment.get_experiment()
    if source_experiment == dest:
        raise ValueError("--extend-from must name a DIFFERENT experiment than the current one")
    registry, manifest = load_registry(), load_manifest()

    with experiment.using(source_experiment):
        src_meta = store.check_fresh(model_id)               # raises if the source activations are missing or stale
        src_registry = load_registry()
        src_tensors = {l: store._load_tensor(store.layer_path(model_id, l)) for l in src_meta["layers"]}
    n_src = len(src_registry)
    if len(registry) <= n_src or registry["user_prompt"].iloc[:n_src].tolist() != src_registry["user_prompt"].tolist():
        raise ValueError(f"The registry of experiment {dest!r} does not start with the registry of {source_experiment!r}, so its "
                         f"activations cannot be reused. Rebuild {dest!r} with the same --seed/--language/--frame-mode as "
                         f"{source_experiment!r} (and the same --negatives as the source build; stance and matched both extend \"main\").")
    layers = src_meta["layers"]
    if get_hidden_size(model) != src_meta["hidden_dim"]:
        raise ValueError(f"hidden size {get_hidden_size(model)} differs from the source store ({src_meta['hidden_dim']}): different model?")
    p_dtype = str(next(model.parameters()).dtype)
    if p_dtype != src_meta["dtype"]:
        raise ValueError(f"The model is loaded as {p_dtype} but the source activations are {src_meta['dtype']}. Mixing precisions inside one "
                         f"store is not allowed: run with --dtype {src_meta['dtype'].replace('torch.', '')} (and the same device type).")

    new_prompts = registry["user_prompt"].iloc[n_src:].tolist()
    print(f"\n[extend] reusing {n_src} stored prompts from experiment {source_experiment!r}; extracting {len(new_prompts)} new ones")
    chat = [format_chat_prompt(tokenizer, p) for p in new_prompts]
    print(f"  example new prompt (formatted):\n{chat[0]!r}\n")
    new, elapsed = _extract_chunks(chat, model, tokenizer, layers, batch_size, chunk_size)
    assert new[layers[0]].shape[0] == len(new_prompts)
    combined = {l: torch.cat([src_tensors[l], new[l].to(src_tensors[l].dtype)], dim=0) for l in layers}
    save_prompt_activations(
        combined, model_id, manifest["registry_sha1"],
        extra_meta={"last_tokens_example": src_meta.get("last_tokens_example"), "chunk_size": chunk_size,
                    "experiment": dest, "language": manifest.get("language"),
                    "extended_from": source_experiment, "n_reused": n_src, "n_extracted": len(new_prompts)},
    )
    print(f"[extend] done in {elapsed / 60:.1f} min: {n_src} reused + {len(new_prompts)} extracted = {len(registry)} prompts")
    return combined


def run_extraction(model, tokenizer, model_id: str, **kwargs) -> dict:
    """Shared entry point: loads nothing itself, so it works with any already-loaded model/tokenizer."""
    manifest = load_manifest()
    print(f"\n[main] Model         : {model_id}")
    print(f"[main] Prompts       : {manifest['n_prompts']} unique (registry {manifest['registry_sha1'][:10]}...)")
    per_layer = extract_registry_activations(model, tokenizer, model_id, **kwargs)
    sample_layer = next(iter(per_layer))
    print(f"\n[main] Activation extraction complete.")
    print(f"[main]   Layers extracted : {len(per_layer)}")
    print(f"[main]   Hidden dim       : {per_layer[sample_layer].shape[1]}")
    print(f"[main]   Prompts          : {per_layer[sample_layer].shape[0]}")
    print(f"[main]   Activations dir  : {ACTIVATIONS_DIR.resolve()}")
    return per_layer


def main(model_id: str = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"), dtype: str = "auto",
         trust_remote_code: bool = False, extend_from: str = None, **kwargs):
    """Load the model, then extract + save activations for every registry prompt (or, with extend_from, only the new ones)."""
    if extend_from and kwargs.get("limit") is not None:
        raise ValueError("--limit (dry run) cannot be combined with --extend-from")
    model, tokenizer = load_causal_lm(model_id, dtype=dtype, trust_remote_code=trust_remote_code)
    if extend_from:
        kwargs.pop("limit", None)
        return extend_registry_activations(model, tokenizer, model_id, extend_from, **kwargs)
    return run_extraction(model, tokenizer, model_id, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract final-token hidden states for every registry prompt.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="DRY RUN: extract only N representative prompts, report speed / peak GPU memory, save nothing")
    parser.add_argument("--extend-from", default=None, metavar="EXPERIMENT",
                        help="reuse the stored activations of EXPERIMENT (its registry must be a prefix of this one) and extract only the new prompts")
    add_model_args(parser)
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    main(args.model_id, dtype=args.dtype, trust_remote_code=args.trust_remote_code, extend_from=args.extend_from,
         batch_size=args.batch_size, chunk_size=args.chunk_size, limit=args.limit)
