"""
Anti-misogyny generation steering using learned per-layer concept vectors.

Implements the additive activation steering method from the paper.

Core perturbation (Eq. 2 from the paper, applied at every token position i): A_tilde_{l,i}(X) = A_{l,i}(X) + epsilon * v_l

How the sign of epsilon works for our case:
  - The concept vectors v_l point toward label=1, i.e. prompts that carry a misogyny instruction (as opposed to the neutral, instruction-free label=0 prompts).
  - Using epsilon > 0 would push generation TOWARD misogynistic content.
  - Using epsilon < 0 (default here) pushes generation AWAY from misogynistic content, nudging the model toward more balanced, non-misogynistic framing.
  - This is exactly analogous to the paper's anti-deception steering in Fig. 2C, where epsilon < 0 steered DeepSeek toward honest behavior.

The concept vectors v_l point from neutral prompts toward misogyny-instruction prompts (from concept-vector training). 
Using epsilon < 0 at inference time additively perturbs block activations AWAY from that direction, producing more balanced, non-misogynistic generations.

This is the same mechanism the paper uses for anti-deception steering in Fig. 2C
where epsilon < 0 steered DeepSeek toward honest behavior.
  
No training data, labels, or gradients are used at inference time.
Steering is purely a forward-pass hook on pre-computed v_l directions, which means there is zero information leakage from the training split.
"""

import os
from dotenv import load_dotenv
import pickle
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from tqdm import tqdm
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from coef_calibration import position_mask_applies
from concept_vector_training import DEFAULT_VECTOR_NAME, load_concept_vectors, CONCEPT_VECTORS_DIR
from experiment import PROJECT_ROOT, exp_dir
from model_utils import add_model_args, get_decoder_layers, load_causal_lm
# The ONE place the chat template is applied. Activation extraction imports the same function,
# so the training prompts and the steering prompts are formatted identically.
from prompting import format_chat_prompt   # noqa: F401  (re-exported: other scripts import it from here)

# Steering coefficient
# Directions point toward label=1 (misogyny instruction present). A negative epsilon reverses that signal and moves generation away from it.
# Typical usable range is -0.1 to -0.5; going beyond -0.5 often causes repetitive or incoherent output (the paper calls this "nonsensical").
DEFAULT_CONTROL_COEF = -0.2


# -----------------------------------------------------------------------
# 1. Hook utilities  (additive steering -- paper Eq. 2)
# -----------------------------------------------------------------------

def hook_model(model, directions, layers_to_control, control_coef, positions="all"):
    """
    Register PyTorch forward hooks on the specified transformer blocks.
    Each hook fires after the block computes its output and adds epsilon * v_l to every token position's hidden state.

    The hook signature (module, input, output) is required by PyTorch; only output is actually used here.

    Args:
        model            : Loaded HuggingFace causal LM (e.g. LlamaForCausalLM).
        directions       : dict  {layer_idx (int, negative) -> tensor of shape (n_components, hidden_dim) or (hidden_dim,)}.
                           The top eigenvector (component 0) is the steering vector.
        layers_to_control: list of negative int layer indices to hook, e.g. list(range(-1, -32, -1)) hooks all 31 blocks.
        control_coef     : Scalar epsilon.  Negative = steer away from misogyny.
        positions        : "all" (default, the paper's setting: every position of every pass) or "generated" (only the decoding steps, not the prompt pass).

    Returns:
        hooks: dict {layer_idx -> torch hook handle}. Pass this to clear_hooks() once generation is done.
    """
    hooks = {}

    for layer_idx in layers_to_control:

        # concept_vector_training.py always saves one direction per layer as shape (1, d), so indexing [0] gives the (d,) steering vector. 
        # This produces the same broadcasted result as the paper's hook_model, which just uses the (1, d) tensor directly.
        control_vec = directions[layer_idx][0]

        # Reshape to (1, 1, hidden_dim) so it broadcasts over the output tensor of shape (batch, seq_len, hidden_dim).
        if len(control_vec.shape) == 1:
            control_vec = control_vec.reshape(1, 1, -1)

        # Access the actual transformer block at this index. get_decoder_layers finds the list of decoder blocks on any
        # supported architecture (Llama / Mistral / Gemma / Qwen: model.layers, GPT-2/NeoX style, Sarvam, ...).
        block = get_decoder_layers(model)[layer_idx]

        def block_hook(module, input, output,
                       control_vec=control_vec,
                       control_coef=control_coef,
                       positions=positions):
            """
            Adds epsilon * v_l to the block's output hidden states.
            module and input are unused but required by the PyTorch hook API.
            """
            # output[0] holds the hidden states tensor
            # hidden_states = output[0]
            hidden_states = output[0] if isinstance(output, tuple) else output

            if not position_mask_applies(hidden_states.shape[1], positions):
                return None                                  # a forward hook that returns None leaves the output unchanged

            # Additive perturbation -- the core of Equation 2
            hidden_states = hidden_states + control_coef * control_vec.to(
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            # Blocks return a tuple (hidden_states, optional_present_key_value, ...).
            # Reconstruct that tuple so the rest of the network receives the right format.
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]

            return hidden_states

        hook_handle = block.register_forward_hook(block_hook)
        hooks[layer_idx] = hook_handle

    return hooks


def clear_hooks(hooks):
    """
    Remove all registered forward hooks.

    Always call this after generation finishes -- even inside a finally block --
    so stale hooks don't accidentally pollute subsequent calls.

    Args:
        hooks: dict returned by hook_model.
    """
    for hook_handle in hooks.values():
        hook_handle.remove()



# -----------------------------------------------------------------------
# 2. Text generation wrapper
# -----------------------------------------------------------------------

def generate_on_text(model, tokenizer, input_text, **kwargs):
    """
    Tokenize a formatted prompt and run model.generate().

    If forward hooks are registered (via hook_model) before this is called, they fire automatically during each token's forward pass. 

    Args:
        model      : The transformer model.
        tokenizer  : Matching tokenizer.
        input_text : Already-formatted prompt string (chat template applied).
        **kwargs   : Forwarded to model.generate (max_new_tokens, do_sample, etc.)

    Returns:
        generated_text: Decoded string including the prompt tokens.
    """
    # If max_new_tokens is explicitly provided, clear max_length so only one length constraint is active. 
    # Without this, the model's default GenerationConfig (e.g. TinyLlama ships with max_length=2048) conflicts with max_new_tokens and
    # transformers prints a noisy warning on every single generate() call.
    if "max_new_tokens" in kwargs:
        kwargs["max_length"] = None

    # add_special_tokens=False because the chat template already includes the BOS token
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    # with torch.no_grad():
    #     output_ids = model.generate(**inputs, **kwargs)
    kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)

    # Decode the full sequence (prompt + new tokens)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return generated_text


# -----------------------------------------------------------------------
# 3. Prompt formatting
# -----------------------------------------------------------------------

_STOP_STRINGS = ("<|eot_id|>", "<|im_end|>", "<|end|>", "</s>", "<eos>", "<|endoftext|>")


def extract_response_only(full_text: str, formatted_prompt: str = None) -> str:
    """
    Strip the echoed prompt/template off generate_on_text()'s output, leaving just the assistant's response.

    generate_on_text() decodes the full sequence (prompt + new tokens), which makes any character-based preview
    (e.g. text[:300]) mostly show template boilerplate. Works for any model:
      1. if `formatted_prompt` (format_chat_prompt(tokenizer, prompt)) is given and found in the text, return what follows it;
      2. otherwise, for Llama-3-style templates, return what follows the last assistant header;
      3. otherwise return the text unchanged (no exception, just no trimming).
    Known end-of-turn markers are trimmed from the end.
    """
    response = None
    if formatted_prompt:
        idx = full_text.find(formatted_prompt)
        if idx != -1:
            response = full_text[idx + len(formatted_prompt):]
    if response is None:
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        idx = full_text.rfind(marker)
        if idx == -1:
            return full_text
        response = full_text[idx + len(marker):]
    for stop in _STOP_STRINGS:
        response = response.split(stop)[0]
    return response.strip()


# format_chat_prompt now lives in prompting.py (imported above) so extraction and steering share it.


# -----------------------------------------------------------------------
# 4. Single-prompt steering entry point
# -----------------------------------------------------------------------

def steer_generation(
    model,
    tokenizer,
    directions,
    prompt,
    layers_to_control=None,
    control_coef=DEFAULT_CONTROL_COEF,
    max_new_tokens=200,
    do_sample=False,
    temperature=1.0,
    **gen_kwargs
):
    """
    Generate a response for a single prompt with anti-misogyny steering active.

    Directions point toward misogyny (label=1).  Passing control_coef < 0 inverts that direction, 
    nudging the model's internal representations toward non-misogynistic framing on every token step.

    Args:
        model             : Transformer model in eval mode.
        tokenizer         : Matching tokenizer.
        directions        : Per-layer concept vectors: {int -> tensor}. Keys are negative layer indices (-1, -2, ...).
        prompt            : Raw user message (chat template applied internally).
        layers_to_control : Layer indices to hook. Defaults to all layers in the directions dict, which is the full set from concept-vector training.
                            You can restrict this to a subset, e.g. only the middle layers, by passing a shorter list.
        control_coef      : Steering strength. Negative steers AWAY from misogyny. Recommended starting point: -0.2.
        max_new_tokens    : Maximum tokens to produce.
        do_sample         : False = greedy (deterministic), True = nucleus/temp sampling.
        temperature       : Sampling temperature (only used when do_sample=True).
        **gen_kwargs      : Any other model.generate() keyword arguments.

    Returns:
        steered_output: Decoded text string (prompt + steered generation).
    """
    # Warn loudly if someone passes a positive coefficient by accident -- that would steer TOWARD misogyny instead of away.
    if control_coef > 0:
        warnings.warn(
            f"control_coef={control_coef} is positive, which steers TOWARD misogyny. "
            "Pass a negative value (e.g. -0.2) to steer away from it.",
            UserWarning
        )

    # Default to all layers that have a trained direction
    if layers_to_control is None:
        layers_to_control = list(directions.keys())

    # Only hook layers for which we actually have a trained direction
    layers_to_control = [l for l in layers_to_control if l in directions]

    if not layers_to_control:
        raise ValueError(
            "No overlap between layers_to_control and the available directions. "
            "Make sure directions uses negative integer keys (e.g. -1, -2, ...)."
        )

    # Apply the instruction-tuned model's chat template to the raw prompt
    formatted_prompt = format_chat_prompt(tokenizer, prompt)

    # Register perturbation hooks on every selected block
    hooks = hook_model(model, directions, layers_to_control, control_coef)

    try:
        # Hooks fire automatically inside here on every forward pass step
        steered_output = generate_on_text(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **gen_kwargs
        )
    finally:
        # Remove hooks no matter what -- prevents them from leaking into later calls
        clear_hooks(hooks)

    return steered_output



# -----------------------------------------------------------------------
# 5. Coefficient sweep (paper evaluates several epsilons per concept)
# -----------------------------------------------------------------------

def sweep_control_coefficients(
    model,
    tokenizer,
    directions,
    prompt,
    coef_range=None,
    layers_to_control=None,
    max_new_tokens=200,
    do_sample=False,
):
    """
    Try multiple control coefficients for the same prompt and return all outputs.

    The paper sweeps several epsilon values per concept and keeps whichever successfully steers the model 
    (for anti-misogyny we check whether the output avoids misogynistic framing).

    Args:
        model             : Transformer model.
        tokenizer         : Matching tokenizer.
        directions        : Per-layer concept direction dict.
        prompt            : Raw user message string.
        coef_range        : List of epsilon values to try. Defaults to [-0.1, -0.2, -0.3, -0.4, -0.5], covering progressively stronger de-misogyny steering.
        layers_to_control : Layer indices to hook (defaults to all in directions).
        max_new_tokens    : Token budget per generation.
        do_sample         : Greedy or sampled generation.

    Returns:
        sweep_results: list of dicts, one per coefficient:
            {
              'control_coef'   : float,
              'steered_output' : str,
            }
    """
    # Negative coefficients only -- we are steering AWAY from misogyny
    if coef_range is None:
        # coef_range = [-0.1, -0.2, -0.3, -0.4, -0.5]
        coef_range = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -1.0]

    sweep_results = []

    for coef in coef_range:
        output = steer_generation(
            model=model,
            tokenizer=tokenizer,
            directions=directions,
            prompt=prompt,
            layers_to_control=layers_to_control,
            control_coef=coef,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        sweep_results.append({
            'control_coef': coef,
            'steered_output': output,
        })

    return sweep_results



# -----------------------------------------------------------------------
# 6. Batch steering over a test set
# -----------------------------------------------------------------------

def run_anti_misogyny_steering(
    model,
    tokenizer,
    directions,
    test_prompts,
    control_coef=DEFAULT_CONTROL_COEF,
    layers_to_control=None,
    max_new_tokens=200,
    do_sample=False,
):
    """
    Apply anti-misogyny steering across an entire list of test prompts.

    Args:
        model             : Transformer model.
        tokenizer         : Matching tokenizer.
        directions        : Per-layer concept vector dict from concept-vector training.
        test_prompts      : List of raw user message strings.
        control_coef      : Negative steering coefficient.
        layers_to_control : Layer indices to hook (defaults to all available).
        max_new_tokens    : Token budget per generation.
        do_sample         : Greedy or sampled generation.

    Returns:
        results: list of dicts, one per prompt:
        {
          'prompt'         : str,   -- original user message
          'base_output'    : str,   -- unsteered generation (no hooks active)
          'steered_output' : str,   -- generation with anti-misogyny hooks active
          'epsilon'        : float, -- control coefficient used (negative = away from misogyny)
        }
    """
    results = []

    for prompt in tqdm(test_prompts, desc="Running anti-misogyny steering"):

        
        formatted_prompt = format_chat_prompt(tokenizer, prompt)
        # Base generation: no hooks, no perturbation. Same pattern as compare_base_vs_steered.
        base_output = generate_on_text(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        # Steered generation: anti-misogyny hooks are active here.
        steered_output = steer_generation(
            model=model,
            tokenizer=tokenizer,
            directions=directions,
            prompt=prompt,
            layers_to_control=layers_to_control,
            control_coef=control_coef,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        results.append({
                    'prompt':          prompt,
                    'base_output':     base_output,
                    'steered_output':  steered_output,
                    'epsilon':         control_coef,
                })

    return results



# -----------------------------------------------------------------------
# 7. Side-by-side comparison: base model vs. steered model
# -----------------------------------------------------------------------

def compare_base_vs_steered(
    model,
    tokenizer,
    directions,
    prompts,
    control_coef=DEFAULT_CONTROL_COEF,
    layers_to_control=None,
    max_new_tokens=200,
    do_sample=False,
):
    """
    Generate both an unsteered (base) and a steered response for each prompt.
    Useful for qualitative inspection of whether the steering is working as intended.

    The base output uses the same model without any hooks -- no coefficient, no direction manipulation.
    The steered output uses control_coef < 0 to subtract the misogyny direction.

    Args:
        model             : Transformer model.
        tokenizer         : Matching tokenizer.
        directions        : Per-layer concept vector dict.
        prompts           : List of raw user message strings.
        control_coef      : Negative epsilon for the steered pass.
        layers_to_control : Layer indices to hook for the steered pass.
        max_new_tokens    : Token budget per generation.
        do_sample         : Greedy or sampled generation.

    Returns:
        comparisons: list of dicts:
            {
              'prompt'         : str,
              'base_output'    : str,  -- no steering
              'steered_output' : str,  -- with anti-misogyny steering
            }
    """
    comparisons = []

    for prompt in tqdm(prompts, desc="Comparing base vs. steered outputs"):
        formatted_prompt = format_chat_prompt(tokenizer, prompt)

        # ---- Base generation (no hooks, no perturbation) ----
        base_output = generate_on_text(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        # ---- Steered generation (anti-misogyny hooks active) ----
        steered_output = steer_generation(
            model=model,
            tokenizer=tokenizer,
            directions=directions,
            prompt=prompt,
            layers_to_control=layers_to_control,
            control_coef=control_coef,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        comparisons.append({
            'prompt': prompt,
            'base_output': base_output,
            'steered_output': steered_output,
        })

    return comparisons



# -----------------------------------------------------------------------
# 8. Persistence helpers
# -----------------------------------------------------------------------

def load_directions(model_id: str, vector_name: str = None) -> dict:
    """
    Load the per-layer concept directions of one named vector (see concept_vector_training.py).

    Args:
        model_id    : HuggingFace model identifier (must match what extraction / training used).
        vector_name : "averaged__misogyny" (default), "pooled__misogyny", "single__misogyny",
                      or "subtype__<slug>". Defaults to $VECTOR_NAME, then averaged__misogyny.
    Returns:
        directions: dict {int (negative layer index) -> Tensor(1, hidden_dim)}
    """
    vector_name = vector_name or DEFAULT_VECTOR_NAME
    directions, signs, method = load_concept_vectors(model_id, vector_name, CONCEPT_VECTORS_DIR)
    print(f"Loaded {len(directions)} layer directions  (model: {model_id} | vector: {vector_name} | method: {method})")
    return directions


def save_steering_results(results, output_path):
    """
    Persist steering results to disk as a pickle file.

    Args:
        results     : List of result dicts (from run_anti_misogyny_steering or compare_base_vs_steered).
        output_path : Destination file path, e.g. 'results/steering_results.pkl'.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Steering results saved to: {output_path}")




# -----------------------------------------------------------------------
# 9. Main entry point
# -----------------------------------------------------------------------

def load_model_for_steering(model_id: str, dtype=None, trust_remote_code: bool = False):
    """Tokenizer + model for generation (same loader as extraction; dtype default: float16 on GPU, float32 on CPU)."""
    return load_causal_lm(model_id, dtype=dtype, trust_remote_code=trust_remote_code)


def _eval_prompts(n=None, language: str = "en"):
    """The evaluation questions (evaluation_prompts/<language>/misogyny_eval_prompts.txt, one per line)."""
    path = PROJECT_ROOT / "evaluation_prompts" / language / "misogyny_eval_prompts.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: add translated evaluation prompts for language {language!r}")
    prompts = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return prompts[:n] if n else prompts


if __name__ == "__main__":
    import argparse
    import experiment
    load_dotenv()

    parser = argparse.ArgumentParser(description="Qualitative anti-misogyny steering demo (see steering_eval.py for the scored version).")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--vector", default=DEFAULT_VECTOR_NAME)
    parser.add_argument("--coef", type=float, default=DEFAULT_CONTROL_COEF,
                        help="negative = steer AWAY from misogyny (default -0.2)")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="negative layer indices to hook (default: all layers of the vector)")
    parser.add_argument("--n-qualitative", type=int, default=5, help="prompts shown side by side (base vs steered)")
    parser.add_argument("--language", default="en", help="language of the evaluation prompts (evaluation_prompts/<language>/)")
    add_model_args(parser)
    experiment.add_experiment_arg(parser)
    parser.add_argument("--max-new-tokens", type=int, default=400,
                        help="200 was too tight for this model's verbose, list-heavy style")
    args = parser.parse_args()
    MODEL_ID = args.model_id
    if not MODEL_ID:
        raise ValueError("MODEL_ID is not set in the .env file (or pass --model-id).")

    experiment.apply_experiment_arg(args)
    RESULTS_PATH = str(exp_dir(PROJECT_ROOT / "results", MODEL_ID) / f"anti_misogyny_steering__{args.vector}.pkl")
    CONTROL_COEF = args.coef
    MAX_NEW_TOKENS = args.max_new_tokens

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_for_steering(MODEL_ID, dtype=args.dtype, trust_remote_code=args.trust_remote_code)

    directions = load_directions(MODEL_ID, args.vector)
    available_layers = sorted(directions.keys())
    print(f"Concept directions available for layers: {available_layers}")

    all_prompts = _eval_prompts(language=args.language)
    sample_prompts = all_prompts[:args.n_qualitative]

    print("\nRunning base vs. steered comparison...")
    comparisons = compare_base_vs_steered(
        model=model, tokenizer=tokenizer, directions=directions, prompts=sample_prompts,
        control_coef=CONTROL_COEF, layers_to_control=args.layers, max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # greedy decoding for reproducibility
    )
    for entry in comparisons:
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"PROMPT:\n  {entry['prompt']}")
        print(f"\nBASE OUTPUT (no steering):\n{extract_response_only(entry['base_output'], format_chat_prompt(tokenizer, entry['prompt']))}")
        print(f"\nSTEERED OUTPUT (anti-misogyny, epsilon={CONTROL_COEF}):\n{extract_response_only(entry['steered_output'], format_chat_prompt(tokenizer, entry['prompt']))}")

    print("\nSweeping control coefficients on the first prompt...")
    sweep = sweep_control_coefficients(
        model=model, tokenizer=tokenizer, directions=directions, prompt=sample_prompts[0],
        coef_range=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -1.0],
        layers_to_control=args.layers, max_new_tokens=MAX_NEW_TOKENS,
    )
    for entry in sweep:
        response_only = extract_response_only(entry['steered_output'], format_chat_prompt(tokenizer, sample_prompts[0]))
        print(f"\nepsilon={entry['control_coef']}:\n{response_only[:300]}...")

    print("\nRunning anti-misogyny steering on all evaluation prompts...")
    steering_results = run_anti_misogyny_steering(
        model=model, tokenizer=tokenizer, directions=directions, test_prompts=all_prompts,
        control_coef=CONTROL_COEF, layers_to_control=args.layers, max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    save_steering_results(steering_results, RESULTS_PATH)
    print("\nDone.")
