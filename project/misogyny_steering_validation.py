"""
Positive-direction steering validation for the anti-misogyny concept vectors.

Companion script to anti_misogyny_steering.py, using the SAME trained per-layer
directions but with epsilon > 0 instead of epsilon < 0 -- i.e. pushing the
model's activations TOWARD the misogyny direction instead of away from it.

Why this matters methodologically:
    Llama-3.1-8B-Instruct is already heavily RLHF'd against producing
    misogynistic content, so epsilon < 0 steering (anti_misogyny_steering.py)
    often looks similar to the unsteered baseline -- the model was already
    avoiding that content, leaving little room to demonstrate improvement.
    That makes it hard to tell, from the negative-direction results alone,
    whether the learned direction v_l is a genuine "misogyny" direction or
    just noise that happens to correlate with the training labels.

    Steering in the OPPOSITE direction (epsilon > 0) is the standard control
    condition for this: if v_l is real, pushing +epsilon should visibly shift
    generations toward more stereotyped/dismissive framing, even if safety
    training keeps it from becoming a fully unguarded response. A clear,
    roughly monotonic effect in BOTH directions is much stronger evidence
    that RFM found the intended concept than either direction alone. This
    mirrors the paper's own use of positive AND negative epsilon (Fig. 2) to
    validate steering directions, e.g. its anti-deception vs. pro-deception
    comparison.

Reuses hook_model / generate_on_text / format_chat_prompt / steer_generation /
compare_base_vs_steered / sweep_control_coefficients / load_directions /
save_steering_results directly from anti_misogyny_steering.py -- nothing about
the underlying mechanism changes, only the sign/magnitude of control_coef and
how results are labeled and saved.

Content note:
    Because this deliberately steers TOWARD the misogyny direction, generations
    at higher |epsilon| may include stereotyped or demeaning framing. That is
    the point of this script -- it exists to validate the direction, not to
    produce usable text. Treat its output as evidence about direction validity,
    not as generation output, and keep results out of anything user-facing.
"""

import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

from anti_misogyny_steering import (
    _eval_prompts,
    load_directions,
    load_model_for_steering,
    extract_response_only,
    compare_base_vs_steered,
    sweep_control_coefficients,
    run_anti_misogyny_steering,
    save_steering_results,
)
from concept_vector_training import DEFAULT_VECTOR_NAME
from experiment import PROJECT_ROOT, exp_dir
from model_utils import add_model_args
from prompting import format_chat_prompt

# Positive coefficient -- pushes TOWARD misogyny, the mirror image of
# anti_misogyny_steering.py's default of -0.2. Kept modest by default since
# the point is to observe a directional shift, not to maximize it.
CONTROL_COEF = 0.2
MAX_NEW_TOKENS = 400


if __name__ == "__main__":
    import argparse
    import experiment

    load_dotenv()
    parser = argparse.ArgumentParser(description="Positive-direction (toward misogyny) validation of a named vector.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--vector", default=DEFAULT_VECTOR_NAME)
    parser.add_argument("--coef", type=float, default=CONTROL_COEF, help="positive = toward misogyny (default 0.2)")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="negative layer indices to hook (default: all layers of the vector)")
    parser.add_argument("--n-qualitative", type=int, default=5)
    parser.add_argument("--language", default="en", help="language of the evaluation prompts (evaluation_prompts/<language>/)")
    add_model_args(parser)
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    MODEL_ID = args.model_id
    if not MODEL_ID:
        raise ValueError("MODEL_ID is not set in the .env file (or pass --model-id).")

    RESULTS_PATH = str(exp_dir(PROJECT_ROOT / "results", MODEL_ID) / f"misogyny_direction_validation__{args.vector}.pkl")
    CONTROL_COEF = args.coef

    print(
        "Running POSITIVE-direction (toward misogyny) steering -- "
        "this is a validation / control condition for the learned direction, "
        "not intended as a generation tool.\n"
    )

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_for_steering(MODEL_ID, dtype=args.dtype, trust_remote_code=args.trust_remote_code)

    # ---- Load the SAME vector used by anti_misogyny_steering.py ----
    directions = load_directions(MODEL_ID, args.vector)
    available_layers = sorted(directions.keys())
    print(f"Concept directions available for layers: {available_layers}")

    # Same evaluation prompts as anti_misogyny_steering.py, for direct side-by-side comparison
    all_prompts = _eval_prompts(language=args.language)
    sample_prompts = all_prompts[:args.n_qualitative]

    # ---- Side-by-side comparison: base vs. positive-steered ----
    print("\nRunning base vs. positive-steered comparison...")
    comparisons = compare_base_vs_steered(
        model=model,
        tokenizer=tokenizer,
        directions=directions,
        prompts=sample_prompts,
        control_coef=CONTROL_COEF,
        layers_to_control=args.layers,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,   # greedy decoding for reproducibility
    )

    for entry in comparisons:
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"PROMPT:\n  {entry['prompt']}")
        print(f"\nBASE OUTPUT (no steering):\n{extract_response_only(entry['base_output'], format_chat_prompt(tokenizer, entry['prompt']))}")
        print(f"\nSTEERED OUTPUT (TOWARD misogyny, epsilon=+{CONTROL_COEF}):\n{extract_response_only(entry['steered_output'], format_chat_prompt(tokenizer, entry['prompt']))}")

    # ---- Coefficient sweep: how strong does epsilon need to be before the shift is visible? ----
    # Past ~1.0 the output tends toward repetition / word salad rather than a cleanly stronger effect:
    # read the actual text at each epsilon rather than assuming higher = more misogynistic.
    print("\nSweeping positive control coefficients on the first prompt...")
    sweep = sweep_control_coefficients(
        model=model,
        tokenizer=tokenizer,
        directions=directions,
        prompt=sample_prompts[0],
        coef_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5],
        layers_to_control=args.layers,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    for entry in sweep:
        response_only = extract_response_only(entry['steered_output'], format_chat_prompt(tokenizer, sample_prompts[0]))
        print(f"\nepsilon=+{entry['control_coef']}:\n{response_only}")

    # ---- Full run on all evaluation prompts, saved separately from the anti-misogyny results ----
    print("\nRunning positive-direction (toward misogyny) steering on all evaluation prompts...")
    steering_results = run_anti_misogyny_steering(
        model=model,
        tokenizer=tokenizer,
        directions=directions,
        test_prompts=all_prompts,
        control_coef=CONTROL_COEF,
        layers_to_control=args.layers,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    save_steering_results(steering_results, RESULTS_PATH)
    print("\nDone. For scored results (judge labels, coherence, refusal rate) use steering_eval.py.")
