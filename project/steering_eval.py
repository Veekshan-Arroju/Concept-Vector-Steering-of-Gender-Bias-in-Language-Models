"""
Quantitative steering evaluation (the analogue of the authors' generate-then-judge benchmark).

For every evaluation prompt and every coefficient:
    1. generate greedily with the vector added to the chosen layers (coefficient 0 = unsteered baseline),
    2. record a keyword refusal heuristic, and (if a judge is configured) the judge's
       misogynistic / coherent / refusal_or_hedge labels,
and then aggregate per coefficient.

Sign convention (same as anti_misogyny_steering.py): vectors point TOWARD misogyny, so
    coef < 0  steers away from misogyny   (anti-misogyny steering)
    coef > 0  steers toward misogyny      (validation / control condition)
Llama-3.1-Instruct is already trained to avoid misogynistic text, so the coef > 0 side is the more
informative test of whether the vector does anything; the coef < 0 side mostly checks that it does
not break coherence or change refusal behaviour.

Usage:
    python steering_eval.py --vector averaged__misogyny --coefs -0.4 -0.2 0 0.2 0.4 --judge none
    python steering_eval.py --vector averaged__misogyny --coefs 0 0.25 0.5 0.75 1.0 1.5 --judge openai   (toward misogyny)
    --language hi   uses evaluation_prompts/hi/ and tells the judge the text is Hindi (any language code with a prompt file)
"""

import argparse
import csv
import os
import pickle
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import torch
from tqdm import tqdm

import anti_misogyny_steering as ams    # hook_model / clear_hooks / loader are looked up at call time
import experiment
from concept_vector_training import DEFAULT_VECTOR_NAME
from experiment import PROJECT_ROOT, exp_dir
from coef_calibration import regime_warnings
from judge import JUDGE_KEYS, degeneration_heuristic, get_judge, judge_one, load_rubric, refusal_heuristic
from model_utils import add_model_args
from prompting import format_chat_prompt

RESULTS_DIR = PROJECT_ROOT / "results"
# Paper regime for Llama-3.1-8B: unit vectors on ALL blocks, |epsilon| 0.55-0.8 (collapse above). Run calibrate_steering.py to find the
# window of YOUR vector; this default only brackets it.
DEFAULT_COEFS = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]


def prompts_path(language: str = "en") -> Path:
    return PROJECT_ROOT / "evaluation_prompts" / language / "misogyny_eval_prompts.txt"


def load_eval_prompts(path: Path = None, n: int = None, language: str = "en") -> list:
    """The evaluation questions, one per line (evaluation_prompts/<language>/misogyny_eval_prompts.txt)."""
    path = Path(path) if path is not None else prompts_path(language)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: add translated evaluation prompts for language {language!r}")
    prompts = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return prompts[:n] if n else prompts


@torch.no_grad()
def generate_response(model, tokenizer, user_prompt: str, directions: dict = None, coef: float = 0.0,
                      layers: list = None, max_new_tokens: int = 200, positions: str = "all") -> str:
    """Greedy generation; returns ONLY the newly generated text (no echoed prompt / template)."""
    formatted = format_chat_prompt(tokenizer, user_prompt)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)
    hooks = {}
    if directions is not None and coef != 0.0:
        layers = [l for l in (layers or list(directions.keys())) if l in directions]
        hooks = ams.hook_model(model, directions, layers, coef, positions=positions)
    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, max_length=None,
                             pad_token_id=tokenizer.eos_token_id)
    finally:
        ams.clear_hooks(hooks)
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def random_unit_directions(directions: dict, seed: int) -> dict:
    """
    Same layers and shapes as `directions`, but every layer's vector is a random unit vector (seeded).
    The control for "does the DIRECTION matter, or only the size of the push?": run the same coefficients with it.
    """
    out = {}
    for layer, v in directions.items():
        g = torch.Generator().manual_seed(int(seed) * 1000 + abs(int(layer)))
        r = torch.randn(v.shape[-1], generator=g)
        out[layer] = (r / r.norm()).reshape(1, -1).to(v.dtype)
    return out


def summarize(rows: list) -> list:
    """Per-coefficient means. Labels that a judge failed to produce (None) are excluded from that mean."""
    summary = []
    for coef in sorted({r["coef"] for r in rows}):
        sub = [r for r in rows if r["coef"] == coef]
        entry = {"coef": coef, "n": len(sub),
                 "refusal_heuristic": sum(r["refusal_heuristic"] for r in sub) / len(sub),
                 "degenerate_heuristic": sum(r.get("degenerate_heuristic", 0) for r in sub) / len(sub)}
        for k in JUDGE_KEYS:
            vals = [r[k] for r in sub if r.get(k) is not None]
            entry[k] = (sum(vals) / len(vals)) if vals else None
        summary.append(entry)
    return summary


def print_summary(summary: list) -> None:
    print("\n" + "=" * 78)
    print(f"  {'coef':>6}  {'n':>4}  {'misogynistic':>12}  {'coherent':>9}  {'refusal/hedge':>13}  {'refusal kw':>10}  {'degenerate':>10}")
    print("=" * 78)
    fmt = lambda v: "     n/a" if v is None else f"{v:8.2f}"
    for e in summary:
        print(f"  {e['coef']:+6.2f}  {e['n']:4d}  {fmt(e['misogynistic']):>12}  {fmt(e['coherent']):>9}  "
              f"{fmt(e['refusal_or_hedge']):>13}  {e['refusal_heuristic']:10.2f}  {e['degenerate_heuristic']:10.2f}")
    print("=" * 78 + "\n")


def run_steering_eval(model, tokenizer, directions, prompts, coefs, layers=None, max_new_tokens=200,
                      judge_fn=None, rubric=None, progress=True, language: str = "en") -> dict:
    """Generate + (optionally) judge every (prompt, coef); return {'rows': [...], 'summary': [...]}."""
    rows = []
    iterator = tqdm(prompts, desc="Steering eval") if progress else prompts
    for prompt in iterator:
        for coef in coefs:
            response = generate_response(model, tokenizer, prompt, directions, coef, layers, max_new_tokens)
            row = {"prompt": prompt, "coef": float(coef), "response": response,
                   "refusal_heuristic": refusal_heuristic(response), "degenerate_heuristic": degeneration_heuristic(response)}
            if judge_fn is not None:
                j = judge_one(prompt, response, judge_fn, rubric, language=language)
                row.update({k: j[k] for k in JUDGE_KEYS})
                row["judge_explanation"] = j["explanation"]
            else:
                row.update({k: None for k in JUDGE_KEYS})
                row["judge_explanation"] = ""
            rows.append(row)
    return {"rows": rows, "summary": summarize(rows)}


def save_eval(result: dict, vector_name: str, model_id: str, results_dir: Path = RESULTS_DIR) -> Path:
    out_dir = exp_dir(results_dir, model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"steering_eval__{vector_name}"
    with open(f"{stem}.pkl", "wb") as f:
        pickle.dump(result, f)
    fields = ["prompt", "coef", "response", "refusal_heuristic", "degenerate_heuristic", *JUDGE_KEYS, "judge_explanation"]
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(result["rows"])
    print(f"[steering_eval] saved -> {stem}.csv / .pkl")
    return stem


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation with an optional LLM judge.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--vector", default=DEFAULT_VECTOR_NAME)
    parser.add_argument("--coefs", nargs="*", type=float, default=DEFAULT_COEFS)
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="negative layer indices to hook (default: every layer of the vector)")
    parser.add_argument("--n-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--judge", default="none", choices=["none", "openai", "anthropic"])
    parser.add_argument("--language", default="en", help="language of the evaluation prompts (evaluation_prompts/<language>/)")
    parser.add_argument("--random-direction", type=int, default=None, metavar="SEED",
                        help="CONTROL: steer with a random unit vector (seeded) instead of the learned one, same layers and coefficients")
    parser.add_argument("--tag", default="", help="added to the output file names, so runs do not overwrite each other")
    add_model_args(parser)
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    if not args.model_id:
        raise ValueError("pass --model-id or set MODEL_ID in .env")

    model, tokenizer = ams.load_model_for_steering(args.model_id, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    directions = ams.load_directions(args.model_id, args.vector)
    if args.random_direction is not None:
        directions = random_unit_directions(directions, args.random_direction)
        print(f"[steering_eval] CONTROL: random unit directions (seed {args.random_direction}) replace the learned vector at every layer")
    hooked = [l for l in (args.layers or list(directions.keys())) if l in directions]
    for w in regime_warnings(len(hooked), args.coefs, len(ams.get_decoder_layers(model)), [float(directions[l].reshape(-1).norm()) for l in hooked]):
        print(f"[steering_eval] WARNING: {w}")
    prompts = load_eval_prompts(n=args.n_prompts, language=args.language)
    judge_fn = get_judge(args.judge)
    rubric = load_rubric() if judge_fn else None

    result = run_steering_eval(model, tokenizer, directions, prompts, args.coefs, layers=args.layers,
                               max_new_tokens=args.max_new_tokens, judge_fn=judge_fn, rubric=rubric, language=args.language)
    print_summary(result["summary"])
    name = args.vector + (f"__random{args.random_direction}" if args.random_direction is not None else "")
    if args.tag:
        name += "__" + re.sub(r"[^A-Za-z0-9_.+-]", "_", args.tag)
    save_eval(result, name, args.model_id)


if __name__ == "__main__":
    main()
