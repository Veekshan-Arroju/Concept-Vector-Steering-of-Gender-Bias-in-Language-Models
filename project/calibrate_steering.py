"""
Find the steering window of ONE vector before spending judge calls: where does the model stop changing, and where does it collapse?

Paper protocol (arXiv:2502.03708): steer ALL blocks with the unit concept vector and a small epsilon; the useful range is a narrow
band just under the point where generations turn into repetition (Llama-3.1-8B: epsilon 0.55-0.8; beyond it "nonsensical,
repetitive outputs"). The band differs per vector, so this script measures it:

    1. for each sign (+ = toward misogyny, - = away) sweep |epsilon| = 0.1, 0.2, ... on a few evaluation prompts, greedy decoding,
       hooking every layer of the vector (override with --layers);
    2. a generation is "collapsed" when judge.degeneration_heuristic fires (no API, no judge);
    3. threshold = smallest |epsilon| from which >= 1/3 of the prompts are collapsed at every larger epsilon tested;
    4. working window = 60-90 % of the threshold (4 values); both are printed and saved;
    5. optionally (--random-seeds) the same sweep with random unit directions: the learned vector should NOT behave like noise;
    6. optionally (--evaluate) the full judged evaluation at the working window (steering_eval.run_steering_eval), learned
       vector and random control, saved next to the usual steering_eval outputs.

Usage:
    python calibrate_steering.py --vector pooled__misogyny --experiment stance --random-seeds 0 1
    python calibrate_steering.py --vector pooled__misogyny --experiment stance --evaluate --judge openai

Before this, run the judge-free direction check (no calibration needed):
    python stance_logit_test.py --vector pooled__misogyny --experiment stance --relative --coefs -0.2 -0.1 0.1 0.2 --with-reversed
    python directionality_check.py results/<slug>/stance/stance_logit__pooled__misogyny__<tag>.pkl
"""

import argparse
import csv
import json
import os
import re
import warnings
warnings.filterwarnings("ignore")

import anti_misogyny_steering as ams
import experiment
from coef_calibration import (DEFAULT_MAX_DEGENERATE_FRAC, collapse_threshold, default_grid, read_prompt_file, regime_warnings, working_window)
from concept_vector_training import DEFAULT_VECTOR_NAME
from experiment import PROJECT_ROOT, exp_dir
from judge import degeneration_heuristic, get_judge, load_rubric, refusal_heuristic
from model_utils import add_model_args
from steering_eval import (generate_response, load_eval_prompts, print_summary, random_unit_directions, run_steering_eval, save_eval)


def sweep(model, tokenizer, directions, prompts, grid, sign, layers, max_new_tokens, max_frac, patience=2, label="learned", positions="all"):
    """Ascending sweep for one sign. Stops after `patience` consecutive collapsed grid points. Returns (rows, {abs_coef: [flags]})."""
    rows, flags, collapsed_run = [], {}, 0
    for c in grid:
        coef = sign * c
        fl = []
        for p in prompts:
            text = generate_response(model, tokenizer, p, directions, coef, layers, max_new_tokens, positions=positions)
            d = degeneration_heuristic(text)
            fl.append(d)
            rows.append({"direction": label, "coef": coef, "prompt": p, "response": text, "degenerate": d,
                         "refusal_heuristic": refusal_heuristic(text)})
        flags[c] = fl
        frac = sum(fl) / len(fl)
        print(f"  [{label:>9}] coef {coef:+.2f}: degenerate {sum(fl)}/{len(fl)}", flush=True)
        collapsed_run = collapsed_run + 1 if frac >= max_frac else 0
        if collapsed_run >= patience:
            break
    return rows, flags


def main():
    ap = argparse.ArgumentParser(description="Calibrate the steering coefficient window of a vector (collapse point, working window).")
    ap.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    ap.add_argument("--vector", default=DEFAULT_VECTOR_NAME)
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="negative layer indices to hook (default: every layer of the vector = the paper's setting)")
    ap.add_argument("--signs", default="both", choices=["both", "positive", "negative"])
    ap.add_argument("--positions", default="all", choices=["all", "generated"],
                    help="all (default, the paper's setting): steer every position; generated: leave the prompt pass alone and steer only while the answer is written")
    ap.add_argument("--coefs", nargs="*", type=float, default=None, help="explicit |epsilon| grid (overrides --max-coef/--step), e.g. --coefs 0.4 0.6 0.8")
    ap.add_argument("--max-coef", type=float, default=1.2)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--n-prompts", type=int, default=4, help="evaluation prompts used for the sweep (N prompts of the file, starting at --prompt-offset)")
    ap.add_argument("--prompts-file", default=None, help="read the prompts from this file (one per line, # comments allowed) instead of the default evaluation prompts; "
                                                         "relative paths are taken from the project folder")
    ap.add_argument("--prompt-offset", type=int, default=0, help="skip the first K evaluation prompts (default 0 = the same first prompts as every earlier calibration)")
    ap.add_argument("--tag", default="", help="suffix for the output files (calibration__<vector>__<tag>.json/.csv) so a new sweep never overwrites an earlier one")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--max-degenerate-frac", type=float, default=DEFAULT_MAX_DEGENERATE_FRAC)
    ap.add_argument("--skip-learned", action="store_true", help="sweep only the random directions (needs --random-seeds): the random control does not depend on the vector")
    ap.add_argument("--random-seeds", nargs="*", type=int, default=[], help="also sweep random unit directions with these seeds (control)")
    ap.add_argument("--evaluate", action="store_true", help="after calibrating, run the judged evaluation at the working window")
    ap.add_argument("--judge", default="none", choices=["none", "openai", "anthropic"])
    ap.add_argument("--language", default="en")
    add_model_args(ap)
    experiment.add_experiment_arg(ap)
    args = ap.parse_args()
    experiment.apply_experiment_arg(args)
    if not args.model_id:
        raise ValueError("pass --model-id or set MODEL_ID in .env")

    model, tokenizer = ams.load_model_for_steering(args.model_id, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    directions = ams.load_directions(args.model_id, args.vector)
    layers = [l for l in (args.layers or list(directions.keys())) if l in directions]
    n_total = len(ams.get_decoder_layers(model))
    norms = [float(directions[l].reshape(-1).norm()) for l in layers]
    for w in regime_warnings(len(layers), [args.max_coef], n_total, norms):
        print(f"[calibrate] WARNING: {w}")
    print(f"[calibrate] vector {args.vector}: hooking {len(layers)} of {n_total - 1} blocks; unit-norm check {min(norms):.3f}..{max(norms):.3f}")

    if args.positions != "all" and not args.tag:
        args.tag = args.positions                      # never overwrite the files of an all-positions sweep
    pool = read_prompt_file(PROJECT_ROOT / args.prompts_file) if args.prompts_file else load_eval_prompts(language=args.language)
    prompts = pool[args.prompt_offset:args.prompt_offset + args.n_prompts]
    if not prompts:
        raise ValueError(f"no evaluation prompts at offset {args.prompt_offset} (the file has {len(pool)})")
    print(f"[calibrate] evaluation prompts {args.prompt_offset}..{args.prompt_offset + len(prompts) - 1}")
    grid = sorted({round(abs(c), 6) for c in args.coefs if c}) if args.coefs else default_grid(args.max_coef, args.step)
    signs = {"both": (+1, -1), "positive": (+1,), "negative": (-1,)}[args.signs]
    all_rows, report = [], {"vector": args.vector, "experiment": experiment.get_experiment(), "layers": layers, "grid": grid,
                            "max_degenerate_frac": args.max_degenerate_frac, "prompt_offset": args.prompt_offset, "positions": args.positions, "prompts_file": args.prompts_file, "n_prompts": len(prompts),
                            "thresholds": {}, "windows": {}}

    if args.skip_learned and not args.random_seeds:
        raise ValueError("--skip-learned needs --random-seeds")
    directions_sets = ([] if args.skip_learned else [("learned", directions)]) + [(f"random{s}", random_unit_directions(directions, s)) for s in args.random_seeds]
    for label, dirs in directions_sets:
        for sg in signs:
            tag = f"{label}{'+' if sg > 0 else '-'}"
            print(f"\n[calibrate] {tag}: sweeping |epsilon| {grid[0]}..{grid[-1]}")
            rows, flags = sweep(model, tokenizer, dirs, prompts, grid, sg, layers, args.max_new_tokens, args.max_degenerate_frac, label=label, positions=args.positions)
            all_rows += rows
            thr = collapse_threshold(flags, args.max_degenerate_frac)
            report["thresholds"][tag] = thr
            report["windows"][tag] = working_window(thr)

    print("\n" + "=" * 72)
    print(f"  collapse threshold (|epsilon|) and working window, vector {args.vector}")
    print("=" * 72)
    for tag, thr in report["thresholds"].items():
        win = report["windows"][tag]
        print(f"  {tag:<12} threshold {('none up to ' + str(grid[-1])) if thr is None else f'{thr:.2f}':<18} working window {win if win else '-'}")
    print("=" * 72)
    print("  Paper (Llama-3.1-8B, all blocks, unit vectors): works at 0.55-0.80, collapses above. A learned vector whose threshold is much\n"
          "  LOWER than a random direction's is pushing the model off-distribution along a real axis; one that collapses at the same\n"
          "  size as noise is not special. Judge the window with --evaluate, never by eye on one prompt.\n")

    out_dir = exp_dir(PROJECT_ROOT / "results", args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / (f"calibration__{args.vector}" + (f"__{args.tag}" if args.tag else ""))
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "coef", "prompt", "response", "degenerate", "refusal_heuristic"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"[calibrate] saved -> {stem}.json / .csv")

    if args.evaluate:
        pos, neg = report["windows"].get("learned+", []), report["windows"].get("learned-", [])
        coefs = sorted({-c for c in neg} | {0.0} | set(pos))
        if len(coefs) <= 1:
            print("[calibrate] no working window found: nothing to evaluate. Lower --step or raise --max-coef.")
            return
        judge_fn = get_judge(args.judge)
        rubric = load_rubric() if judge_fn else None
        full_prompts = load_eval_prompts(language=args.language)
        for label, dirs in [("learned", directions)] + ([(f"random{args.random_seeds[0]}", random_unit_directions(directions, args.random_seeds[0]))]
                                                        if args.random_seeds else []):
            print(f"\n[calibrate] judged evaluation, {label}, coefficients {coefs}")
            res = run_steering_eval(model, tokenizer, dirs, full_prompts, coefs, layers=layers, max_new_tokens=200,
                                    judge_fn=judge_fn, rubric=rubric, language=args.language)
            print_summary(res["summary"])
            name = args.vector + ("" if label == "learned" else f"__{label}") + "__" + re.sub(r"[^A-Za-z0-9_.+-]", "_", "calibrated")
            save_eval(res, name, args.model_id)


if __name__ == "__main__":
    main()
