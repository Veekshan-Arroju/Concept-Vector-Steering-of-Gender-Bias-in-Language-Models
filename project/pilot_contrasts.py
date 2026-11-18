"""
Pilot: which TRAINING CONTRAST gives a direction that is specific to misogyny?   (no new extraction, no GPU, seconds)

Why
---
Trained on "instruction present vs absent" (the current datasets), the vectors score every instruction above its neutral
twin, and score non-gender hateful prompts as high as, or higher than, misogyny prompts (control_eval.py: AUC mis/hat
about 0.4-0.7). So they track "a hateful/ideological stance was requested" more than misogyny in particular.

What it does
------------
Each FRESH statement exists in four versions with the same sentence: neutral / misogyny / benign-concept / hateful-concept.
Using only those saved activations, it fits a mean-difference direction under four contrasts and scores each one on
statements it was NOT fitted on (5-fold cross-validation BY STATEMENT, so every statement is scored exactly once):

    vs_neutral : misogyny  -  neutral            (the contrast of the current datasets)
    vs_benign  : misogyny  -  benign concept     (the instruction itself cancels)
    vs_hateful : misogyny  -  other hateful concepts (the shared "hateful stance" cancels)
    vs_both    : misogyny  -  mean(benign, hateful)
    vs_mixed   : misogyny  -  mean(neutral, benign, hateful)   (keeps an ordinary no-instruction prompt as a negative too)
    (matched experiment only: vs_egalitarian, vs_all_controls, vs_mixed_all)

For each contrast and layer it reports the fraction of pairs where the misogyny prompt outscores the neutral / benign /
hateful prompt of the same statement (chance 0.50), on the SEEN phrases and on the four UNSEEN (held-out) phrases.
A specific direction scores high against all three on the unseen phrases.

Read with care
--------------
* The controls differ from misogyny in TOPIC (women) as well as in stance. A direction that passes may encode "about women /
  sexism" rather than "misogynistic stance". A topic-matched control (for example "gender equality", "respect for women")
  is not among the arms yet.
* 218 statements: differences of a few points are noise.
* Mean-difference only; it says which contrast is promising, not what RFM will do.

Usage:  python pilot_contrasts.py [--layers -4 -8 ...] [--folds 5] [--seed 0] [--experiment tag]
"""

import argparse
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from dotenv import load_dotenv

import activation_store as store
import dataset_io as dio
import experiment
from experiment import PROJECT_ROOT, exp_dir

load_dotenv()
PILOT_DIR = PROJECT_ROOT / "pilot_results"
DEFAULT_LAYERS = [-4, -8, -12, -16, -20, -24, -28]
REFS = ("neutral", "benign", "hateful")                       # base reference arms (the matched experiment adds "egalitarian")
CONTRASTS = {
    "vs_neutral": ("neutral",),
    "vs_benign": ("benign",),
    "vs_hateful": ("hateful",),
    "vs_both": ("benign", "hateful"),
    "vs_mixed": ("neutral", "benign", "hateful"),               # keeps the neutral baseline AND the instruction controls
}
EXTRA_CONTRASTS = {                                           # only when the experiment has an egalitarian arm
    "vs_egalitarian": ("egalitarian",),
    "vs_all_controls": ("benign", "hateful", "egalitarian"),
    "vs_mixed_all": ("neutral", "benign", "hateful", "egalitarian"),
}


def arms_and_contrasts() -> tuple:
    """(reference arms, contrasts) for the current experiment."""
    refs = tuple(a for a in dio.arm_names() if a != "subtype")
    contrasts = dict(CONTRASTS)
    if "egalitarian" in refs:
        contrasts.update(EXTRA_CONTRASTS)
    return refs, contrasts


# -----------------------------------------------------------------------
# Pure helpers (tested without a model)
# -----------------------------------------------------------------------
def make_folds(items: np.ndarray, n_folds: int, seed: int) -> list:
    """Split `items` into n_folds disjoint test sets that together cover every item exactly once."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(items))
    return [np.sort(f) for f in np.array_split(perm, n_folds)]


def fit_direction(sub: torch.Tensor, refs: list) -> torch.Tensor:
    """Unit vector: mean(misogyny prompts) - average of the mean of each reference arm."""
    ref_mean = torch.stack([r.mean(dim=0) for r in refs]).mean(dim=0)
    d = sub.mean(dim=0) - ref_mean
    return d / d.norm()


def _rate(chunks: list) -> float:
    return float(np.concatenate(chunks).mean())


# -----------------------------------------------------------------------
# The pilot
# -----------------------------------------------------------------------
def run_pilot(model_id: str, layers: list = None, n_folds: int = 5, seed: int = 0) -> dict:
    """{layer -> {contrast -> {"seen": {ref: rate}, "unseen": {ref: rate}}}} plus direction cosines."""
    layers = layers or DEFAULT_LAYERS
    refs_all, contrasts = arms_and_contrasts()
    results = {}
    for l in layers:
        main_acts, main_tab = store.load_arm_activations(model_id, "main", layers=[l])
        held_acts, held_tab = store.load_arm_activations(model_id, "heldout", layers=[l])
        sub = main_acts["subtype"][l].float()
        refs_m = {a: main_acts[a][l].float() for a in refs_all}
        hsub = held_acts["subtype"][l].float()
        refs_h = {a: held_acts[a][l].float() for a in refs_all}
        pid_m, pid_h = main_tab["pair_id"].values, held_tab["pair_id"].values

        folds = make_folds(np.unique(pid_m), n_folds, seed)
        acc = {c: {"seen": {r: [] for r in refs_all}, "unseen": {r: [] for r in refs_all}} for c in contrasts}
        for fold in folds:
            test_m, test_h = np.isin(pid_m, fold), np.isin(pid_h, fold)
            for cname, ref_names in contrasts.items():
                d = fit_direction(sub[~test_m], [refs_m[r][~test_m] for r in ref_names])   # fitted on TRAIN statements only
                s_sub = sub[test_m] @ d
                u_sub = hsub[test_h] @ d
                for r in refs_all:
                    acc[cname]["seen"][r].append((s_sub > refs_m[r][test_m] @ d).numpy())
                    acc[cname]["unseen"][r].append((u_sub > refs_h[r][test_h] @ d).numpy())

        per_c = {c: {k: {r: _rate(v) for r, v in acc[c][k].items()} for k in ("seen", "unseen")} for c in contrasts}
        dirs = {c: fit_direction(sub, [refs_m[r] for r in contrasts[c]]) for c in contrasts}
        per_c["_cosines"] = {f"{a}|{b}": float(dirs[a] @ dirs[b]) for i, a in enumerate(contrasts) for b in list(contrasts)[i + 1:]}
        results[l] = per_c
    return results


def specificity(entry: dict) -> float:
    """Smallest unseen win rate against the controls (benign, hateful and, in the matched experiment, egalitarian)."""
    return min(v for r, v in entry["unseen"].items() if r != "neutral")


def print_pilot(results: dict) -> None:
    layers = sorted(results, reverse=True)
    contrasts = [c for c in results[layers[0]] if not c.startswith("_")]
    refs = list(results[layers[0]][contrasts[0]]["unseen"])
    short = {"neutral": "neu", "benign": "ben", "hateful": "hat", "egalitarian": "egl"}
    width = 66 + 12 * len(refs)
    print("\n" + "=" * width)
    print("  Training-contrast pilot (out-of-fold by statement; win rate of the misogyny prompt vs the reference prompt)")
    print(f"  columns: SEEN phrases vs {' / '.join(refs)} | UNSEEN phrases vs the same | specificity")
    print(f"  specificity = the smallest unseen win rate against the controls ({', '.join(r for r in refs if r != 'neutral')}); chance 0.50")
    print("=" * width)
    best = {}
    for c in contrasts:
        print(f"\n  contrast: {c}")
        cols = " ".join(f"{short[r]:>5}" for r in refs)
        print(f"  {'layer':>6} | {cols} | {cols} | {'spec':>5}")
        for l in layers:
            e = results[l][c]
            print(f"  {l:+6d} | " + " ".join(f"{e['seen'][r]:5.3f}" for r in refs) + " | "
                  + " ".join(f"{e['unseen'][r]:5.3f}" for r in refs) + f" | {specificity(e):5.3f}")
        bl = max(layers, key=lambda l: specificity(results[l][c]))
        best[c] = (bl, specificity(results[bl][c]))
    print("\n" + "-" * width)
    print("  best unseen specificity per contrast: " + " | ".join(f"{c}: {v:.3f} (layer {l})" for c, (l, v) in best.items()))
    mid = layers[len(layers) // 2]
    print(f"  cosine between the fitted directions at layer {mid}: "
          + ", ".join(f"{k.replace('|', ' ~ ')} {v:.2f}" for k, v in results[mid]["_cosines"].items()))
    print("=" * width + "\n")


def main(model_id: str = os.getenv("MODEL_ID"), layers: list = None, n_folds: int = 5, seed: int = 0) -> dict:
    results = run_pilot(model_id, layers, n_folds, seed)
    print_pilot(results)
    out = exp_dir(PILOT_DIR, model_id) / "pilot_contrasts.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"[pilot_contrasts] saved -> {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare training contrasts for a misogyny-specific direction.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--layers", nargs="*", type=int, default=None, help=f"negative layer indices (default {DEFAULT_LAYERS})")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    if not args.model_id:
        raise SystemExit("pass --model-id or set MODEL_ID in .env")
    main(args.model_id, args.layers, args.folds, args.seed)
    sys.exit(0)
