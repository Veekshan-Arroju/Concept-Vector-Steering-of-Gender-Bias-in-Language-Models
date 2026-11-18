"""
Layer profile of ONE vector: at which layers does it satisfy ALL the orderings a misogyny vector should satisfy?

For a vector that is meant to measure a misogynistic stance, a misogyny prompt should score higher than
    neu : the neutral prompt of the same statement (no instruction)
    ben : a benign-concept instruction
    hat : an instruction naming a non-gender hateful concept
    egl : (matched experiment only) an egalitarian instruction: the same topic, the opposite stance
The numbers are control_eval.py's: neu = misog>neut (pair win rate), ben / hat / egl = AUC of the misogyny prompts against that
control. Chance is 0.50. "worst" is the smallest of them.

Honest layer choice
-------------------
The layers are SELECTED on the seen-phrase test ("main") only: a layer passes when its worst-case is at least --threshold.
The unseen-phrase test ("heldout") is printed next to it as a confirmation and is never used to choose. Choosing the layer
by looking at the unseen table would inflate the result.

Layers that pass are grouped into runs of neighbouring layers: a single isolated layer is a fragile place to steer, a run of
several is a plateau.

Usage:  python layer_profile.py --vector pooled__misogyny [--threshold 0.95] [--experiment matched] [--vector-experiment X]
Needs the vector trained on many layers, e.g.
        python concept_vector_training.py --method rfm --vectors pooled__misogyny --layers $(seq -1 -1 -31) --experiment matched
"""

import argparse
import contextlib
import io
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

import control_eval as ce
import experiment
from experiment import PROJECT_ROOT, exp_dir

load_dotenv()
PROFILE_DIR = PROJECT_ROOT / "profile_results"
KEYS = (("neu", "pair_acc_subtype"), ("ben", "auroc_subtype_vs_benign"), ("hat", "auroc_subtype_vs_hateful"),
        ("egl", "auroc_subtype_vs_egalitarian"))


def orderings(row: dict) -> dict:
    """{short name -> value} for the orderings present in this experiment."""
    return {k: row[full] for k, full in KEYS if full in row}


def worst(row: dict) -> float:
    return min(orderings(row).values())


def contiguous_runs(layers: list) -> list:
    """[-26, -25, -24, -20]  ->  [[-26, -25, -24], [-20]]   (neighbouring layers grouped, most negative first)"""
    runs = []
    for l in sorted(layers):
        if runs and l == runs[-1][-1] + 1:
            runs[-1].append(l)
        else:
            runs.append([l])
    return runs


def run_profile(model_id: str, vector_name: str, threshold: float = 0.95, vector_experiment: str = None) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):               # control_eval prints a table per call
        seen = ce.main(model_id, vector_name, eval_set="main", vector_experiment=vector_experiment)
        unseen = ce.main(model_id, vector_name, eval_set="heldout", vector_experiment=vector_experiment)
    passing = [l for l in seen if worst(seen[l]) >= threshold]
    best_seen = max(seen, key=lambda l: (worst(seen[l]), sum(orderings(seen[l]).values())))
    return {"vector": vector_name, "threshold": threshold, "seen": seen, "unseen": unseen,
            "passing_on_seen": sorted(passing), "runs": contiguous_runs(passing), "best_seen_layer": best_seen}


def print_profile(r: dict) -> None:
    arms = list(orderings(next(iter(r["seen"].values()))))
    w = 30 + 14 * len(arms)
    print("\n" + "=" * w)
    print(f"  Layer profile of {r['vector']}   (chance 0.50; * = selected: worst-case on the SEEN phrases >= {r['threshold']:.2f})")
    print(f"  SEEN phrases choose the layer; UNSEEN phrases only confirm it")
    print("=" * w)
    head = " ".join(f"{a:>5}" for a in arms)
    print(f"  {'layer':>6}   | SEEN {head} {'worst':>5} | UNSEEN {head} {'worst':>5}")
    for l in sorted(r["seen"], reverse=True):
        s, u = orderings(r["seen"][l]), orderings(r["unseen"][l])
        mark = "*" if l in r["passing_on_seen"] else " "
        print(f"  {l:+6d} {mark} |      " + " ".join(f"{s[a]:5.3f}" for a in arms) + f" {worst(r['seen'][l]):5.3f} |        "
              + " ".join(f"{u[a]:5.3f}" for a in arms) + f" {worst(r['unseen'][l]):5.3f}")
    print("-" * w)
    if r["runs"]:
        print(f"  selected layers: {r['runs']}")
        for run in r["runs"]:
            uw = [worst(r["unseen"][l]) for l in run]
            print(f"    run {run[0]} .. {run[-1]} ({len(run)} layer{'s' if len(run) > 1 else ''}): unseen worst-case "
                  f"min {min(uw):.3f}, max {max(uw):.3f}")
    else:
        print(f"  no layer reaches {r['threshold']:.2f} on the seen phrases; best seen worst-case: layer {r['best_seen_layer']} "
              f"-> {worst(r['seen'][r['best_seen_layer']]):.3f} (unseen {worst(r['unseen'][r['best_seen_layer']]):.3f})")
    print("=" * w + "\n")


def main(model_id: str = os.getenv("MODEL_ID"), vector_name: str = "pooled__misogyny", threshold: float = 0.95,
         vector_experiment: str = None) -> dict:
    r = run_profile(model_id, vector_name, threshold, vector_experiment)
    print_profile(r)
    tag = f"{vector_name}" + (f"__vec-{vector_experiment}" if vector_experiment else "")
    out = exp_dir(PROFILE_DIR, model_id) / f"layer_profile__{tag}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(r, f)
    print(f"[layer_profile] saved -> {out}")
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which layers satisfy every ordering? Chosen on seen phrases, confirmed on unseen.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--vector", default="pooled__misogyny")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--vector-experiment", default=None, metavar="EXPERIMENT")
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    if not args.model_id:
        raise SystemExit("pass --model-id or set MODEL_ID in .env")
    main(args.model_id, args.vector, args.threshold, args.vector_experiment)
