"""
Four-arm control evaluation: does a vector track MISOGYNY, or just "an instruction to adopt a stance"?

The paired tests in linear_probe / direction_projection_probe / projection_monitor compare
neutral vs misogyny prompts. A vector that only encodes "role-play instruction present" or
"hateful request present" passes that test too, because the misogyny prompt carries those
features as well. So every fresh statement is also rendered with

    benign  : same instruction frame, benign concept   ("...believes in minimalism")
    hateful : same instruction frame, non-gender hateful concept ("...endorses xenophobia")

For each layer and each vector this script reports (all computed from raw projections a_l . v_l):

    pair_acc_<arm>        fraction of statements where the arm scores HIGHER than its neutral twin
    auroc_subtype_vs_*    how well the misogyny arm is separated from the benign / hateful arm

How to read it (a guide, not a verdict):
    * misogyny > neutral high, benign > neutral ~0.5, hateful > neutral ~0.5
          -> specific to misogyny content
    * benign > neutral also high
          -> the vector mostly encodes the generic "adopt a persona" instruction
    * hateful > neutral high but benign low
          -> the vector encodes "hateful stance" in general, not misogyny in particular
    Refusal/hedging behaviour can only be judged from generated text (steering_eval.py).

Everything is raw cosine-free projection arithmetic on saved activations: no model is loaded.
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from activation_store import load_arm_activations
from concept_vector_training import DEFAULT_VECTOR_NAME, load_concept_vectors
from dataset_io import concept_for_vector
import experiment
from experiment import PROJECT_ROOT, exp_dir

CONTROL_RESULTS_DIR = PROJECT_ROOT / "control_results"


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def evaluate_controls(arm_activations: dict, directions: dict) -> dict:
    """
    Args:
        arm_activations : {arm -> {layer -> Tensor(n_pairs, d)}} from load_arm_activations()
        directions      : {layer -> Tensor(1, d)} signed unit vectors
    Returns:
        {layer -> metrics dict}
    """
    results = {}
    for layer in sorted(directions.keys()):
        v = directions[layer][0].float()
        proj = {arm: (acts[layer].float() @ v).numpy() for arm, acts in arm_activations.items()}
        neutral = proj["neutral"]
        row = {f"pair_acc_{arm}": float(np.mean(proj[arm] > neutral)) for arm in ("subtype", "benign", "hateful")}
        row.update({
            "auroc_subtype_vs_neutral": _auroc(proj["subtype"], neutral),
            "auroc_subtype_vs_benign":  _auroc(proj["subtype"], proj["benign"]),
            "auroc_subtype_vs_hateful": _auroc(proj["subtype"], proj["hateful"]),
        })
        row["specificity"] = min(row["auroc_subtype_vs_benign"], row["auroc_subtype_vs_hateful"])
        if "egalitarian" in proj:   # matched experiment: topic-matched, opposite-stance control
            row["pair_acc_egalitarian"] = float(np.mean(proj["egalitarian"] > neutral))
            row["auroc_subtype_vs_egalitarian"] = _auroc(proj["subtype"], proj["egalitarian"])
            row["specificity_all"] = min(row["specificity"], row["auroc_subtype_vs_egalitarian"])
        row.update({f"mean_proj_{arm}": float(np.mean(p)) for arm, p in proj.items()})
        results[layer] = row
    return results


def print_control_summary(results: dict, n_pairs: int) -> None:
    egal = any("auroc_subtype_vs_egalitarian" in r for r in results.values())
    width = 100 if egal else 86
    print("\n" + "=" * width)
    print(f"  Control evaluation  ({n_pairs} pairs; chance = 0.50)")
    print("=" * width)
    head = (f"  {'Layer':>6}  {'misog>neut':>10}  {'benign>neut':>11}  {'hateful>neut':>12}  "
            f"{'AUC mis/ben':>11}  {'AUC mis/hat':>11}")
    if egal:
        head += f"  {'egal>neut':>9}  {'AUC mis/egal':>12}"
    print(head)
    print("-" * width)
    key = "specificity_all" if egal else "specificity"
    for l in sorted(results):
        r = results[l]
        line = (f"  {l:+6d}  {r['pair_acc_subtype']:10.3f}  {r['pair_acc_benign']:11.3f}  {r['pair_acc_hateful']:12.3f}  "
                f"{r['auroc_subtype_vs_benign']:11.3f}  {r['auroc_subtype_vs_hateful']:11.3f}")
        if egal:
            line += f"  {r['pair_acc_egalitarian']:9.3f}  {r['auroc_subtype_vs_egalitarian']:12.3f}"
        print(line)
    best = max(results, key=lambda l: results[l][key])
    print("-" * width)
    what = "min(AUC mis/ben, mis/hat, mis/egal)" if egal else "min(AUC mis/ben, AUC mis/hat)"
    print(f"  Most specific layer (max of {what}): {best:+d}  ->  {results[best][key]:.3f}")
    print("=" * width + "\n")


def _results_path(model_id: str, tag: str, save_dir: Path) -> Path:
    return exp_dir(save_dir, model_id) / f"control_eval__{tag}.pkl"


def main(model_id: str = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
         vector_name: str = DEFAULT_VECTOR_NAME, eval_scope: str = "all", eval_set: str = "main",
         save_dir: Path = CONTROL_RESULTS_DIR, vector_experiment: str = None) -> dict:
    """
    eval_set  : "main" = fresh statements with SEEN phrases; "heldout" = with the HELD-OUT phrases (872 pairs)
    eval_scope: "all" or "own" (own = only pairs rendered with this subtype vector's phrase)
    vector_experiment: load the VECTOR from this experiment while the activations and test sets come from the current one.
                       This is how vectors trained with one contrast (e.g. "main", neutral negatives) are scored on exactly
                       the same prompts as vectors trained with another (e.g. "matched").
    """
    with (experiment.using(vector_experiment) if vector_experiment else nullcontext()):
        directions, signs, method = load_concept_vectors(model_id=model_id, vector_name=vector_name)
    layers = sorted(directions.keys())

    concept = None
    if eval_scope == "own":
        concept = concept_for_vector(vector_name)
        if concept is None:
            raise ValueError("eval_scope='own' needs a subtype__* vector")

    # one layer at a time: the held-out set has 4 arms x 872 rows, which is large for all layers at once
    results, n_pairs = {}, 0
    for layer in layers:
        arm_acts, table = load_arm_activations(model_id, eval_set, concept=concept, layers=[layer])
        n_pairs = len(table)
        results.update(evaluate_controls(arm_acts, {layer: directions[layer]}))
    print(f"[control_eval] vector={vector_name} eval_set={eval_set} scope={eval_scope} method={method}")
    if vector_experiment:
        print(f"[control_eval] vector trained in experiment {vector_experiment!r}, scored on the data of experiment {experiment.get_experiment()!r}")
    print_control_summary(results, n_pairs=n_pairs)

    tag = f"{vector_name}__{eval_set}__{eval_scope}" + (f"__vec-{vector_experiment}" if vector_experiment else "")
    out = _results_path(model_id, tag, save_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"[control_eval] saved -> {out}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Four-arm control evaluation for one named vector.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--vector", default=DEFAULT_VECTOR_NAME)
    parser.add_argument("--set", dest="eval_set", default="main", choices=["main", "heldout"], help="main = seen phrases, heldout = unseen phrases")
    parser.add_argument("--scope", default="all", choices=["all", "own"])
    parser.add_argument("--vector-experiment", default=None, metavar="EXPERIMENT",
                        help="load the vector from this experiment (e.g. main) and score it on the data of --experiment")
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    main(model_id=args.model_id, vector_name=args.vector, eval_scope=args.scope, eval_set=args.eval_set,
         vector_experiment=args.vector_experiment)
