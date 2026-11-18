"""
Compare the combined vectors on SEEN and UNSEEN phrases.

    S  single__misogyny       trained on the phrase "misogyny" only (never saw any subtype phrase)
    P  pooled__misogyny       trained on a mix of the NON-held-out subtypes
    A  averaged__misogyny     mean of the NON-held-out subtype vectors
    A* averaged_all__misogyny mean of ALL subtype vectors  -> has seen the held-out phrases, reference only
    O  oracle                 each held-out phrase's own vector (trained on exactly that phrase) -> reference only

The question: do P and A, which saw varied subtype phrases, detect misogyny wordings none of S / P / A has seen
better than the single phrase S does?

Method (all on saved activations, no model is loaded)
-----------------------------------------------------
* Score = projection of the final-token hidden state on the vector, layer by layer.
* A pair WINS when the prompt with the misogyny instruction scores higher than the REFERENCE prompt of the same statement
  (--against): the neutral twin, the benign-concept twin, the non-gender hateful twin, or (matched experiment only) the
  egalitarian twin (same topic, opposite stance). Pair accuracy = fraction of wins
  (chance 0.5; below 0.5 means the reference prompts outscore the misogyny prompts).
* Against NEUTRAL almost any instruction wins (a vector for "an instruction is present" scores 1.000), so that comparison
  saturates and cannot tell S, P and A apart. The informative comparisons are --against benign and --against hateful;
  --against all runs the three.
* The layer of each vector is chosen on the SEEN-phrase test ("main"), not on the held-out test, so the held-out
  number is not inflated by picking its best layer. (--layer L forces one layer for every vector instead.)
* Held-out test = every fresh statement x every held-out phrase. The 4 phrase-versions of one statement share the
  same sentence, so they are not independent: confidence intervals resample STATEMENTS (cluster bootstrap), keeping all
  of a statement's phrase-versions together. Differences between vectors use the SAME resamples (paired bootstrap).

Usage:  python compare_vectors.py --against all [--model-id <id>] [--experiment tag] [--n-boot 2000] [--layer -16]
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
from contextlib import nullcontext

import numpy as np

from activation_store import load_arm_activations
from concept_vector_training import load_concept_vectors, vector_exists
from dataset_io import arm_names, get_arm_table, heldout_concepts, subtype_vector_names
import experiment
from experiment import PROJECT_ROOT, exp_dir
from raw_data import AVERAGED_ALL_NAME, AVERAGED_NAME, POOLED_NAME, SINGLE_NAME

COMPARISON_DIR = PROJECT_ROOT / "comparison_results"
AGAINST = ("neutral", "benign", "hateful", "egalitarian")    # egalitarian exists only in the matched experiment
FAIR = [SINGLE_NAME, POOLED_NAME, AVERAGED_NAME]          # have not seen the held-out phrases
REFERENCE = [AVERAGED_ALL_NAME]                           # have seen them: reference only


# -----------------------------------------------------------------------
# Pure numpy helpers (tested without any model)
# -----------------------------------------------------------------------
def pair_wins(projections: np.ndarray) -> np.ndarray:
    """projections laid out [neutral_0, misogyny_0, neutral_1, misogyny_1, ...] -> bool wins per pair."""
    return projections[1::2] > projections[0::2]


def win_matrix(wins: np.ndarray, pair_ids: np.ndarray, concepts: np.ndarray, phrase_order: list) -> np.ndarray:
    """
    wins/pair_ids/concepts are per held-out row. Returns W[n_statements, n_phrases] (bool), statements in sorted pair_id
    order and phrases in `phrase_order`.
    """
    stmts = np.unique(pair_ids)
    W = np.zeros((len(stmts), len(phrase_order)), dtype=bool)
    pos = {p: i for i, p in enumerate(stmts)}
    col = {c: k for k, c in enumerate(phrase_order)}
    for w, p, c in zip(wins, pair_ids, concepts):
        W[pos[p], col[c]] = w
    return W


def cluster_bootstrap(W_by_name: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Resample statements (rows of every W) with replacement, jointly for all vectors.
    Returns {"acc": {name: (point, lo, hi)}, "diff": {(a, b): (point, lo, hi)}}  (diff = acc_a - acc_b).
    """
    names = list(W_by_name)
    n = next(iter(W_by_name.values())).shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    acc = {k: W.astype(np.float32)[idx].mean(axis=(1, 2)) for k, W in W_by_name.items()}
    out = {"acc": {}, "diff": {}}
    for k in names:
        out["acc"][k] = (float(W_by_name[k].mean()), *np.percentile(acc[k], [2.5, 97.5]).tolist())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = acc[a] - acc[b]
            out["diff"][(a, b)] = (float(W_by_name[a].mean() - W_by_name[b].mean()), *np.percentile(d, [2.5, 97.5]).tolist())
    return out


# -----------------------------------------------------------------------
# Scoring from saved activations
# -----------------------------------------------------------------------
def score_vectors(model_id: str, vectors: dict, eval_sets=("main", "heldout"), against: str = "neutral") -> dict:
    """
    {set_name -> {vector_name -> {layer -> bool array of pair wins}}}, loading one layer at a time.
    A win = the misogyny prompt scores higher than the `against` prompt of the same statement.
    `vectors`: {name -> {layer -> Tensor(1, d)}}
    """
    if against not in AGAINST:
        raise ValueError(f"against must be one of {AGAINST}")
    layers = sorted(set.intersection(*[set(v) for v in vectors.values()]))
    out = {s: {n: {} for n in vectors} for s in eval_sets}
    for l in layers:
        for s in eval_sets:
            acts, _ = load_arm_activations(model_id, s, layers=[l])
            if against not in acts:
                raise ValueError(f"This experiment has no {against!r} arm (arms: {sorted(acts)}). "
                                 f"The egalitarian arm exists only in the matched experiment (dataset_builder.py --negatives matched).")
            sub, ref = acts["subtype"][l].float(), acts[against][l].float()
            for n, d in vectors.items():
                v = d[l][0].float()
                out[s][n][l] = (sub @ v).numpy() > (ref @ v).numpy()
    return out


def choose_layer(main_wins_by_layer: dict) -> int:
    """Layer with the highest seen-phrase pair accuracy (first one on ties)."""
    return max(sorted(main_wins_by_layer), key=lambda l: main_wins_by_layer[l].mean())


def choose_layer_min_over_arms(wins_by_arm: dict) -> int:
    """
    {arm -> {layer -> bool array of seen-phrase pair wins}}  ->  the layer whose WORST arm is best.
    Ties: the higher mean over arms, then the first (most negative) layer. Uses only the seen-phrase test.
    """
    layers = sorted(next(iter(wins_by_arm.values())))

    def key(l):
        accs = [float(wins_by_arm[a][l].mean()) for a in wins_by_arm]
        return (min(accs), float(np.mean(accs)))
    return max(layers, key=key)


def select_layers_min_over_arms(model_id: str, vector_names: list, vector_experiment: str = None) -> dict:
    """
    One layer per vector, chosen on the SEEN-phrase test only, as the layer with the best worst-case win rate over every
    reference arm (neutral, benign, hateful and, in the matched experiment, egalitarian). Choosing a layer per table, as
    --against X alone does, can give a vector a different layer in every table and favours early, wording-driven layers.
    """
    arms = [a for a in AGAINST if a in arm_names()]
    with (experiment.using(vector_experiment) if vector_experiment else nullcontext()):
        vectors = {n: load_concept_vectors(model_id, n)[0] for n in vector_names if vector_exists(model_id, n)}
    per_arm = {a: score_vectors(model_id, vectors, eval_sets=("main",), against=a)["main"] for a in arms}
    return {n: choose_layer_min_over_arms({a: per_arm[a][n] for a in arms}) for n in vectors}


def compare(model_id: str, vector_names: list = None, n_boot: int = 2000, seed: int = 0, force_layer: int = None,
            against: str = "neutral", vector_experiment: str = None, layer_by_vector: dict = None) -> dict:
    """
    `vector_experiment`: load the vectors from that experiment, score them on the data of the current one.
    `layer_by_vector`: {vector -> layer} fixed in advance (see select_layers_min_over_arms); overrides the per-arm choice.
    """
    heldout = heldout_concepts()
    held_vec_names = subtype_vector_names("heldout")
    with (experiment.using(vector_experiment) if vector_experiment else nullcontext()):
        names = [n for n in (vector_names or FAIR + REFERENCE) if vector_exists(model_id, n)]
        oracle = [n for n in held_vec_names if vector_exists(model_id, n)]
        vectors = {n: load_concept_vectors(model_id, n)[0] for n in names + oracle}
    wins = score_vectors(model_id, vectors, against=against)

    held_tab = get_arm_table("heldout")
    pair_ids, concepts = held_tab["pair_id"].values, held_tab["subtype_concept"].values

    chosen, W = {}, {}
    for n in vectors:
        chosen[n] = (force_layer if force_layer is not None
                     else layer_by_vector[n] if layer_by_vector is not None and n in layer_by_vector
                     else choose_layer(wins["main"][n]))
        W[n] = win_matrix(wins["heldout"][n][chosen[n]], pair_ids, concepts, heldout)

    boot_in = {n: W[n] for n in names}
    if len(oracle) == len(heldout):
        # oracle: column k = the vector trained on held-out phrase k, scored on phrase k
        W_oracle = np.stack([W[o][:, k] for k, o in enumerate(oracle)], axis=1)
        boot_in["oracle (own-phrase vectors)"] = W_oracle
    boot = cluster_bootstrap(boot_in, n_boot=n_boot, seed=seed)

    result = {
        "model_id": model_id, "against": against, "vector_experiment": vector_experiment,
        "layer_rule": ("forced" if force_layer is not None else "min over arms on the seen test" if layer_by_vector is not None
                       else "per arm on the seen test"), "heldout": heldout, "chosen_layer": chosen, "n_statements": int(W[next(iter(W))].shape[0]),
        "seen_acc": {n: float(wins["main"][n][chosen[n]].mean()) for n in names},
        "heldout_acc": boot["acc"], "diff": boot["diff"],
        "per_phrase_acc": {n: {h: float(W[n][:, k].mean()) for k, h in enumerate(heldout)} for n in names},
        "per_layer_heldout_acc": {n: {l: float(w.mean()) for l, w in wins["heldout"][n].items()} for n in names},
        "per_layer_seen_acc": {n: {l: float(w.mean()) for l, w in wins["main"][n].items()} for n in names},
    }
    return result


def print_comparison(r: dict) -> None:
    held = r["heldout"]
    print("\n" + "=" * 92)
    print(f"  Unseen-phrase comparison: {r['n_statements']} statements x {len(held)} held-out phrases = "
          f"{r['n_statements'] * len(held)} pairs (chance 0.50)")
    print(f"  held-out phrases: {held}")
    if r.get("vector_experiment"):
        print(f"  vectors trained in experiment {r['vector_experiment']!r}, scored on the data of the current experiment")
    print(f"  a pair WINS when the misogyny prompt scores higher than the {r['against'].upper()} prompt of the same statement"
          + ("" if r["against"] == "neutral" else " (below 0.50: the control prompts outscore the misogyny prompts)"))
    print(f"  layer of each vector: {r.get('layer_rule', 'per arm on the seen test')}; intervals resample statements (95%)")
    print("=" * 92)
    print(f"  {'vector':<30} {'layer':>6} {'seen acc':>9}   {'unseen acc [95% CI]':<26}")
    print("-" * 92)
    for n, (pt, lo, hi) in r["heldout_acc"].items():
        layer = r["chosen_layer"].get(n)
        seen = r["seen_acc"].get(n)
        note = "  (saw the held-out phrases: reference only)" if n == AVERAGED_ALL_NAME else ""
        print(f"  {n:<30} {layer if layer is not None else '':>6} {'' if seen is None else format(seen, '9.3f'):>9}   "
              f"{pt:5.3f} [{lo:5.3f}, {hi:5.3f}]{note}")
    print("-" * 92)
    print("  differences in unseen accuracy (first minus second; interval excludes 0 = clear difference):")
    for (a, b), (pt, lo, hi) in r["diff"].items():
        if AVERAGED_ALL_NAME in (a, b) or "oracle" in a or "oracle" in b:
            continue
        flag = "  *" if (lo > 0 or hi < 0) else ""
        print(f"    {a:<26} - {b:<26} {pt:+6.3f} [{lo:+6.3f}, {hi:+6.3f}]{flag}")
    print("-" * 92)
    print("  per held-out phrase (unseen accuracy at the chosen layer):")
    print(f"    {'vector':<30}" + "".join(f"{h[:18]:>20}" for h in held))
    for n, d in r["per_phrase_acc"].items():
        print(f"    {n:<30}" + "".join(f"{d[h]:20.3f}" for h in held))
    print("=" * 92 + "\n")


def main(model_id: str = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"), vector_names: list = None,
         n_boot: int = 2000, seed: int = 0, force_layer: int = None, against: str = "neutral",
         vector_experiment: str = None, select: str = None):
    """
    against: 'neutral' | 'benign' | 'hateful' | 'egalitarian' | 'all'. Returns the result dict (or {arm: result} for 'all').
    select : 'min-controls' = ONE layer per vector for every table, chosen on the seen test as the best worst-case over all arms
             'per-arm'      = a layer per table, chosen on that table's own seen-phrase accuracy
             default: min-controls for 'all', per-arm for a single arm.
    """
    arms = [a for a in AGAINST if a in arm_names()] if against == "all" else [against]
    select = select or ("min-controls" if against == "all" else "per-arm")
    layer_by_vector = None
    if select == "min-controls" and force_layer is None:
        wanted = [n for n in (vector_names or FAIR + REFERENCE)] + subtype_vector_names("heldout")
        layer_by_vector = select_layers_min_over_arms(model_id, wanted, vector_experiment)
        print(f"[compare_vectors] layers chosen on the seen test (best worst-case over {arms + ['neutral'] if 'neutral' not in arms else arms}): "
              + ", ".join(f"{n.split('__')[0]}:{l}" for n, l in layer_by_vector.items() if n in (vector_names or FAIR + REFERENCE)))
    results = {}
    for arm in arms:
        r = compare(model_id, vector_names, n_boot, seed, force_layer, against=arm, vector_experiment=vector_experiment,
                    layer_by_vector=layer_by_vector)
        print_comparison(r)
        suffix = f"__vec-{vector_experiment}" if vector_experiment else ""
        out = exp_dir(COMPARISON_DIR, model_id) / f"compare_unseen_phrases__{arm}{suffix}.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(r, f)
        print(f"[compare_vectors] saved -> {out}")
        results[arm] = r
    return results if against == "all" else results[against]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare S, P and A on unseen (held-out) phrases.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--vectors", nargs="*", default=None, help=f"default: {FAIR + REFERENCE}")
    parser.add_argument("--against", default="all", choices=list(AGAINST) + ["all"],
                        help="reference prompt of each pair: neutral (saturates), benign, hateful, or all three (default)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=None, help="force this (negative) layer for every vector instead of choosing on the seen test")
    parser.add_argument("--select", default=None, choices=["min-controls", "per-arm"],
                        help="layer choice (always on the seen test): min-controls = one layer per vector, best worst-case over all arms "
                             "(default with --against all); per-arm = a layer per table (default for a single arm)")
    parser.add_argument("--vector-experiment", default=None, metavar="EXPERIMENT",
                        help="load the vectors from this experiment (e.g. main) and score them on the data of --experiment")
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    main(args.model_id, args.vectors, args.n_boot, args.seed, args.layer, args.against, args.vector_experiment, args.select)
