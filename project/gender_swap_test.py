"""
Can the instruction monitor tell MISOGYNY from MISANDRY?   (monitoring, no steering, no generation)

Every control so far (benign, non-gender hateful, egalitarian prompts) differs from misogyny prompts in topic or in stance. The control
that decides whether the vector is about MISOGYNY, and not about "a persona hostile to a gender", is the same persona aimed at the OTHER
gender. Six pairs of instruction concepts differ only in who is the target:

    dehumanization of women      <->  dehumanization of men          (held-out phrase)
    doubting women's competence  <->  doubting men's competence      (held-out phrase)
    distrust of women            <->  distrust of men                (training phrase)
    contempt for career women    <->  contempt for career men        (training phrase)
    infantilization of women     <->  infantilization of men         (training phrase)
    misogyny                     <->  misandry                       (the single phrase)

The misandry concepts were never used for training or for any earlier test. For each statement, BOTH members of a pair are wrapped in the
SAME instruction frame and the SAME statement, so the only difference is the target gender. At every layer the table gives the win rate:
the share of (pair, statement) cases in which the misogyny prompt projects HIGHER on the vector than its misandry twin (0.5 = the vector
cannot tell, 1.0 = always the misogyny one, 0.0 = always the misandry one), run with the 8 TRAINING frames and with the 8 NEW frames of
frame_generalization_test.py. It is compared with 200 random unit directions at the same layer (empirical two-sided p; smallest p = 1/(k+1)).

How to read it:  about 0.5 -> the monitor detects hostility toward a gender, not misogyny.  Close to 1.0 over the stance-plateau layers ->
it reads the target.  Below 0.5 -> it prefers the misandry persona (unexpected; report it).

Usage:  python gender_swap_test.py --vector pooled__misogyny --experiment matched      (CPU: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=48 ...)
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

import anti_misogyny_steering as ams
import dataset_io as dio
import experiment
from experiment import PROJECT_ROOT, exp_dir
from frame_generalization_test import NEW_FRAMES, balanced, random_unit_dirs, read_arm_hidden
from model_utils import add_model_args
from raw_data import all_concept_ids, build_positive_prompt, load_language
from stance_logit_test import null_summary

RESULTS_DIR = PROJECT_ROOT / "results"
SWAP_PAIRS = [
    ("dehumanization of women", "dehumanization of men"),
    ("doubting women's competence", "doubting men's competence"),
    ("distrust of women", "distrust of men"),
    ("contempt for career women", "contempt for career men"),
    ("infantilization of women", "infantilization of men"),
    ("misogyny", "misandry"),
]
FRAME_SETS = ("training", "new")
PRESELECTED = (-23, -24, -25, -26)


def pack_with_swaps(pack: dict, pairs: list = None) -> dict:
    """A copy of the language pack whose phrase table also knows the misandry concepts (concept id = phrase text)."""
    phrases = dict(pack["phrases"])
    for a, b in (SWAP_PAIRS if pairs is None else pairs):
        phrases.setdefault(a, a)
        phrases.setdefault(b, b)
    return {**pack, "phrases": phrases}


def build_swap_prompts(statements: list, pack: dict, pairs: list = None, seed: int = 0) -> dict:
    """
    {'<frame set>|<pair index>|mis' / '|man' -> [prompt per statement]}.  Both members of a pair use the SAME frame for the same
    statement, so the target gender is the only difference.
    """
    pairs = SWAP_PAIRS if pairs is None else pairs
    pk = pack_with_swaps(pack, pairs)
    out = {}
    for fs, frames in (("training", pk["frames"]), ("new", NEW_FRAMES)):
        fpk = {**pk, "frames": frames}
        for i, (mis, man) in enumerate(pairs):
            fi = balanced(len(statements), len(frames), seed * 1000 + 10 * i + 1)
            out[f"{fs}|{i}|mis"] = [build_positive_prompt(fpk, int(fi[j]), mis, s) for j, s in enumerate(statements)]
            out[f"{fs}|{i}|man"] = [build_positive_prompt(fpk, int(fi[j]), man, s) for j, s in enumerate(statements)]
    return out


def pooled_win_rate(hidden: dict, directions: dict, fs: str, n_pairs: int, layer: int, direction: torch.Tensor = None) -> tuple:
    """(per-pair win rates, pooled win rate): share of cases where the misogyny prompt projects higher than its misandry twin."""
    v = (directions[layer][0] if direction is None else direction).float()
    rates = []
    for i in range(n_pairs):
        a = hidden[f"{fs}|{i}|mis"][layer].float() @ v
        b = hidden[f"{fs}|{i}|man"][layer].float() @ v
        rates.append(float(((a > b).float() + 0.5 * (a == b).float()).mean()))
    return rates, float(np.mean(rates))


def run_swap_test(hidden: dict, directions: dict, n_pairs: int = len(SWAP_PAIRS), n_random: int = 200, seed: int = 0) -> dict:
    """{frame set -> {layer -> {'pairs': [...], 'pooled': x, 'null': null_summary}}}"""
    layers = sorted(directions)
    d = directions[layers[0]].shape[1]
    out = {fs: {} for fs in FRAME_SETS}
    rnd = {fs: {l: [] for l in layers} for fs in FRAME_SETS}
    for j in range(n_random):
        rd = random_unit_dirs(layers, d, seed * 100003 + j)
        for fs in FRAME_SETS:
            for l in layers:
                rnd[fs][l].append(pooled_win_rate(hidden, directions, fs, n_pairs, l, rd[l][0])[1])
    for fs in FRAME_SETS:
        for l in layers:
            pairs, pooled = pooled_win_rate(hidden, directions, fs, n_pairs, l)
            out[fs][l] = {"pairs": pairs, "pooled": pooled, "null": null_summary(pooled, rnd[fs][l])}
    return out


def print_table(res: dict, pairs: list = None) -> None:
    pairs = SWAP_PAIRS if pairs is None else pairs
    layers = sorted(res["training"], reverse=True)
    k = res["training"][layers[0]]["null"]["n"]
    w = 96
    print("\n" + "=" * w)
    print("  Can the monitor tell misogyny from misandry? win rate of the misogyny prompt over its misandry twin (0.5 = cannot tell)")
    print(f"  same frame, same statement, only the target gender differs; p = empirical two-sided p against {k} random directions (smallest p = {1 / (k + 1):.3f}); * = p <= 0.05")
    print("=" * w)
    print(f"  {'layer':>6} | {'TRAINING frames':^22} | {'NEW frames':^22} |")
    print(f"  {'':>6} | {'win rate':>9} {'p':>7}{'':>5} | {'win rate':>9} {'p':>7}{'':>5} |")
    for l in layers:
        cells = []
        for fs in FRAME_SETS:
            r = res[fs][l]
            cells.append(f"{r['pooled']:9.3f} {r['null']['p']:6.3f}{'*' if r['null']['p'] <= 0.05 else ' '}    ")
        mark = " <- stance plateau" if l in PRESELECTED else ""
        print(f"  {l:+6d} | {cells[0]} | {cells[1]} |{mark}")
    print("-" * w)
    print("  per pair, at the stance-plateau layers (training frames / new frames):")
    for l in PRESELECTED:
        if l in res["training"]:
            print(f"    layer {l}: " + "  ".join(f"{i}:{res['training'][l]['pairs'][i]:.2f}/{res['new'][l]['pairs'][i]:.2f}" for i in range(len(pairs))))
    print("  pairs: " + "; ".join(f"{i}={a} vs {b}" for i, (a, b) in enumerate(pairs)))
    print("=" * w + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Can the instruction monitor tell misogyny from misandry?")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--vector", default="pooled__misogyny")
    parser.add_argument("--n-statements", type=int, default=60, help="statements per pair (a random subset of the 218 fresh ones)")
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    add_model_args(parser)
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    if not args.model_id:
        raise ValueError("pass --model-id or set MODEL_ID in .env")
    reg = dio.load_registry()
    allst = reg[reg["kind"] == "test_neutral"].sort_values("pair_id")["statement"].tolist()
    pick = np.sort(np.random.default_rng(args.seed).choice(len(allst), size=min(args.n_statements, len(allst)), replace=False))
    statements = [allst[i] for i in pick]
    pack = load_language(dio.load_manifest()["language"], all_concept_ids())
    prompts = build_swap_prompts(statements, pack, seed=args.seed)
    print(f"[gender_swap] {len(statements)} statements x {len(SWAP_PAIRS)} pairs x 2 members x 2 frame sets = {sum(len(v) for v in prompts.values())} prompts; "
          f"example: {prompts['new|1|man'][0]!r}")
    model, tokenizer = ams.load_model_for_steering(args.model_id, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    directions = ams.load_directions(args.model_id, args.vector)
    hidden = read_arm_hidden(model, tokenizer, prompts, sorted(directions), args.batch_size)
    out_dir = exp_dir(RESULTS_DIR, args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / (f"gender_swap__{args.vector}" + (f"__{args.tag}" if args.tag else ""))
    torch.save({"hidden": hidden, "statements": statements}, f"{stem}.hidden.pt")        # so the statistics can be redone without the model
    res = run_swap_test(hidden, directions, len(SWAP_PAIRS), args.n_random, args.seed)
    print_table(res)
    with open(f"{stem}.pkl", "wb") as f:
        pickle.dump(res, f)
    print(f"[gender_swap] saved -> {stem}.pkl")


if __name__ == "__main__":
    main()
