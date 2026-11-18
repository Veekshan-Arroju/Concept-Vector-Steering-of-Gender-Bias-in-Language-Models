"""
Where does a MISANDRY prompt sit on the misogyny vector?   (monitoring only; follows gender_swap_test.py)

gender_swap_test.py compared a misogyny prompt with its misandry twin. That is a comparison of two instruction prompts with each
other; it does not say where either of them sits relative to the ordinary controls. This script reads the same misandry and misogyny
prompts (re-using the saved readings of the gender-swap run, no model needed for them) and adds, for the SAME 60 statements, the four
control arms of the frame test: neutral (no instruction), benign, other-hatred (xenophobia, racial supremacy, ableism) and egalitarian
(gender equality, respect for women, feminism), each with the 8 training frames and with the 8 new frames.

For each layer and frame set, the win rate of a misogyny prompt, and of a misandry prompt, over the control prompt of the same statement
(averaged over the six pairs and the statements; 0.5 = no difference). 'worst' = the lowest of the four, the number the instruction monitor
is judged by, and p = how often one of 200 random unit directions at that layer is at least as good (one-sided; smallest p = 1/(k+1)).

How to read it:
  misandry 'worst' high, like misogyny's  -> the monitor fires on a persona hostile to a gender, whoever the target is;
  misandry above neutral and benign but about 0.5 against other hatred -> it fires on hostility in general, not on gender;
  misandry about 0.5 against neutral too -> it fires on women/misogyny wording only.

Usage:  python misandry_position_test.py --vector pooled__misogyny --experiment matched   (CPU: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=48 ...)
        needs results/<model>/<experiment>/gender_swap__<vector>.hidden.pt from gender_swap_test.py
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import torch

import anti_misogyny_steering as ams
import dataset_io as dio
import experiment
from experiment import PROJECT_ROOT, exp_dir
from frame_generalization_test import NEW_FRAMES, balanced, concepts_for_this_experiment, random_unit_dirs, read_arm_hidden
from gender_swap_test import FRAME_SETS, PRESELECTED, SWAP_PAIRS
from model_utils import add_model_args
from raw_data import all_concept_ids, build_neutral_prompt, build_positive_prompt, load_language

RESULTS_DIR = PROJECT_ROOT / "results"
CTRL = ("neutral", "benign", "hateful", "egalitarian")


def ctrl_key(fs: str, c: str) -> str:
    return "neutral" if c == "neutral" else f"{fs}|{c}"


def build_control_prompts(statements: list, pack: dict, concepts: dict, seed: int = 0) -> dict:
    """{'neutral' | '<frame set>|<benign/hateful/egalitarian>' -> [prompt per statement]}; frames and concepts balanced within each arm."""
    if not all(a in concepts for a in ("benign", "hateful", "egalitarian")):
        raise ValueError("this needs the egalitarian arm: use the matched experiment (--experiment matched)")
    out = {"neutral": [build_neutral_prompt(pack, s) for s in statements]}
    for fs, frames in (("training", pack["frames"]), ("new", NEW_FRAMES)):
        fpack = {**pack, "frames": frames}
        for i, arm in enumerate(("benign", "hateful", "egalitarian")):
            cs = concepts[arm]
            ci = balanced(len(statements), len(cs), seed * 1000 + 10 * i + 1)
            fi = balanced(len(statements), len(frames), seed * 1000 + 10 * i + 2)
            out[f"{fs}|{arm}"] = [build_positive_prompt(fpack, int(fi[j]), cs[int(ci[j])], s) for j, s in enumerate(statements)]
    return out


def beat_rates(scores: torch.Tensor, ctrl: dict) -> dict:
    """scores: (n_pairs, n) projections of the prompts under test; ctrl: {arm -> (n,) projections}. Win rate over the control of the same statement."""
    out, per = {}, []
    for c in CTRL:
        w = (scores > ctrl[c]).float() + 0.5 * (scores == ctrl[c]).float()
        out[c] = float(w.mean())
        per.append(w.mean(1))
    out["worst"] = min(out[c] for c in CTRL)
    out["pair_worst"] = [float(x) for x in torch.stack(per).min(0).values]
    return out


def evaluate(swap_hidden: dict, ctrl_hidden: dict, directions: dict, n_pairs: int = len(SWAP_PAIRS), n_random: int = 200, seed: int = 0) -> dict:
    """{frame set -> {layer -> {'mis': beat_rates, 'man': beat_rates, 'p_mis': .., 'p_man': ..}}}"""
    layers = sorted(directions)
    d = directions[layers[0]].shape[1]
    stack = {fs: {side: {l: torch.stack([swap_hidden[f"{fs}|{i}|{side}"][l].float() for i in range(n_pairs)]) for l in layers}
                  for side in ("mis", "man")} for fs in FRAME_SETS}
    cstack = {fs: {l: {c: ctrl_hidden[ctrl_key(fs, c)][l].float() for c in CTRL} for l in layers} for fs in FRAME_SETS}

    def rates(fs, side, l, v):
        return beat_rates(stack[fs][side][l] @ v, {c: cstack[fs][l][c] @ v for c in CTRL})

    rnd = {fs: {side: {l: [] for l in layers} for side in ("mis", "man")} for fs in FRAME_SETS}
    for j in range(n_random):
        rd = random_unit_dirs(layers, d, seed * 100003 + j)
        for fs in FRAME_SETS:
            for side in ("mis", "man"):
                for l in layers:
                    rnd[fs][side][l].append(rates(fs, side, l, rd[l][0].float())["worst"])
    out = {fs: {} for fs in FRAME_SETS}
    for fs in FRAME_SETS:
        for l in layers:
            v = directions[l][0].float()
            r = {side: rates(fs, side, l, v) for side in ("mis", "man")}
            for side in ("mis", "man"):
                r[f"p_{side}"] = (1 + sum(x >= r[side]["worst"] for x in rnd[fs][side][l])) / (len(rnd[fs][side][l]) + 1)
            out[fs][l] = r
    return out


def print_table(res: dict, k: int = 200) -> None:
    layers = sorted(res["training"], reverse=True)
    for fs in FRAME_SETS:
        w = 112
        print("\n" + "=" * w)
        print(f"  Where do the prompts sit? {fs.upper()} frames. Win rate over the control prompt of the same statement (0.5 = no difference)")
        print(f"  worst = lowest of the four; p = share of random directions at least as good on 'worst' (one-sided, {k} directions); * = p <= 0.05")
        print("=" * w)
        print(f"  {'layer':>6} | {'MISOGYNY prompts':^37} | {'MISANDRY prompts':^37} |")
        print(f"  {'':>6} | {'neu':>5} {'ben':>5} {'hat':>5} {'egl':>5} {'worst':>6} {'p':>6}  | {'neu':>5} {'ben':>5} {'hat':>5} {'egl':>5} {'worst':>6} {'p':>6}  |")
        for l in layers:
            r = res[fs][l]
            cells = []
            for side in ("mis", "man"):
                x = r[side]
                cells.append(f"{x['neutral']:5.3f} {x['benign']:5.3f} {x['hateful']:5.3f} {x['egalitarian']:5.3f} {x['worst']:6.3f} {r['p_' + side]:5.3f}{'*' if r['p_' + side] <= 0.05 else ' '}")
            print(f"  {l:+6d} | {cells[0]} | {cells[1]} |" + (" <- stance plateau" if l in PRESELECTED else ""))
        print("-" * w)
        print("  misandry, worst of the four controls per pair, at the stance-plateau layers:")
        for l in PRESELECTED:
            if l in res[fs]:
                print(f"    layer {l}: " + "  ".join(f"{i}:{x:.2f}" for i, x in enumerate(res[fs][l]["man"]["pair_worst"])))
    print("  pairs: " + "; ".join(f"{i}={a} vs {b}" for i, (a, b) in enumerate(SWAP_PAIRS)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Where does a misandry prompt sit on the misogyny vector?")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--vector", default="pooled__misogyny")
    parser.add_argument("--swap-tag", default="", help="tag of the gender_swap_test run to read")
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    add_model_args(parser)
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    if not args.model_id:
        raise ValueError("pass --model-id or set MODEL_ID in .env")
    out_dir = exp_dir(RESULTS_DIR, args.model_id)
    src = out_dir / (f"gender_swap__{args.vector}" + (f"__{args.swap_tag}" if args.swap_tag else "") + ".hidden.pt")
    if not src.exists():
        raise FileNotFoundError(f"{src} not found: run gender_swap_test.py first")
    saved = torch.load(src, map_location="cpu")
    statements = saved["statements"]
    pack = load_language(dio.load_manifest()["language"], all_concept_ids())
    prompts = build_control_prompts(statements, pack, concepts_for_this_experiment(), seed=args.seed)
    print(f"[misandry_position] {len(statements)} statements; reading {len(prompts)} control arms = {sum(len(v) for v in prompts.values())} prompts "
          f"(the misogyny and misandry readings come from {src.name}); example: {prompts['new|hateful'][0]!r}")
    model, tokenizer = ams.load_model_for_steering(args.model_id, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    directions = ams.load_directions(args.model_id, args.vector)
    ctrl_hidden = read_arm_hidden(model, tokenizer, prompts, sorted(directions), args.batch_size)
    stem = out_dir / f"misandry_position__{args.vector}"
    torch.save({"hidden": ctrl_hidden, "statements": statements}, f"{stem}.hidden.pt")
    res = evaluate(saved["hidden"], ctrl_hidden, directions, len(SWAP_PAIRS), args.n_random, args.seed)
    print_table(res, args.n_random)
    with open(f"{stem}.pkl", "wb") as f:
        pickle.dump(res, f)
    print(f"[misandry_position] saved -> {stem}.pkl")


if __name__ == "__main__":
    main()
