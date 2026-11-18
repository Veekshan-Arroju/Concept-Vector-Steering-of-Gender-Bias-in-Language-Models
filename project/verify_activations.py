"""
Read-only check of the saved activations (run it after activation_extraction.py, before training vectors).

    python verify_activations.py [--model-id <id>] [--experiment <tag>]

What it checks (all on the saved files; no model is loaded, nothing is written):
  * meta.json matches the current dataset (registry hash), prompt count, layer list, dtype, hidden size
  * every layer file has the right shape, no NaN/Inf, no all-zero rows (a typical padding failure), sane row norms
  * no two prompts share the same activation vector (every prompt is unique, so every row must be)
  * the four arms of a fresh statement (neutral / misogyny / benign / hateful) really differ
  * the loaders used by every later step return the expected shapes (training sets, seen test, unseen test)
It prints [OK] / [WARN] / [FAIL] lines and a summary.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from dotenv import load_dotenv

import activation_store as store
import dataset_io as dio
import experiment

load_dotenv()          # reads MODEL_ID from .env like the other scripts
_levels = []


def report(level: str, msg: str) -> None:
    print(f"[{level:<4}] {msg}")
    _levels.append(level)


def _cos_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())


def verify(model_id: str) -> list:
    """Run every check; returns the list of levels reported (also printed)."""
    _levels.clear()
    registry = dio.load_registry()
    n = len(registry)

    # ---- meta ----
    try:
        meta = store.check_fresh(model_id)
    except (FileNotFoundError, RuntimeError) as e:
        report("FAIL", str(e))
        return _levels
    layers = sorted(meta["layers"], reverse=True)
    report("INFO", f"model {meta['model_id']} | experiment {meta.get('experiment')} | language {meta.get('language')} | "
                   f"dtype {meta['dtype']} | hidden size {meta['hidden_dim']}")
    report("OK" if meta["n_prompts"] == n else "FAIL", f"{meta['n_prompts']} stored prompts vs {n} in prompts.csv")
    report("OK", f"registry hash matches the current dataset; {len(layers)} layers ({layers[0]} ... {layers[-1]})")
    report("INFO", f"token read at the end of the prompt: neutral -> {meta['last_tokens_example']['neutral'][-1]!r}, "
                   f"subtype -> {meta['last_tokens_example']['subtype'][-1]!r} (should be the assistant-turn header)")

    # ---- per-layer content ----
    norm_rows = []
    bad = 0
    for l in layers:
        t = store._load_tensor(store.layer_path(model_id, l))
        if tuple(t.shape) != (n, meta["hidden_dim"]):
            report("FAIL", f"layer {l}: shape {tuple(t.shape)}, expected {(n, meta['hidden_dim'])}")
            bad += 1
            continue
        f = t.float()
        n_bad = int((~torch.isfinite(f)).sum())
        n_zero = int((f.abs().sum(dim=1) == 0).sum())
        norms = f.norm(dim=1)
        if n_bad or n_zero:
            report("FAIL", f"layer {l}: {n_bad} non-finite values, {n_zero} all-zero rows")
            bad += 1
        norm_rows.append((l, float(norms.mean()), float(norms.median()), float(norms.max())))
        if float(norms.max()) > 50 * float(norms.median()):
            report("WARN", f"layer {l}: the largest row norm is more than 50x the median (outlier rows)")
    if not bad:
        report("OK", f"all {len(layers)} layer files: right shape, no NaN/Inf, no all-zero rows")
    show = [r for r in norm_rows if r[0] in (-1, -8, -16, -24, -31)] or norm_rows[:5]
    report("INFO", "row norms (mean / median / max): " + " | ".join(f"layer {l}: {a:.1f} / {m:.1f} / {x:.1f}" for l, a, m, x in show))

    # ---- uniqueness and arms, on one middle layer ----
    mid = layers[len(layers) // 2]
    t = store._load_tensor(store.layer_path(model_id, mid)).float()
    unique_rows = len({row.numpy().tobytes() for row in t})
    report("OK" if unique_rows == n else "FAIL", f"layer {mid}: {unique_rows} distinct activation vectors for {n} distinct prompts")
    arms = dio.get_arm_table("main")
    rows = {a: t[arms[a].astype(int).tolist()] for a in dio.arm_names()}
    neu, sub = rows["neutral"], rows["subtype"]
    identical = sum(int((neu == rows[a]).all(dim=1).sum()) for a in rows if a != "neutral")
    others = "/".join(a for a in rows if a not in ("neutral",))
    report("OK" if identical == 0 else "FAIL", f"layer {mid}: neutral vs {others} rows of the same statement are never identical ({identical} identical)")
    shuffled = neu[torch.randperm(len(neu), generator=torch.Generator().manual_seed(0))]
    report("INFO", f"layer {mid}: cosine of a statement's neutral row with its own misogyny row {_cos_mean(neu, sub):.3f}, "
                   f"with a different statement's neutral row {_cos_mean(neu, shuffled):.3f}")

    # ---- extended store: the reused rows must be exactly the source rows ----
    if meta.get("extended_from"):
        src = meta["extended_from"]
        n_re, n_new = meta["n_reused"], meta["n_extracted"]
        report("OK" if n_re + n_new == n else "FAIL", f"extended from experiment {src!r}: {n_re} reused + {n_new} newly extracted = {n_re + n_new} (store has {n})")
        with experiment.using(src):
            src_registry = dio.load_registry()
            src_t = store._load_tensor(store.layer_path(model_id, mid)).float()
        same_prefix = registry["user_prompt"].iloc[:n_re].tolist() == src_registry["user_prompt"].tolist()
        report("OK" if same_prefix else "FAIL", f"the first {n_re} prompts of this experiment are identical to those of {src!r}")
        identical_rows = bool(torch.equal(t[:n_re], src_t))
        report("OK" if identical_rows else "FAIL", f"layer {mid}: the {n_re} reused activation rows are bit-identical to the source store")

    # ---- loaders used by every later step ----
    try:
        a, y = store.load_train_activations(model_id, "single__misogyny", layers=[mid])
        report("OK" if (a[mid].shape[0], len(y)) == (400, 400) else "FAIL", f"training set single__misogyny: {tuple(a[mid].shape)}, labels {y[:4]}...")
        for name in ("main", "heldout"):
            a, y = store.load_eval_activations(model_id, name, layers=[mid])
            ok = a[mid].shape[0] == len(y) and sum(y) == len(y) / 2
            report("OK" if ok else "FAIL", f"eval set {name}: {a[mid].shape[0]} rows, exactly half label 1")
    except Exception as e:   # a loader problem is exactly what this script exists to catch
        report("FAIL", f"loader error: {type(e).__name__}: {e}")

    print(f"\nSUMMARY: {_levels.count('OK')} OK, {_levels.count('WARN')} WARN, {_levels.count('FAIL')} FAIL")
    return _levels


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only check of the saved activations.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    if not args.model_id:
        raise SystemExit("pass --model-id or set MODEL_ID in .env")
    verify(args.model_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
