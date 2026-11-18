"""
Steering-coefficient calibration helpers (pure Python / numpy: no torch, no model, testable on CPU).

Why this exists
---------------
The paper steers EVERY block with a UNIT concept vector and a small control coefficient: for Llama-3.1-8B the authors
swept epsilon in {0.55 ... 0.8} (SI Appendix F) and report "nonsensical, repetitive outputs" beyond that. The window between
"nothing changes" and "collapse" is narrow, and it depends on the vector, so the coefficient has to be CALIBRATED, not guessed:

    1. sweep |epsilon| upward on a few prompts,
    2. find the smallest |epsilon| at which generations collapse (this module: `collapse_threshold`),
    3. evaluate with the judge at 60-90 % of that threshold (`working_window`),
    4. repeat with a random unit direction of the same size (the control).

`directionality` is the judge-free sanity check that came out of the v5 stance-logit test: a real concept direction moves a
stance score in OPPOSITE directions for +epsilon and -epsilon. A direction that moves it the same way for both signs is only
disrupting the model.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

PAPER_WINDOW_LLAMA31_8B = (0.55, 0.80)        # SI Appendix F; larger values -> nonsensical, repetitive outputs
DEFAULT_MAX_DEGENERATE_FRAC = 0.34            # "collapsed" = at least a third of the sampled generations are degenerate


def default_grid(max_coef: float = 1.2, step: float = 0.1) -> List[float]:
    """Ascending |epsilon| grid 0.1, 0.2, ... max_coef (rounded to avoid 0.30000000000000004)."""
    if step <= 0 or max_coef <= 0:
        raise ValueError("step and max_coef must be positive")
    n = int(round(max_coef / step))
    return [round(step * k, 6) for k in range(1, n + 1)]


def collapse_threshold(degenerate_by_coef: Dict[float, Sequence[int]],
                       max_frac: float = DEFAULT_MAX_DEGENERATE_FRAC) -> Optional[float]:
    """
    Smallest |epsilon| whose share of degenerate generations is >= max_frac AND stays collapsed at every larger tested
    |epsilon| (so one unlucky prompt at a small epsilon does not move the threshold). None if nothing collapsed.

    degenerate_by_coef : {abs_coef: [0/1 per prompt]}  (1 = judge-free degeneration heuristic fired)
    """
    if not degenerate_by_coef:
        return None
    coefs = sorted(degenerate_by_coef)
    frac = {c: (sum(degenerate_by_coef[c]) / len(degenerate_by_coef[c])) if len(degenerate_by_coef[c]) else 0.0 for c in coefs}
    for i, c in enumerate(coefs):
        if frac[c] >= max_frac and all(frac[d] >= max_frac for d in coefs[i:]):
            return c
    return None


def working_window(threshold: Optional[float], lo: float = 0.6, hi: float = 0.9, n: int = 4, decimals: int = 2) -> List[float]:
    """
    Coefficients to evaluate: n values between lo*threshold and hi*threshold (inclusive), rounded. The paper's own sets are
    of this kind (6 values just below the collapse point). Returns [] when there is no threshold.
    """
    if threshold is None or threshold <= 0:
        return []
    if not 0 < lo <= hi < 1.0 + 1e-9:
        raise ValueError("need 0 < lo <= hi <= 1")
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        vals = [(lo + hi) / 2 * threshold]
    else:
        vals = [(lo + (hi - lo) * k / (n - 1)) * threshold for k in range(n)]
    out = []
    for v in vals:
        r = round(v, decimals)
        if r > 0 and r not in out:
            out.append(r)
    return out


def regime_warnings(n_layers_hooked: int, coefs: Iterable[float], n_layers_total: int,
                    vector_norms: Optional[Iterable[float]] = None) -> List[str]:
    """Human-readable warnings when a steering run is far from the paper's configuration (all blocks, unit vectors, |eps| < ~1)."""
    msgs = []
    coefs = [float(c) for c in coefs]
    biggest = max((abs(c) for c in coefs), default=0.0)
    norms = [float(x) for x in vector_norms] if vector_norms is not None else []
    if norms and (max(norms) > 1.05 or min(norms) < 0.95):
        msgs.append(f"vector norms are in [{min(norms):.2f}, {max(norms):.2f}], not 1: the coefficients below are not on the paper's scale")
    unit = not norms or (0.95 <= min(norms) and max(norms) <= 1.05)
    frac = n_layers_hooked / max(1, n_layers_total - 1)
    if frac >= 0.6 and unit and biggest > 1.2:
        msgs.append(f"|epsilon| up to {biggest:g} on {n_layers_hooked} layers with unit vectors is beyond the point where Llama-3.1-8B collapses in the "
                    f"paper (> {PAPER_WINDOW_LLAMA31_8B[1]:g}); expect repetition. Run calibrate_steering.py to find this vector's collapse point.")
    if frac < 0.4:
        msgs.append(f"only {n_layers_hooked} of {n_layers_total - 1} blocks are hooked; the paper steers ALL blocks with a small epsilon. "
                    f"Few layers need a much larger epsilon, and that is where the off-distribution gibberish comes from.")
    return msgs


# -----------------------------------------------------------------------
# Directionality: does +epsilon and -epsilon move the stance score in OPPOSITE directions?
# -----------------------------------------------------------------------
def directionality(table: Sequence[dict], key: str = "stance_shift", min_abs: float = 0.0) -> dict:
    """
    table : rows like those saved by stance_logit_test.py ({'coef': float, key: float, 'null_stance_shift': {...}, ...}).

    For every |c| present with both signs, compares the effect at +c and at -c. A concept direction oriented toward the concept
    should give effect(+c) > 0 > effect(-c) for `stance_shift` (= effect on sexist minus effect on egalitarian statements).
    Same sign at +c and -c means the perturbation disrupts the model's judgement without moving it along the concept.
    """
    by = {round(float(r["coef"]), 6): r for r in table}
    pairs = []
    for c in sorted({abs(c) for c in by}):
        if c in by and -c in by:
            p, m = float(by[c][key]), float(by[-c][key])
            opposite = (p > min_abs) and (m < -min_abs)
            p_null = by[c].get("null_" + key, {}) or {}
            m_null = by[-c].get("null_" + key, {}) or {}
            pairs.append({"abs_coef": c, "plus": p, "minus": m, "opposite_signs": bool(opposite),
                          "asymmetry": p - m, "z_plus": p_null.get("z"), "z_minus": m_null.get("z"),
                          "p_plus": p_null.get("p"), "p_minus": m_null.get("p")})
    n_ok = sum(x["opposite_signs"] for x in pairs)
    verdict = ("directional" if pairs and n_ok == len(pairs) else
               "partly directional" if n_ok else
               "not directional" if pairs else "no +/- pairs in the table")
    return {"key": key, "pairs": pairs, "n_pairs": len(pairs), "n_opposite": n_ok, "verdict": verdict}


def format_directionality(result: dict) -> str:
    lines = [f"  directionality of {result['key']}: {result['verdict']}  ({result['n_opposite']}/{result['n_pairs']} coefficient pairs with opposite signs)",
             f"  {'|eps|':>6} {'effect(+)':>10} {'effect(-)':>10} {'opposite?':>10} {'z(+)':>6} {'z(-)':>6}"]
    f = lambda v: "   n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:6.2f}"
    for x in result["pairs"]:
        lines.append(f"  {x['abs_coef']:6.2f} {x['plus']:10.2f} {x['minus']:10.2f} {'yes' if x['opposite_signs'] else 'NO':>10} {f(x['z_plus'])} {f(x['z_minus'])}")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Where is the steering vector added? (used by anti_misogyny_steering.hook_model)
# -----------------------------------------------------------------------
STEER_POSITIONS = ("all", "generated")


def position_mask_applies(seq_len: int, positions: str = "all") -> bool:
    """
    Should a steering hook add the vector in a forward pass whose hidden states have `seq_len` positions?

    "all"       : every pass (the paper's setting and every run so far): the question tokens AND the generated tokens are steered.
    "generated" : only the incremental decoding passes (seq_len == 1, with the KV cache). The prompt pass (seq_len > 1) is left alone, so the model
                  reads the question undisturbed and the push acts only while it writes its answer. The FIRST generated token comes out of the
                  prompt pass and is therefore not steered.
    """
    if positions not in STEER_POSITIONS:
        raise ValueError(f"positions must be one of {STEER_POSITIONS}, got {positions!r}")
    return True if positions == "all" else int(seq_len) == 1


# -----------------------------------------------------------------------
# Prompt files for calibrate_steering.py --prompts-file
# -----------------------------------------------------------------------
def read_prompt_file(path) -> list:
    """One prompt per line; blank lines and lines starting with '#' are ignored. Raises if the file is missing or has no prompts."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompt file not found: {p}")
    prompts = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"no prompts in {p}")
    return prompts
