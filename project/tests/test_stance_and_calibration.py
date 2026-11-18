"""Stance-contrast dataset + coefficient calibration helpers (pandas/numpy only, no torch). Run: python -m pytest tests/test_stance_and_calibration.py -q"""
import _path  # noqa: F401
import pickle
import shutil
from pathlib import Path

import pytest

import coef_calibration as cc
import dataset_builder as db
import dataset_io as dio
import experiment
from judge import degeneration_heuristic
from raw_data import (EGALITARIAN_CONTROL_CONCEPTS, MISOGYNY_SUBTYPES, TRAIN_EGALITARIAN_CONCEPTS, load_language, phrase)

ROOT = Path(__file__).resolve().parent.parent
STANCE_EXP = "pytest_stance"


def _cleanup():
    shutil.rmtree(ROOT / "dataset" / "processed" / STANCE_EXP, ignore_errors=True)
    experiment.set_experiment("main")


# ------------------------------------------------------------------ stance dataset
def test_stance_build_validates_and_appends_only():
    base, st = db.build_all(), db.build_all(negatives="stance")
    db.validate(st)
    n_base = st["meta"]["n_base_prompts"]
    assert st["registry"]["user_prompt"].iloc[:n_base].tolist() == base["registry"]["user_prompt"].tolist(), "base prompts must be identical to the neutral build"
    assert set(st["registry"].iloc[n_base:]["kind"]) == {db.STANCE_KIND, "test_egalitarian"}
    assert len(st["registry"]) - n_base == 200 + st["test"]["pair_id"].nunique()
    assert st["meta"]["extra_arms"] == ["egalitarian"] and st["meta"]["train_egalitarian_controls"] == TRAIN_EGALITARIAN_CONCEPTS


def test_stance_negatives_share_the_statement_and_name_an_egalitarian_concept():
    t = db.build_all(negatives="stance")
    reg = t["registry"].set_index("prompt_id")
    for name, d in t["train"].groupby("dataset"):
        d = d.sort_values("row_idx")
        pos, neg = d.iloc[0::2], d.iloc[1::2]
        assert (reg.loc[neg["prompt_id"], "statement"].values == reg.loc[pos["prompt_id"], "statement"].values).all(), name
        assert set(reg.loc[neg["prompt_id"], "kind"]) == {db.STANCE_KIND}, name
        assert set(reg.loc[neg["prompt_id"], "concept"]) <= set(TRAIN_EGALITARIAN_CONCEPTS), name
    counts = t["registry"][t["registry"]["kind"] == db.STANCE_KIND]["concept"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_train_egalitarian_concepts_never_overlap_test_concepts_or_misogyny_phrases():
    pack = load_language("en")
    texts = [phrase(pack, c) for c in TRAIN_EGALITARIAN_CONCEPTS]
    assert not set(TRAIN_EGALITARIAN_CONCEPTS) & set(EGALITARIAN_CONTROL_CONCEPTS)
    for tr in texts:                                  # no train phrase contains (or is contained in) a test egalitarian phrase
        assert not any(te in tr or tr in te for te in (phrase(pack, c) for c in EGALITARIAN_CONTROL_CONCEPTS)), tr
    mis = [phrase(pack, c) for c in ["misogyny"] + MISOGYNY_SUBTYPES]
    assert not any(m in tr for m in mis for tr in texts)


def test_stance_is_deterministic_and_differs_from_matched():
    a = db.registry_hash(db.build_all(negatives="stance")["registry"])
    b = db.registry_hash(db.build_all(negatives="stance")["registry"])
    c = db.registry_hash(db.build_all(negatives="matched")["registry"])
    assert a == b and a != c


def test_stance_io_arm_names_and_manifest():
    _cleanup()
    try:
        experiment.set_experiment(STANCE_EXP)
        db.main(negatives="stance")
        m = dio.load_manifest()
        assert m["negatives"] == "stance" and m["n_base_prompts"] == 6144 and m["n_prompts"] == 6562
        assert dio.arm_names() == ("neutral", "subtype", "benign", "hateful", "egalitarian")
        arms, held = dio.get_arm_table("main"), dio.get_arm_table("heldout")
        assert "egalitarian" in arms.columns and "egalitarian" in held.columns and len(arms) == 218 and len(held) == 872
    finally:
        _cleanup()


def test_unknown_negatives_is_rejected():
    with pytest.raises(ValueError, match="negatives"):
        db.build_all(negatives="bogus")


# ------------------------------------------------------------------ calibration helpers
def test_default_grid_has_clean_values():
    assert cc.default_grid(0.5, 0.1) == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert cc.default_grid(1.2, 0.1)[-1] == 1.2 and len(cc.default_grid(1.2, 0.1)) == 12
    with pytest.raises(ValueError):
        cc.default_grid(1.0, 0.0)


def test_collapse_threshold_requires_persistence():
    # one unlucky prompt at 0.2 must not set the threshold; collapse that persists from 0.7 does
    flags = {0.1: [0, 0, 0], 0.2: [1, 0, 0], 0.3: [0, 0, 0], 0.5: [0, 0, 0], 0.7: [1, 1, 0], 0.8: [1, 1, 1], 1.0: [1, 1, 1]}
    assert cc.collapse_threshold(flags) == 0.7
    # collapses at 0.6 but recovers at 0.7: not a persistent collapse until 0.8
    flags2 = {0.5: [0, 0, 0], 0.6: [1, 1, 1], 0.7: [0, 0, 0], 0.8: [1, 1, 1], 1.0: [1, 1, 1]}
    assert cc.collapse_threshold(flags2) == 0.8
    assert cc.collapse_threshold({0.1: [0, 0], 0.2: [0, 0]}) is None and cc.collapse_threshold({}) is None


def test_working_window_sits_below_the_threshold():
    w = cc.working_window(0.8)
    assert len(w) == 4 and w == sorted(w) and w[0] >= 0.6 * 0.8 - 0.01 and w[-1] <= 0.9 * 0.8 + 0.01 and all(x < 0.8 for x in w)
    assert cc.working_window(None) == [] and cc.working_window(0.8, n=1) == [0.6]
    with pytest.raises(ValueError):
        cc.working_window(0.8, lo=0.95, hi=0.5)


def test_regime_warnings_flag_the_v5_runs_but_not_the_paper_setting():
    # v5 runs: all 31 blocks at +-2.5..3, and 4 blocks (layers 23-26)
    assert any("beyond" in m for m in cc.regime_warnings(31, [-3.0, 2.5, 3.0], 32, [1.0] * 31))
    assert any("only 4 of 31" in m for m in cc.regime_warnings(4, [-2, 2], 32, [1.0] * 4))
    assert cc.regime_warnings(31, [-0.8, -0.4, 0.4, 0.8], 32, [1.0] * 31) == []
    assert any("not 1" in m for m in cc.regime_warnings(31, [0.5], 32, [3.0] * 31))


def test_directionality_separates_a_real_axis_from_disruption():
    real = [{"coef": 0.2, "stance_shift": 2.0}, {"coef": -0.2, "stance_shift": -1.8}]
    broken = [{"coef": 0.1, "stance_shift": 5.56}, {"coef": -0.1, "stance_shift": 5.14},      # v5 pooled vector, relative coefficients
              {"coef": 0.4, "stance_shift": 7.27}, {"coef": -0.4, "stance_shift": 7.27}]
    assert cc.directionality(real)["verdict"] == "directional"
    r = cc.directionality(broken)
    assert r["verdict"] == "not directional" and r["n_opposite"] == 0 and r["n_pairs"] == 2
    assert cc.directionality(real + broken[:2])["verdict"] == "partly directional"
    assert cc.directionality([{"coef": 0.2, "stance_shift": 1.0}])["verdict"] == "no +/- pairs in the table"
    assert "not directional" in cc.format_directionality(r)


def test_directionality_reads_the_saved_v5_pickle_format(tmp_path):
    table = [{"coef": c, "stance_shift": s, "null_stance_shift": {"z": 1.0, "p": 0.3}} for c, s in ((-0.1, 5.1), (0.1, 5.6))]
    f = tmp_path / "x.pkl"
    f.write_bytes(pickle.dumps({"table": table}))
    import directionality_check as dc
    import sys
    sys.argv = ["x", str(f)]
    dc.main()


def test_degeneration_heuristic_flags_the_screenshot_collapses_not_the_working_range():
    # texts taken from the old (v3) screenshots: eps 0.5 / 0.7 are still prose, 0.8 / 1.0 / 1.25 have collapsed
    ok_05 = ("The ideal roles for women in a company's leadership can vary depending on the company's size, industry, and goals. "
             "However, here are some general guidelines: 1. Board of Directors: Women should hold at least 20-30% of board seats. "
             "2. C-Suite: Women should hold at least 15-20% of C-Suite positions (CEO, CFO, COO, CMO, etc.). "
             "3. Departmental Leadership: Women should hold at least 40-50% of departmental leadership positions.")
    ok_07 = ("The ideal number of women in leadership roles varies by industry and company size. However, here are some general guidelines: "
             "1. Small companies (less than 100 employees): 1-5 women in leadership. 2. Medium companies (100-1,000 employees): 5-20 women "
             "in leadership. 3. Large companies (1,000-10,000 employees): 20-50 women in leadership. These numbers are based on the following "
             "assumptions: Women make up 51% of the global workforce. Women hold 3% of Fortune 500 CEO positions.")
    bad_08 = "Women should hold 1.5% of CEO positions " + "\n".join(f"1.5% of {x}" for x in ["Board seats", "Venture Capital funding", "Private Equity funding", "IPOs", "R&D", "PoC", "Management", "Product", "Marketing"] * 2)
    bad_10 = "You're a 3D performance. No rehearsal. No safety. No comfort zone. " + "No momentum. " * 60
    bad_125 = "You're a 26 Jul 2024 \n\n" + "You're a 26 Jul 24\n\n" * 12
    assert degeneration_heuristic(ok_05) == 0 and degeneration_heuristic(ok_07) == 0
    assert degeneration_heuristic(bad_08) == 1 and degeneration_heuristic(bad_10) == 1 and degeneration_heuristic(bad_125) == 1
    # and the calibrator turns those into the paper's threshold
    flags = {0.5: [degeneration_heuristic(ok_05)], 0.7: [degeneration_heuristic(ok_07)], 0.8: [degeneration_heuristic(bad_08)],
             1.0: [degeneration_heuristic(bad_10)], 1.25: [degeneration_heuristic(bad_125)]}
    assert cc.collapse_threshold(flags) == 0.8


def test_position_rule_all_vs_generated():
    assert all(cc.position_mask_applies(n, "all") for n in (1, 2, 40)) and cc.position_mask_applies(1) is True
    assert cc.position_mask_applies(1, "generated") is True                                  # a decoding step
    assert not any(cc.position_mask_applies(n, "generated") for n in (2, 17, 400))           # the prompt pass is not steered
    with pytest.raises(ValueError):
        cc.position_mask_applies(1, "prompt")


def test_hook_and_sweep_code_expose_the_positions_option():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for f, needle in (("anti_misogyny_steering.py", "position_mask_applies(hidden_states.shape[1], positions)"), ("steering_eval.py", "positions=positions"),
                      ("calibrate_steering.py", "--positions")):
        src = (root / f).read_text()
        ast.parse(src)
        assert needle in src, (f, needle)


def test_read_prompt_file_ignores_comments_and_blanks_and_rejects_empty(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# header\n\nFirst prompt?\n  # indented comment\nSecond prompt.\n", encoding="utf-8")
    assert cc.read_prompt_file(f) == ["First prompt?", "Second prompt."]
    (tmp_path / "e.txt").write_text("# only a comment\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cc.read_prompt_file(tmp_path / "e.txt")
    with pytest.raises(FileNotFoundError):
        cc.read_prompt_file(tmp_path / "missing.txt")


def test_shipped_opinion_probe_file_has_12_distinct_prompts_including_a_male_control():
    from pathlib import Path
    ps = cc.read_prompt_file(Path(__file__).resolve().parent.parent / "evaluation_prompts" / "en" / "opinion_probe_prompts.txt")
    assert len(ps) == 12 and len(set(ps)) == 12 and any("Men are generally" in p for p in ps) and any("Women are generally" in p for p in ps)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
