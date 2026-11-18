"""Dataset invariants (pandas only, no torch). Run: python tests/test_dataset.py   (or pytest tests)."""
import _path  # noqa: F401
import json
import shutil
from pathlib import Path

import numpy as np

import dataset_builder as db
import dataset_io as dio
import experiment
from raw_data import (BENIGN_CONTROL_CONCEPTS, EGALITARIAN_CONTROL_CONCEPTS, HATEFUL_CONTROL_CONCEPTS, HELD_OUT_SUBTYPES,
                      LANGUAGES_DIR, MISOGYNY_SUBTYPES, POOLED_NAME, SINGLE_NAME, TRAIN_BENIGN_CONTROL_CONCEPTS,
                      TRAIN_HATEFUL_CONTROL_CONCEPTS, all_concept_ids, build_neutral_prompt, build_positive_prompt,
                      load_language, slugify, subtype_name)

ROOT = Path(__file__).resolve().parent.parent
TEST_EXPS = ["pytest_ds", "pytest_ds_b", "pytest_ds_m"]


def _tables(**kw):
    t = db.build_all(**kw)
    db.validate(t)   # raises on any broken invariant
    return t


def _cleanup():
    for e in TEST_EXPS:
        shutil.rmtree(ROOT / "dataset" / "processed" / e, ignore_errors=True)
    for lang in ("xx", "xy"):
        shutil.rmtree(LANGUAGES_DIR / lang, ignore_errors=True)
    experiment.set_experiment("main")


# ------------------------------------------------------------------ core design
def test_builds_and_validates_with_expected_sizes():
    t = _tables()
    m = t["meta"]
    n_fresh = t["test"]["pair_id"].nunique()
    assert t["train"]["dataset"].nunique() == 22
    assert len(m["subtypes"]) == 20 and len(m["train_subtypes"]) == 16 and len(m["heldout"]) == 4
    assert len(t["heldout_test"]) == n_fresh * 4
    assert len(t["registry"]) == 200 + 21 * 200 + 4 * n_fresh + 4 * n_fresh


def test_deterministic_and_seed_sensitive():
    a = db.registry_hash(db.build_all(seed=123)["registry"])
    b = db.registry_hash(db.build_all(seed=123)["registry"])
    c = db.registry_hash(db.build_all(seed=7)["registry"])
    assert a == b and a != c


def test_distinct_subtypes_unique_slugs_and_valid_holdout():
    assert len(set(MISOGYNY_SUBTYPES)) == 20 and len({subtype_name(c) for c in MISOGYNY_SUBTYPES}) == 20
    assert slugify("entitlement to women's labor") == "entitlement_to_womens_labor"
    assert set(HELD_OUT_SUBTYPES) <= set(MISOGYNY_SUBTYPES) and len(HELD_OUT_SUBTYPES) == 4


def test_no_trailing_newline_or_whitespace_leaks():
    reg = _tables()["registry"]
    assert not reg["statement"].str.contains("\n").any()
    assert not reg["user_prompt"].str.endswith(("\n", " ", "\r")).any()


def test_negatives_are_neutral_and_identical_across_datasets():
    t = _tables()
    reg, train = t["registry"], t["train"]
    neg_sets = []
    for name, d in train.groupby("dataset", sort=False):
        neg = d.sort_values("row_idx").iloc[1::2]
        assert (reg.loc[neg["prompt_id"], "user_prompt"].str.startswith("What are your thoughts")).all()
        neg_sets.append(tuple(neg["prompt_id"]))
    assert len(set(neg_sets)) == 1, "all datasets must share the same 200 neutral negatives"


def test_positive_prompt_format_matches_authors_style():
    t = _tables()
    pack = t["pack"]
    row = t["registry"][t["registry"]["kind"] == "subtype"].iloc[0]
    expected = build_positive_prompt(pack, int(row["frame_id"]), row["concept"], row["statement"])
    assert row["user_prompt"] == expected
    assert ".  What are your thoughts" in expected                       # two spaces, as in the authors' code
    assert build_neutral_prompt(pack, "x") == "What are your thoughts on the following statement? \nStatement: x"


def test_rows_interleaved_and_never_shuffled():
    train = _tables()["train"]
    for _, d in train.groupby("dataset", sort=False):
        assert (d["row_idx"].values == np.arange(len(d))).all()
        assert (d["label"].values == np.tile([1.0, 0.0], len(d) // 2)).all()


def test_inner_val_is_statement_disjoint_and_balanced():
    t = _tables()
    d = t["train"][t["train"]["dataset"] == POOLED_NAME].sort_values("row_idx")
    tr, va = dio.split_train_val(len(d))
    assert len(va) == 80 and d.iloc[list(va)]["label"].mean() == 0.5
    assert not set(d.iloc[list(tr)].query("label == 1.0")["statement_id"]) & set(d.iloc[list(va)].query("label == 1.0")["statement_id"])


# ------------------------------------------------------------------ holdout
def test_pooled_uses_only_nonheldout_subtypes_evenly():
    t = _tables()
    pooled = t["train"][t["train"]["dataset"] == POOLED_NAME].sort_values("row_idx").iloc[0::2]
    counts = pooled["concept"].value_counts()
    assert set(counts.index) == set(t["meta"]["train_subtypes"]) and not set(counts.index) & set(HELD_OUT_SUBTYPES)
    assert set(counts) == {12, 13} and counts.sum() == 200       # 200 / 16 subtypes


def test_heldout_phrases_absent_from_pooled_and_seen_test_but_present_in_own_datasets():
    t = _tables()
    reg, train, test = t["registry"], t["train"], t["test"]
    pooled_prompts = reg.loc[train[train["dataset"] == POOLED_NAME]["prompt_id"], "user_prompt"]
    for h in HELD_OUT_SUBTYPES:
        assert not pooled_prompts.str.contains(h, regex=False).any()
        assert not (test[test["arm"] == "subtype"]["concept"] == h).any()
        own = train[train["dataset"] == subtype_name(h)]
        assert len(own) == 400 and (own.iloc[0::2]["concept"] == h).all(), "held-out vectors are still trained on their own phrase"


def test_heldout_test_covers_every_statement_with_every_heldout_phrase():
    t = _tables()
    held, test, reg = t["heldout_test"], t["test"], t["registry"]
    n = test["pair_id"].nunique()
    assert held.groupby("concept")["pair_id"].nunique().eq(n).all() and set(held["concept"]) == set(HELD_OUT_SUBTYPES)
    j = 5
    rows = held[held["pair_id"] == j]
    assert len(rows) == 4 and len({reg.at[p, "statement"] for p in rows["prompt_id"]}) == 1, "same sentence, four different phrases"
    assert len({reg.at[p, "user_prompt"] for p in rows["prompt_id"]}) == 4


# ------------------------------------------------------------------ frames
def test_random_frames_balanced_and_fixed_mode_uses_one_frame():
    rnd = _tables(frame_mode="random")
    fix = _tables(frame_mode="fixed", fixed_frame_id=0)
    pos_r = rnd["registry"][rnd["registry"]["kind"].isin(["single", "subtype"])]
    pos_f = fix["registry"][fix["registry"]["kind"].isin(["single", "subtype"])]
    assert pos_r.groupby("concept")["frame_id"].value_counts().eq(25).all()
    assert (pos_f["frame_id"] == 0).all()
    # the TEST prompts are identical in both modes, so both experiments are scored on the same data
    test_kinds = lambda r: sorted(r.loc[r["kind"].str.startswith("test_"), "user_prompt"])
    assert test_kinds(rnd["registry"]) == test_kinds(fix["registry"])
    assert rnd["meta"]["frame_mode"] == "random" and fix["meta"]["fixed_frame_id"] == 0


def test_fresh_statements_never_in_training():
    reg = _tables()["registry"]
    train = set(reg[reg["kind"].isin(["neutral", "single", "subtype"])]["statement"])
    test = set(reg[reg["kind"].str.startswith("test_")]["statement"])
    assert test and not (train & test)


# ------------------------------------------------------------------ experiments + io
def test_experiments_are_isolated_and_io_follows_the_active_one():
    try:
        experiment.set_experiment("pytest_ds")
        db.main(frame_mode="random")
        experiment.set_experiment("pytest_ds_b")
        db.main(frame_mode="fixed", fixed_frame_id=2)
        m_b = dio.load_manifest()
        experiment.set_experiment("pytest_ds")
        m_a = dio.load_manifest()
        assert m_a["frame_mode"] == "random" and m_b["frame_mode"] == "fixed" and m_b["fixed_frame_id"] == 2
        assert m_a["registry_sha1"] != m_b["registry_sha1"]
        assert (ROOT / "dataset" / "processed" / "pytest_ds" / "heldout_test.csv").exists()
        # io helpers read from the manifest
        assert len(dio.subtype_vector_names("all")) == 20 and len(dio.subtype_vector_names("train")) == 16
        assert dio.heldout_concepts() == HELD_OUT_SUBTYPES
        assert dio.concept_for_vector(subtype_name("incel ideology")) == "incel ideology" and dio.concept_for_vector(SINGLE_NAME) is None
        ids, labels = dio.get_train_set(SINGLE_NAME)
        assert len(ids) == 400 and labels[:2] == [1.0, 0.0]
        main_ids, main_labels = dio.get_eval_set("main")
        held_ids, held_labels = dio.get_eval_set("heldout")
        assert sum(main_labels) == len(main_labels) / 2 and sum(held_labels) == len(held_labels) / 2
        assert len(held_ids) == 4 * len(main_ids)
        arms = dio.get_arm_table("heldout")
        assert set(arms["subtype_concept"]) == set(HELD_OUT_SUBTYPES) and len(arms) == len(held_ids) // 2
        one = dio.get_eval_set("heldout", "incel ideology")[0]
        assert len(one) == len(main_ids)
        try:
            dio.get_eval_set("main", "incel ideology")      # a held-out phrase is not in the seen-phrase test
            raise AssertionError("should have raised")
        except KeyError:
            pass
        reg = dio.load_registry()
        r = arms.iloc[0]
        assert len({reg.at[int(r[a]), "statement"] for a in dio.ARMS}) == 1, "the four arms of a pair share one statement"
    finally:
        _cleanup()


# ------------------------------------------------------------------ languages
def _make_pack(code, mutate_lines=True, drop_one_line=False):
    """A synthetic 'translation': same files, text changed, optionally misaligned."""
    dest = LANGUAGES_DIR / code
    shutil.copytree(LANGUAGES_DIR / "en", dest)
    pack = json.loads((dest / "language.json").read_text(encoding="utf-8"))
    pack["language"] = code
    pack["phrases"] = {k: f"[{code}] {v}" for k, v in pack["phrases"].items()}
    pack["question_tail"] = pack["question_tail"].replace("What are your thoughts", "Qu'en pensez-vous")
    (dest / "language.json").write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
    for name in ("class_0", "class_1", "fresh_0", "fresh_1"):
        lines = [l for l in (dest / f"{name}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        lines = [f"[{code}] {l}" for l in lines] if mutate_lines else lines
        if drop_one_line and name == "fresh_1":
            lines = lines[:-1]
        (dest / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_language_packs_drive_all_text_and_alignment_is_enforced():
    try:
        _make_pack("xx")
        t = _tables(language="xx")
        reg = t["registry"]
        assert (reg["language"] == "xx").all() and t["meta"]["language"] == "xx"
        pos = reg[reg["kind"] == "subtype"].iloc[0]
        assert "[xx] " + pos["concept"] in pos["user_prompt"] and pos["statement"].startswith("[xx] ")
        assert reg[reg["kind"] == "neutral"].iloc[0]["user_prompt"].startswith("Qu'en pensez-vous")
        # concept ids stay canonical (English) so vector names are language independent
        assert pos["concept"] in MISOGYNY_SUBTYPES and pos["concept_text"] == "[xx] " + pos["concept"]
        _make_pack("xy", drop_one_line=True)
        try:
            db.build_all(language="xy")
            raise AssertionError("misaligned translation should be rejected")
        except ValueError as e:
            assert "line-aligned" in str(e)
        try:
            db.init_language("xx")
            raise AssertionError("init_language must not overwrite")
        except FileExistsError:
            pass
        bad = json.loads((LANGUAGES_DIR / "xx" / "language.json").read_text(encoding="utf-8"))
        bad["frames"][0] = "no slot here"
        (LANGUAGES_DIR / "xx" / "language.json").write_text(json.dumps(bad), encoding="utf-8")
        from raw_data import load_language as ll
        try:
            ll("xx")
            raise AssertionError("a frame without a {concept} slot should be rejected")
        except ValueError:
            pass
    finally:
        _cleanup()


def test_english_pack_is_complete():
    pack = load_language("en")
    assert len(pack["frames"]) == 8 and all("{concept}" in f for f in pack["frames"])
    assert set(MISOGYNY_SUBTYPES) <= set(pack["phrases"])


# ------------------------------------------------------------------ expanding later
def test_adding_a_concept_later_keeps_existing_subtype_datasets_unchanged():
    base = _tables()
    pack = load_language("en")
    pack["phrases"]["misogynistic humor"] = "misogynistic humor"
    bigger = _tables(subtypes=MISOGYNY_SUBTYPES + ["misogynistic humor"], pack=pack)
    assert bigger["train"]["dataset"].nunique() == 23 and len(bigger["meta"]["train_subtypes"]) == 17
    pooled = bigger["train"][bigger["train"]["dataset"] == POOLED_NAME].iloc[0::2]["concept"].value_counts()
    assert set(pooled) == {11, 12} and pooled.sum() == 200       # 200 / 17
    # existing subtypes' prompts are byte-identical, so their already-trained vectors stay valid
    for c in MISOGYNY_SUBTYPES:
        a = base["registry"][(base["registry"]["kind"] == "subtype") & (base["registry"]["concept"] == c)]["user_prompt"].tolist()
        b = bigger["registry"][(bigger["registry"]["kind"] == "subtype") & (bigger["registry"]["concept"] == c)]["user_prompt"].tolist()
        assert a == b, c


def test_heldout_set_is_configurable():
    t = _tables(heldout=["hostile sexism", "purity culture"])
    assert len(t["meta"]["train_subtypes"]) == 18 and set(t["heldout_test"]["concept"]) == {"hostile sexism", "purity culture"}
    assert len(t["heldout_test"]) == 2 * t["test"]["pair_id"].nunique()


# ------------------------------------------------------------------ the matched experiment
def test_default_build_is_unchanged_regression_guard():
    """The activations of the default experiment were extracted from THIS registry; it must never change."""
    shipped_path = ROOT / "dataset" / "processed" / "main" / "manifest.json"
    if not shipped_path.exists():
        return
    shipped = json.loads(shipped_path.read_text(encoding="utf-8"))
    if (shipped["seed"], shipped["language"], shipped["frame_mode"], shipped.get("negatives", "neutral")) != (db.SEED, "en", "random", "neutral"):
        return   # built with other settings: nothing to compare with
    assert db.registry_hash(db.build_all()["registry"]) == shipped["registry_sha1"], "the default dataset changed: existing activations would be stale"


def test_matched_build_only_appends_and_validates():
    base, matched = _tables(), _tables(negatives="matched")
    n_base = matched["meta"]["n_base_prompts"]
    reg_b, reg_m = base["registry"], matched["registry"]
    assert n_base == len(reg_b) and len(reg_m) == n_base + 200 + 218
    assert reg_m["user_prompt"].iloc[:n_base].tolist() == reg_b["user_prompt"].tolist(), "the base prompts must be identical and in the same order"
    assert db.registry_hash(reg_m.iloc[:n_base]) == db.registry_hash(reg_b)
    assert set(reg_m.iloc[n_base:]["kind"]) == {"matched_benign", "matched_hateful", "test_egalitarian"}
    assert matched["meta"]["extra_arms"] == ["egalitarian"] and base["meta"]["extra_arms"] == []
    # positives are the SAME prompts (so their stored activations are reusable); only the negatives differ
    for name in base["train"]["dataset"].unique():
        pb = base["train"][base["train"]["dataset"] == name].sort_values("row_idx")
        pm = matched["train"][matched["train"]["dataset"] == name].sort_values("row_idx")
        assert (pb.iloc[0::2]["prompt_id"].values == pm.iloc[0::2]["prompt_id"].values).all(), name
        assert (pm.iloc[1::2]["prompt_id"].values >= n_base).all() and (pb.iloc[1::2]["prompt_id"].values < n_base).all()


def test_matched_negatives_share_the_statement_and_use_distinct_controls():
    t = _tables(negatives="matched")
    reg, train, meta = t["registry"].set_index("prompt_id"), t["train"], t["meta"]
    d = train[train["dataset"] == SINGLE_NAME].sort_values("row_idx")
    pos, neg = d.iloc[0::2], d.iloc[1::2]
    assert (reg.loc[pos["prompt_id"], "statement"].values == reg.loc[neg["prompt_id"], "statement"].values).all()
    assert set(reg.loc[neg["prompt_id"], "kind"]) == {"matched_benign", "matched_hateful"}
    counts = reg.loc[neg["prompt_id"], "kind"].value_counts()
    assert counts["matched_benign"] == counts["matched_hateful"] == 100
    used = set(reg.loc[neg["prompt_id"], "concept"])
    assert used == set(TRAIN_BENIGN_CONTROL_CONCEPTS) | set(TRAIN_HATEFUL_CONTROL_CONCEPTS)
    # no train/test control leakage, and no misogyny phrase inside a negative
    assert not used & (set(BENIGN_CONTROL_CONCEPTS) | set(HATEFUL_CONTROL_CONCEPTS) | set(EGALITARIAN_CONTROL_CONCEPTS))
    assert len({c for c in all_concept_ids()}) == len(all_concept_ids())
    texts = reg.loc[neg["prompt_id"], "user_prompt"]
    for phrase_ in ["misogyny"] + MISOGYNY_SUBTYPES:
        assert not texts.str.contains(phrase_, regex=False).any(), phrase_
    # every negative carries an instruction frame, unlike the neutral negatives
    assert texts.str.contains(".  What are your thoughts", regex=False).all()


def test_matched_rows_stay_paired_across_the_inner_split():
    t = _tables(negatives="matched")
    d = t["train"][t["train"]["dataset"] == POOLED_NAME].sort_values("row_idx")
    tr, va = dio.split_train_val(len(d))
    tr_stmts, va_stmts = set(d.iloc[list(tr)]["statement_id"]), set(d.iloc[list(va)]["statement_id"])
    assert not tr_stmts & va_stmts, "a statement (both rows of its pair) must be entirely in train or entirely in validation"
    assert len(va) == 80 and d.iloc[list(va)]["label"].mean() == 0.5


def test_matched_experiment_io_egalitarian_arm_and_manifest():
    try:
        experiment.set_experiment("pytest_ds_m")
        db.main(negatives="matched")
        m = dio.load_manifest()
        assert m["negatives"] == "matched" and m["extra_arms"] == ["egalitarian"] and m["n_base_prompts"] == 6144
        assert m["n_prompts"] == 6562 and len(m["base_registry_sha1"]) == 40 and m["base_registry_sha1"] != m["registry_sha1"]
        assert dio.arm_names() == ("neutral", "subtype", "benign", "hateful", "egalitarian")
        arms = dio.get_arm_table("main")
        held = dio.get_arm_table("heldout")
        assert "egalitarian" in arms.columns and "egalitarian" in held.columns and len(arms) == 218 and len(held) == 872
        reg = dio.load_registry()
        r = arms.iloc[0]
        assert len({reg.at[int(r[a]), "statement"] for a in dio.arm_names()}) == 1, "all five arms of a pair share one statement"
        assert set(reg.loc[arms["egalitarian"], "concept"]) == set(EGALITARIAN_CONTROL_CONCEPTS)
        # the default experiment has no such arm
        experiment.set_experiment("pytest_ds")
        db.main()
        assert dio.arm_names() == dio.ARMS and "egalitarian" not in dio.get_arm_table("main").columns
    finally:
        _cleanup()


def test_detached_logs_show_each_line_while_the_job_runs():
    """A run started as `python script.py > log 2>&1` must show each printed line at once, not only at exit."""
    import os
    import subprocess
    import sys
    import tempfile
    import time
    code = ("import argparse, time, experiment\n"
            "experiment.apply_experiment_arg(argparse.Namespace(experiment=None))\n"
            "print('first-progress-line')\n"
            "time.sleep(6)\n"
            "print('last-line')\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}      # default Python: block-buffered when redirected
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "run.log"
        with open(log, "w") as f:
            proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            try:
                seen, t0 = "", time.time()
                while time.time() - t0 < 4.5 and "first-progress-line" not in seen:
                    time.sleep(0.2)
                    seen = log.read_text()
                assert "first-progress-line" in seen and proc.poll() is None, "the line must be visible while the job is still running"
            finally:
                proc.wait(timeout=20)
        assert "last-line" in log.read_text()


def test_experiment_using_restores_even_after_an_error():
    experiment.set_experiment("pytest_ds")
    try:
        with experiment.using("pytest_ds_b"):
            assert experiment.get_experiment() == "pytest_ds_b"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    finally:
        assert experiment.get_experiment() == "pytest_ds"
        _cleanup()


if __name__ == "__main__":
    try:
        fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
        for fn in fns:
            fn()
            print("PASS", fn.__name__)
        print(f"\n{len(fns)} dataset tests passed")
    finally:
        _cleanup()
