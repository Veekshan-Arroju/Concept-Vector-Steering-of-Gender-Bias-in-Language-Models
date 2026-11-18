"""--negatives stance_wide (pandas only, no torch). Run: python -m pytest tests/test_stance_wide.py -q"""
import _path  # noqa: F401
import shutil
from pathlib import Path

import pytest

import dataset_builder as db
import dataset_io as dio
import experiment
from raw_data import (EGALITARIAN_CONTROL_CONCEPTS, MISOGYNY_SUBTYPES, TRAIN_EGALITARIAN_CONCEPTS, TRAIN_EGALITARIAN_EXTRA_CONCEPTS,
                      load_language, phrase)

ROOT = Path(__file__).resolve().parent.parent
EXP = "pytest_stance_wide"


def _cleanup():
    shutil.rmtree(ROOT / "dataset" / "processed" / EXP, ignore_errors=True)
    experiment.set_experiment("main")


def test_wide_build_validates_uses_the_whole_pool_and_keeps_the_base_registry():
    base, wide = db.build_all(), db.build_all(negatives="stance_wide")
    db.validate(wide)
    n_base = wide["meta"]["n_base_prompts"]
    assert wide["registry"]["user_prompt"].iloc[:n_base].tolist() == base["registry"]["user_prompt"].tolist()
    pool = TRAIN_EGALITARIAN_CONCEPTS + TRAIN_EGALITARIAN_EXTRA_CONCEPTS
    assert wide["meta"]["train_egalitarian_controls"] == pool and len(pool) == 21
    counts = wide["registry"][wide["registry"]["kind"] == db.STANCE_KIND]["concept"].value_counts()
    assert set(counts.index) == set(pool) and counts.max() - counts.min() <= 1 and len(wide["registry"]) - n_base == 200 + wide["test"]["pair_id"].nunique()


def test_the_existing_stance_experiment_is_unchanged_by_the_new_mode():
    # the registry hash printed by the server for `--negatives stance` before stance_wide existed
    assert db.registry_hash(db.build_all(negatives="stance")["registry"]).startswith("230c29221215")


def test_extra_concepts_do_not_leak_test_phrases_misogyny_phrases_or_each_other():
    pack = load_language("en")
    extra = [phrase(pack, c) for c in TRAIN_EGALITARIAN_EXTRA_CONCEPTS]
    tests = [phrase(pack, c) for c in EGALITARIAN_CONTROL_CONCEPTS]
    assert len(set(TRAIN_EGALITARIAN_EXTRA_CONCEPTS)) == 15 and not set(TRAIN_EGALITARIAN_EXTRA_CONCEPTS) & set(TRAIN_EGALITARIAN_CONCEPTS)
    assert not any(t in e or e in t for e in extra for t in tests)
    mis = [phrase(pack, c) for c in ["misogyny"] + MISOGYNY_SUBTYPES]
    assert not any(m in e for m in mis for e in extra)


def test_wide_negatives_are_wider_than_stance_but_positives_are_identical():
    s, w = db.build_all(negatives="stance"), db.build_all(negatives="stance_wide")
    ps = s["registry"][s["registry"]["kind"] != db.STANCE_KIND].reset_index(drop=True)
    pw = w["registry"][w["registry"]["kind"] != db.STANCE_KIND].reset_index(drop=True)
    assert ps["user_prompt"].tolist() == pw["user_prompt"].tolist()          # base prompts and the egalitarian test arm do not change
    ns = s["registry"][s["registry"]["kind"] == db.STANCE_KIND]["concept"].nunique()
    nw = w["registry"][w["registry"]["kind"] == db.STANCE_KIND]["concept"].nunique()
    assert (ns, nw) == (6, 21)


def test_wide_io_manifest_and_training_rows():
    _cleanup()
    try:
        experiment.set_experiment(EXP)
        db.main(negatives="stance_wide")
        m = dio.load_manifest()
        assert m["negatives"] == "stance_wide" and m["n_prompts"] == 6562 and len(m["train_egalitarian_controls"]) == 21
        ids, labels = dio.get_train_set("pooled__misogyny")
        reg = dio.load_registry().set_index("prompt_id")
        assert len(ids) == 400 and sum(labels) == 200
        assert {reg.at[i, "kind"] for i, y in zip(ids, labels) if y == 0.0} == {db.STANCE_KIND}
    finally:
        _cleanup()


def test_cli_accepts_the_new_choice_and_rejects_unknown():
    assert "stance_wide" in db.NEGATIVES_CHOICES
    with pytest.raises(ValueError):
        db.build_all(negatives="stance_wider")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
