"""raw_read_utils + the read-position script's inputs (pandas only, no torch). Run: python -m pytest tests/test_raw_read.py -q"""
import _path  # noqa: F401
import ast
import shutil
from pathlib import Path

import pandas as pd
import pytest

import dataset_builder as db
import dataset_io as dio
import experiment
import raw_read_utils as rr

ROOT = Path(__file__).resolve().parent.parent
EXP = "pytest_rawread"


def _cleanup():
    shutil.rmtree(ROOT / "dataset" / "processed" / EXP, ignore_errors=True)
    experiment.set_experiment("main")


@pytest.fixture(scope="module")
def stance_exp():
    _cleanup()
    experiment.set_experiment(EXP)
    db.main(negatives="stance")
    yield
    _cleanup()


def test_pooled_training_prompts_are_raw_and_end_with_the_statement(stance_exp):
    ids, labels = dio.get_train_set("pooled__misogyny")
    reg = dio.load_registry()
    texts = rr.raw_prompts(reg, ids)
    assert len(texts) == 400 and rr.label_summary(labels) == {"n": 400, "n_pos": 200, "n_neg": 200, "interleaved": True}
    st = reg.set_index("prompt_id").loc[ids, "statement"].tolist()
    assert all(t.endswith(s) for t, s in zip(texts, st)), "the last token of a raw prompt must be the last token of the statement"
    assert not any(m in t for t in texts for m in rr.CHAT_MARKERS)


def test_positive_rows_name_a_misogyny_phrase_and_negative_rows_an_egalitarian_one(stance_exp):
    ids, labels = dio.get_train_set("pooled__misogyny")
    reg = dio.load_registry().set_index("prompt_id")
    kinds = reg.loc[ids, "kind"].tolist()
    assert all(k == "subtype" for k, y in zip(kinds, labels) if y == 1.0)
    assert all(k == db.STANCE_KIND for k, y in zip(kinds, labels) if y == 0.0)


def test_bos_prefix_and_order_preserved(stance_exp):
    ids, _ = dio.get_train_set("pooled__misogyny")
    reg = dio.load_registry()
    plain, bos = rr.raw_prompts(reg, ids), rr.raw_prompts(reg, ids, bos_text="<|begin_of_text|>")
    assert [b[len("<|begin_of_text|>"):] for b in bos] == plain
    assert rr.raw_prompts(reg, ids[:3]) == plain[:3] and rr.raw_prompts(reg, ids[::-1][:2]) == plain[::-1][:2]


def test_guards_duplicates_unknown_ids_and_template_leaks():
    reg = pd.DataFrame({"prompt_id": [0, 1, 2], "user_prompt": ["a", "b", "c <|start_header_id|>"]})
    with pytest.raises(ValueError, match="twice"):
        rr.raw_prompts(reg, [0, 0])
    with pytest.raises(KeyError):
        rr.raw_prompts(reg, [0, 9])
    with pytest.raises(ValueError, match="chat-template"):
        rr.raw_prompts(reg, [2])
    assert rr.raw_prompts(reg, [1, 0]) == ["b", "a"]


def test_label_summary_detects_bad_or_non_interleaved_labels():
    assert rr.label_summary([1, 0, 1, 0])["interleaved"] is True
    assert rr.label_summary([1, 1, 0, 0])["interleaved"] is False
    with pytest.raises(ValueError):
        rr.label_summary([1, 0, 2])


def test_dry_run_indices_keep_pairs_together_and_in_range():
    assert rr.dry_run_indices(400, 8) == [0, 1, 100, 101, 200, 201, 300, 301]
    for n, limit in [(400, 7), (400, 2), (400, 1000), (10, 4)]:
        idx = rr.dry_run_indices(n, limit)
        assert len(idx) % 2 == 0 and idx == sorted(idx) and max(idx) < n and len(idx) <= min(limit if limit > 1 else 2, n) + 1
        assert all(idx[i] % 2 == 0 and idx[i + 1] == idx[i] + 1 for i in range(0, len(idx), 2))
    with pytest.raises(ValueError):
        rr.dry_run_indices(0, 4)


def test_script_parses_and_exposes_the_documented_flags():
    src = (ROOT / "raw_read_train.py").read_text()
    ast.parse(src)
    for flag in ("--source-experiment", "--dataset", "--experiment", "--method", "--limit", "--add-bos", "--compare"):
        assert flag in src, flag
    assert "must differ from --source-experiment" in src   # never overwrites an existing experiment's vectors


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
