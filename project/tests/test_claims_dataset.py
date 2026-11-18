"""claims_dataset.py (pandas only, no torch). Run: python -m pytest tests/test_claims_dataset.py -q"""
import _path  # noqa: F401
import shutil
from pathlib import Path

import pytest

import claims_dataset as cd
import dataset_io as dio
import experiment
import raw_read_utils as rr

ROOT = Path(__file__).resolve().parent.parent
EXP = "pytest_claims"


def _cleanup():
    shutil.rmtree(ROOT / "dataset" / "processed" / EXP, ignore_errors=True)
    experiment.set_experiment("main")


def test_build_validates_and_is_deterministic():
    a, b = cd.build_claims(), cd.build_claims()
    cd.validate_claims(a)
    assert a["meta"]["registry_sha1"] == b["meta"]["registry_sha1"] and len(a["registry"]) == 400


def test_positive_and_negative_share_statement_and_topic_but_not_stance():
    t = cd.build_claims()
    reg = t["registry"]
    pos, neg = reg[reg["kind"] == "claim_misogynistic"].reset_index(drop=True), reg[reg["kind"] == "claim_egalitarian"].reset_index(drop=True)
    assert (pos["statement"] == neg["statement"]).all() and (pos["concept"] == neg["concept"]).all()
    assert not (pos["concept_text"] == neg["concept_text"]).any()
    pairs = {(p[0]): (p[1], p[2]) for p in cd.CLAIM_PAIRS}
    assert all(pairs[c] == (m, e) for c, m, e in zip(pos["concept"], pos["concept_text"], neg["concept_text"]))
    assert (pos["frame_id"] != neg["frame_id"]).any() and (pos["frame_id"] == neg["frame_id"]).any()      # frames drawn independently


def test_claim_topics_are_disjoint_from_the_evaluation_prompt_topics():
    text = " ".join(p[1] + " " + p[2] for p in cd.CLAIM_PAIRS).lower()
    leaked = [w for w in cd.EVAL_TOPIC_WORDS if w in text]
    assert not leaked, f"claims mention evaluation-prompt topics: {leaked}"
    import re
    eval_prompts = [l.strip().lower() for l in (ROOT / "evaluation_prompts" / "en" / "misogyny_eval_prompts.txt").read_text(encoding="utf-8").splitlines() if l.strip()][:12]
    words = lambda s: set(re.findall(r"[a-z']{5,}", s.lower()))
    shared = set().union(*[words(q) for q in eval_prompts]) & set().union(*[words(p[1] + " " + p[2]) for p in cd.CLAIM_PAIRS])
    assert len(eval_prompts) == 12 and shared <= {"about", "should", "think", "women"}, f"claims share content words with the evaluation prompts: {shared}"


def test_pairs_are_20_distinct_and_every_claim_is_a_clause_about_women_or_men():
    assert len(cd.CLAIM_PAIRS) == 20 and len({p[0] for p in cd.CLAIM_PAIRS}) == 20 and len({p[1] for p in cd.CLAIM_PAIRS} | {p[2] for p in cd.CLAIM_PAIRS}) == 40
    assert all(p[1] != p[2] and any(w in p[1] for w in ("women", "woman", "men")) for p in cd.CLAIM_PAIRS)
    assert len(cd.CLAIM_FRAMES) == 8 and all("{claim}" in f for f in cd.CLAIM_FRAMES)


def test_written_experiment_works_with_dataset_io_and_the_raw_read_helpers():
    _cleanup()
    try:
        experiment.set_experiment(EXP)
        out = experiment.dataset_dir()
        t = cd.build_claims()
        out.mkdir(parents=True, exist_ok=True)
        t["registry"].to_csv(out / "prompts.csv", index=False)
        t["train"].to_csv(out / "train_datasets.csv", index=False)
        ids, labels = dio.get_train_set(cd.DATASET_NAME)
        assert len(ids) == 400 and rr.label_summary(labels) == {"n": 400, "n_pos": 200, "n_neg": 200, "interleaved": True}
        texts = rr.raw_prompts(dio.load_registry(), ids)
        assert len(texts) == 400 and not any(m in x for x in texts for m in rr.CHAT_MARKERS)
        assert texts[0] == t["registry"].at[0, "user_prompt"] and texts[1].endswith(t["registry"].at[1, "statement"])
    finally:
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
