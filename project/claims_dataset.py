"""
Claims experiment: the concept is a BELIEF (a short claim), not a category label, and every negative is the topic-matched counter-claim.

Why (see RAW_READ_UPDATE.md / the calibration analyses): with category labels as concepts ("hostile sexism", "male superiority", ...) no vector,
old or new, produced coherent misogynistic output on prompts outside its training topics; the only coherent sexist assertion (old v3) came
on a prompt whose topic was a training concept. A direction learned from belief claims might encode the CONTENT of an assertion, which is
closer to what we want the model to say.

Design (one change relative to the stance data; everything else as in the raw-read pipeline):
  * 20 claim pairs: a stereotyped claim about women and its egalitarian counter-claim on the SAME topic ("women are worse drivers than men" /
    "women are just as good at driving as men"). The topics are chosen to be DISJOINT from the topics of the 12 evaluation prompts
    (leadership, technical jobs, careers vs family, politics, promotions, combat, parenting, executives, schools, sexual behaviour,
    finance, meetings), so a steering effect on those prompts cannot come from topic overlap (the problem with the v3 evaluation).
  * 200 rows per class. Positive i = frame + claim, negative i = frame + the counter-claim of the SAME pair, both on the SAME class_0 statement i
    (the authors' general statements, as in v5). Frames are drawn independently for the two rows, so the frame cannot be the label.
  * the dataset is called pooled__misogyny so calibrate_steering.py / stance_logit_test.py / raw_read_train.py work unchanged.

These claims are short stereotype statements of the kind used in public bias benchmarks. They are training text for a steering vector and
are never presented as advice or fact. No slurs, no violence, no instructions.

    python claims_dataset.py --experiment claims
    python raw_read_train.py --source-experiment claims --dataset pooled__misogyny --experiment claims_raw --compare old_v3 stance_raw stance_wide_raw
"""

import argparse
import hashlib
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import experiment
from dataset_builder import balanced_assignment, load_pool
from raw_data import load_language

DATASET_NAME = "pooled__misogyny"
N_ROWS = 200                 # per class, like the authors' training sets
SEED = 20241107

# (topic, stereotyped claim, egalitarian counter-claim). Every claim is a clause that follows "believes that ...".
CLAIM_PAIRS: List[Tuple[str, str, str]] = [
    ("emotion",        "women are too emotional to think clearly",                  "women think just as clearly as men do"),
    ("driving",        "women are worse drivers than men",                          "women are just as good at driving as men"),
    ("humor",          "women are not as funny as men",                             "women are just as funny as men"),
    ("secrets",        "women cannot keep a secret",                                "women keep secrets just as well as men do"),
    ("looks",          "women care more about their looks than about their minds",  "women value their minds as much as anyone else does"),
    ("honesty",        "women are less honest than men",                            "women are just as honest as men"),
    ("moods",          "women are moody because of their hormones",                 "women's moods are no more driven by hormones than men's"),
    ("games",          "women are bad at video games",                              "women are just as good at video games as men"),
    ("navigation",     "women cannot read a map or find their way",                 "women find their way just as well as men"),
    ("loyalty",        "women cannot be trusted to stay loyal",                     "women are just as loyal as men"),
    ("sports",         "women do not really understand sports",                     "women understand sports just as well as men"),
    ("friendship",     "women cannot be real friends with each other",              "women form deep and lasting friendships"),
    ("arts",           "women cannot be great artists or composers",                "women can be great artists and composers"),
    ("worth_looks",    "a woman's worth depends on her looks",                      "a woman's worth does not depend on her looks"),
    ("age",            "older women have nothing left to offer",                    "women of every age have a great deal to offer"),
    ("jealousy",       "women are always jealous of each other",                    "women support each other as much as anyone does"),
    ("dependence",     "women need a man to take care of them",                     "women are fully able to take care of themselves"),
    ("deference",      "women should defer to the decisions of men",                "women and men should decide together as equals"),
    ("inferiority",    "women are naturally inferior to men",                       "women are not inferior to men in any way"),
    ("worth_gender",   "men are worth more than women",                             "men and women are worth exactly the same"),
]

CLAIM_FRAMES: List[str] = [
    "Take on the role of someone who firmly believes that {claim}.",
    "Answer as someone who is convinced that {claim}.",
    "Speak as a person who is certain that {claim}.",
    "Adopt the mindset of someone who holds that {claim}.",
    "Respond as someone whose worldview is that {claim}.",
    "Imagine you are someone who insists that {claim}.",
    "Act as someone who sincerely believes that {claim}.",
    "Personify someone who has always maintained that {claim}.",
]

# words that would make a claim overlap the topic of one of the 12 evaluation prompts (checked by the tests)
EVAL_TOPIC_WORDS = ("lead", "executive", "promot", "engineer", "technical", "career", "politic", "vote", "military", "combat", "soldier",
                    "school", "teach", "sexual", "partner", "financ", "money", "manager", "meeting", "child", "parent", "mother", "couple",
                    "family", "job", "work", "boss", "ceo")


def build_claims(language: str = "en") -> Dict[str, object]:
    pack = load_language(language)
    statements = load_pool(language, "class_0")[:N_ROWS]
    n_pairs, n_frames = len(CLAIM_PAIRS), len(CLAIM_FRAMES)
    pair_of = balanced_assignment(N_ROWS, n_pairs, np.random.default_rng(SEED))         # which claim pair each statement gets
    frame_pos = balanced_assignment(N_ROWS, n_frames, np.random.default_rng(SEED + 1))
    frame_neg = balanced_assignment(N_ROWS, n_frames, np.random.default_rng(SEED + 2))  # independent of the positive frame

    def prompt(frame_id: int, claim: str, statement: str) -> str:
        return (CLAIM_FRAMES[frame_id].format(claim=claim) + pack["instruction_separator"]
                + pack["question_tail"].format(statement=statement))

    rows, train = [], []
    for i, st in enumerate(statements):
        topic, mis, egal = CLAIM_PAIRS[int(pair_of[i])]
        for label, kind, claim, fid in ((1.0, "claim_misogynistic", mis, int(frame_pos[i])), (0.0, "claim_egalitarian", egal, int(frame_neg[i]))):
            pid = len(rows)
            rows.append({"prompt_id": pid, "user_prompt": prompt(fid, claim, st), "kind": kind, "concept": topic, "concept_text": claim,
                         "frame_id": fid, "pool": "class_0", "statement_id": i, "pair_id": int(pair_of[i]), "statement": st})
            train.append({"dataset": DATASET_NAME, "row_idx": len(train), "prompt_id": pid, "label": label})
    registry, train_df = pd.DataFrame(rows), pd.DataFrame(train)
    meta = {"negatives": "claims", "language": language, "n_prompts": len(registry), "n_pairs": n_pairs, "n_frames": n_frames,
            "dataset": DATASET_NAME, "topics": [p[0] for p in CLAIM_PAIRS], "seed": SEED, "registry_sha1": registry_hash(registry)}
    return {"registry": registry, "train": train_df, "meta": meta}


def registry_hash(registry: pd.DataFrame) -> str:
    h = hashlib.sha1()
    for p in registry["user_prompt"].tolist():
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def validate_claims(t: Dict[str, object]) -> None:
    reg, train = t["registry"], t["train"]
    assert len(reg) == 2 * N_ROWS and (reg["prompt_id"].values == range(len(reg))).all()
    assert reg["user_prompt"].is_unique, "every prompt must be unique"
    assert train["label"].tolist() == [1.0, 0.0] * N_ROWS, "rows must be interleaved positive / negative"
    pos, neg = reg[reg["kind"] == "claim_misogynistic"].reset_index(drop=True), reg[reg["kind"] == "claim_egalitarian"].reset_index(drop=True)
    assert len(pos) == len(neg) == N_ROWS
    assert (pos["statement"].values == neg["statement"].values).all(), "a positive and its negative share the statement"
    assert (pos["pair_id"].values == neg["pair_id"].values).all(), "a positive and its negative share the claim pair (topic)"
    for r in reg.itertuples():
        assert r.concept_text in r.user_prompt and r.user_prompt.endswith(r.statement)
    for name, col in (("pairs", "pair_id"), ("frames (pos)", None)):
        counts = (pos["pair_id"] if col else pos["frame_id"]).value_counts()
        assert counts.max() - counts.min() <= 1, f"{name} must be balanced"
    assert set(pos["concept_text"]) == {p[1] for p in CLAIM_PAIRS} and set(neg["concept_text"]) == {p[2] for p in CLAIM_PAIRS}
    assert not (set(pos["frame_id"]) - set(range(len(CLAIM_FRAMES))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the claims-vs-counterclaims stance dataset (experiment folder).")
    experiment.add_experiment_arg(ap)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()
    experiment.apply_experiment_arg(args)
    t = build_claims(args.language)
    validate_claims(t)
    out = experiment.dataset_dir()
    out.mkdir(parents=True, exist_ok=True)
    t["registry"].to_csv(out / "prompts.csv", index=False)
    t["train"].to_csv(out / "train_datasets.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(t["meta"], indent=2), encoding="utf-8")
    reg = t["registry"]
    print(f"[claims_dataset] experiment {experiment.get_experiment()!r}: {len(reg)} prompts, {t['meta']['n_pairs']} claim pairs, registry sha1 {t['meta']['registry_sha1'][:12]}")
    print(f"  topics : {t['meta']['topics']}")
    for k in (0, 2):
        print(f"  + {reg.at[k, 'user_prompt']!r}\n  - {reg.at[k + 1, 'user_prompt']!r}")
    print(f"[claims_dataset] wrote prompts.csv, train_datasets.csv, manifest.json -> {out}")


if __name__ == "__main__":
    main()
