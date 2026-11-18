"""
Dataset builder (v5): authors-style contrastive data for the misogyny concept, with held-out subtypes.

What it builds (all model-agnostic: raw user prompts, no chat template, no tokenizer)
-------------------------------------------------------------------------------------
into dataset/processed/<experiment>/ :

1. prompts.csv        Registry of EVERY unique prompt (one row per prompt_id). Activations are extracted once per
                      registry row, so the 200 neutral negatives shared by all datasets are only run once.
2. train_datasets.csv 22 authors-sized training sets, 400 rows each, strictly interleaved [pos_0, neg_0, pos_1, ...]:
                        subtype__<slug>   x20  one subtype phrase (random frames) on all 200 class_0 statements
                        single__misogyny  x1   the broad phrase "misogyny"
                        pooled__misogyny  x1   each class_0 statement gets ONE random NON-held-out subtype
                                               (12-13 each for 16 subtypes)
                      Negatives are always the 200 class_1 statements with the neutral question and NO instruction.
3. paired_test.csv    Every FRESH statement (never used in training) rendered four ways: neutral / misogyny subtype /
                      benign-concept control / non-gender-hateful control. The misogyny arm uses only NON-held-out
                      subtypes ("seen phrases").
4. heldout_test.csv   Every fresh statement rendered with EACH held-out subtype ("unseen phrases"). Compare S, P and A here.
5. manifest.json      Seed, language, frame mode, concept lists, counts and a hash of the registry. Downstream scripts
                      read the concept lists from here, and activations store the hash, so stale activations are
                      detected automatically.

The "matched" experiment (--negatives matched)
----------------------------------------------
The default datasets use NEUTRAL negatives (the question with no instruction). The vectors learned from them score every
instruction above its neutral twin and score non-gender hateful prompts as high as misogyny prompts (see control_eval.py,
pilot_contrasts.py). `--negatives matched` builds the alternative:
  * each negative is the SAME class_0 statement as its positive, with an instruction naming a CONTROL concept (half benign,
    half non-gender hateful, random frame). The instruction itself cancels, and both rows of a pair share the statement.
  * the training controls (gardening, chess, classical music / religious bigotry, classism, ageism) are different from the
    test controls, so nothing is learned by name; the hateful ones avoid gender and sexuality.
  * a fifth TEST arm, "egalitarian" (gender equality, respect for women, feminism): same topic as misogyny, opposite stance.
  * everything the default build registers is registered first and in the same order; the matched negatives and the
    egalitarian arm are APPENDED, so the first n_base_prompts prompts are identical to the default build and their
    activations can be reused: `python activation_extraction.py --experiment matched --extend-from main` extracts only the
    new prompts (about 420).

The "stance" experiment (--negatives stance)  <- the authors' recipe for STANCE concepts
--------------------------------------------------------------------------------------
For liberal vs conservative the authors do NOT use neutral negatives: half of the statements get "Take on a conservative stance",
the other half "Take on a liberal stance" (arXiv:2502.03708, SI B.2), so the learned vector is stance A minus stance B with the
instruction format cancelled. The neutral recipe is the one used for persona / topic concepts. Misogyny is a stance, so:
  * each negative is the SAME class_0 statement as its positive, with an instruction naming an EGALITARIAN concept (anti-sexism,
    women's empowerment, women's rights, equal pay for women, women's autonomy, women's education), random balanced frame;
  * the training egalitarian concepts are different from the TEST egalitarian arm (gender equality, respect for women, feminism);
  * the test egalitarian arm is built exactly as in "matched", so control_eval / compare_vectors / pilot_contrasts work unchanged;
  * like "matched", everything is APPENDED after the default registry (`activation_extraction.py --extend-from main` extracts only
    the ~420 new prompts).
This is the contrast of the old (v3) dataset, which steered, in the authors' format and with held-out subtypes.

Knobs
-----
--frame-mode random|fixed   random (default): every positive draws one of the frames (seeded, balanced).
                            fixed: every TRAINING positive uses --fixed-frame-id (authors' single-template style).
                            Test prompts are random-frame in BOTH modes, so both experiments are scored on identical test data.
--negatives neutral|matched|stance|stance_wide   neutral (default): negatives are the question with no instruction. matched / stance: see above.
                      stance_wide: stance with 21 egalitarian training concepts (the 6 of stance + 15 respect / dignity names) instead of 6.
--language <code>           which data/languages/<code>/ pack to use (default en).
--experiment <tag>          output folder namespace (use a different tag for each variant you want to keep).

Design rules (and why)
----------------------
* Rows are NEVER shuffled. `split_train_val` takes the first 80% of rows as inner-train and the rest as inner-val. Every
  statement appears in exactly one row of a dataset and rows alternate pos/neg, so that split is statement-disjoint
  and exactly balanced, mirroring the authors' `split_indices`.
* Statements are `.strip()`ped when read, so no trailing newline reaches a prompt.
* Positives use class_0 statements and negatives use class_1 statements (as in the authors' code). Those pools are
  stylistically distinguishable, which is why the paired tests (same statement with and without the instruction) exist.
"""

import argparse
import hashlib
import json
import os
import shutil
import warnings
import zlib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import experiment
from experiment import PROJECT_ROOT, dataset_dir
from raw_data import (
    BENIGN_CONTROL_CONCEPTS,
    EGALITARIAN_CONTROL_CONCEPTS,
    HATEFUL_CONTROL_CONCEPTS,
    HELD_OUT_SUBTYPES,
    LANGUAGES_DIR,
    MISOGYNY_SUBTYPES,
    POOLED_NAME,
    SINGLE_NAME,
    SINGLE_PHRASE_CONCEPT,
    TRAIN_BENIGN_CONTROL_CONCEPTS,
    TRAIN_EGALITARIAN_CONCEPTS,
    TRAIN_EGALITARIAN_EXTRA_CONCEPTS,
    TRAIN_HATEFUL_CONTROL_CONCEPTS,
    build_neutral_prompt,
    build_positive_prompt,
    load_language,
    phrase,
    subtype_name,
)

load_dotenv()

SEED = int(os.getenv("SEED", 123))
LANGUAGE = os.getenv("LANGUAGE", "en")
FRAME_MODE = os.getenv("FRAME_MODE", "random")
FIXED_FRAME_ID = int(os.getenv("FIXED_FRAME_ID", 0))
NEGATIVES = os.getenv("NEGATIVES", "neutral")

N_TRAIN_PER_POOL = 200          # class_0 -> positives, class_1 -> negatives (authors' sizes)
MIN_FRESH_PER_POOL = 50         # sanity floor for the test statements of each pool
PAIRED_ARMS = ("neutral", "subtype", "benign", "hateful")
EXTRA_ARM = "egalitarian"                       # test-only fifth arm, matched experiment only
MATCHED_KINDS = ("matched_benign", "matched_hateful")
STANCE_KIND = "stance_egalitarian"               # negatives of the stance experiment: same statement, EGALITARIAN instruction
NEGATIVES_CHOICES = ("neutral", "matched", "stance", "stance_wide")
POOL_FILES = ("class_0", "class_1", "fresh_0", "fresh_1")


# -----------------------------------------------------------------------
# 1. Statement pools (one folder per language, line-aligned across languages)
# -----------------------------------------------------------------------
def load_pool(language: str, name: str) -> list:
    """Read data/languages/<language>/<name>.txt: strip whitespace/newlines, drop blank lines."""
    path = LANGUAGES_DIR / language / f"{name}.txt"
    if not path.exists():
        hint = " Run `python dataset_builder.py --export-fresh` first." if name.startswith("fresh_") and language == "en" else ""
        raise FileNotFoundError(f"{path} not found.{hint}")
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [s for s in lines if s]


def load_pools(language: str = "en") -> dict:
    """class_0, class_1 (training) and fresh_0, fresh_1 (test-only), with duplicate/overlap/alignment checks."""
    pools = {k: load_pool(language, k) for k in POOL_FILES}
    for k, v in pools.items():
        if len(set(v)) != len(v):
            raise ValueError(f"{language}/{k}.txt contains {len(v) - len(set(v))} duplicate line(s); every statement must be unique")
    for k in ("class_0", "class_1"):
        if len(pools[k]) != N_TRAIN_PER_POOL:
            raise ValueError(f"{language}/{k}.txt has {len(pools[k])} statements, expected {N_TRAIN_PER_POOL}")
    everything = [s for k in POOL_FILES for s in pools[k]]
    if len(set(everything)) != len(everything):
        raise ValueError(f"{language}: a statement appears in more than one of {POOL_FILES} (train/test leakage)")
    for k in ("fresh_0", "fresh_1"):
        if len(pools[k]) < MIN_FRESH_PER_POOL:
            raise ValueError(f"{language}/{k}.txt has {len(pools[k])} statements; need at least {MIN_FRESH_PER_POOL}")
    if language != "en":
        for k in POOL_FILES:
            ref = len(load_pool("en", k))
            if len(pools[k]) != ref:
                raise ValueError(f"{language}/{k}.txt has {len(pools[k])} lines but en/{k}.txt has {ref}: "
                                 f"translations must be line-aligned with the English files")
    return pools


def export_fresh_lists() -> None:
    """
    Derive en/fresh_0.txt and en/fresh_1.txt from the `_long` source files: the unique lines of each `_long` file that
    are not in class_0 or class_1, in file order. (The `_long` files repeat statements heavily: ~315 and ~303 unique.)
    Translators translate these files too, line by line.
    """
    src = PROJECT_ROOT / "data" / "general_statements"
    en = LANGUAGES_DIR / "en"
    rd = lambda p: [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    class_0, class_1 = rd(en / "class_0.txt"), rd(en / "class_1.txt")
    train = set(class_0) | set(class_1)
    for out_name, long_name in (("fresh_0", "class_0_long"), ("fresh_1", "class_1_long")):
        fresh = [s for s in dict.fromkeys(rd(src / f"{long_name}.txt")) if s not in train]
        (en / f"{out_name}.txt").write_text("\n".join(fresh) + "\n", encoding="utf-8")
        print(f"[export_fresh] {out_name}.txt: {len(fresh)} statements")


def init_language(language: str) -> None:
    """Copy the English pack to data/languages/<language>/ as a starting point for translation."""
    dest = LANGUAGES_DIR / language
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; refusing to overwrite")
    dest.mkdir(parents=True)
    for name in ("class_0.txt", "class_1.txt", "fresh_0.txt", "fresh_1.txt"):
        shutil.copy(LANGUAGES_DIR / "en" / name, dest / name)
    pack = json.loads((LANGUAGES_DIR / "en" / "language.json").read_text(encoding="utf-8"))
    pack["language"] = language
    (dest / "language.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[init_language] created {dest} (English text, to be translated).\n"
          f"  Translate: language.json (frames, question_tail, every phrase) and the four .txt files LINE BY LINE\n"
          f"  (same number of lines, same order). Keep {{concept}} and {{statement}} slots in place.\n"
          f"  Then: python dataset_builder.py --language {language} --experiment <tag>")


# -----------------------------------------------------------------------
# 2. Seeded, balanced assignment helpers
# -----------------------------------------------------------------------
def _stable_id(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def balanced_assignment(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """n slots filled with 0..k-1 as evenly as possible (counts differ by at most 1), then shuffled."""
    base = np.resize(np.arange(k), n)
    rng.shuffle(base)
    return base


# -----------------------------------------------------------------------
# 3. Registry + datasets
# -----------------------------------------------------------------------
class _Registry:
    """Collects unique prompts and hands out sequential prompt_ids."""

    COLUMNS = ["prompt_id", "kind", "concept", "concept_text", "frame_id", "pool", "statement_id", "pair_id",
               "statement", "language", "user_prompt"]

    def __init__(self, language: str):
        self.language = language
        self.rows = []
        self._seen = set()

    def add(self, user_prompt: str, **meta) -> int:
        if user_prompt in self._seen:
            raise ValueError(f"Duplicate prompt text would break the one-row-per-prompt invariant:\n{user_prompt!r}")
        self._seen.add(user_prompt)
        prompt_id = len(self.rows)
        self.rows.append({"prompt_id": prompt_id, "user_prompt": user_prompt, "language": self.language, **meta})
        return prompt_id

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)[self.COLUMNS]


def build_all(seed: int = None, language: str = None, frame_mode: str = None, fixed_frame_id: int = None,
              subtypes: list = None, heldout: list = None, pack: dict = None, negatives: str = None) -> dict:
    """
    Build the registry, the training datasets, the paired test and the held-out test.
    negatives="neutral" (default) or "matched" (see the module docstring).

    Everything defaults to the module settings / raw_data master lists; the keyword arguments exist for the
    command line and for tests (for example to build with an extra concept).
    """
    seed = SEED if seed is None else seed
    language = LANGUAGE if language is None else language
    frame_mode = FRAME_MODE if frame_mode is None else frame_mode
    fixed_frame_id = FIXED_FRAME_ID if fixed_frame_id is None else fixed_frame_id
    negatives = NEGATIVES if negatives is None else negatives
    if negatives not in NEGATIVES_CHOICES:
        raise ValueError(f"negatives must be one of {NEGATIVES_CHOICES}, got {negatives!r}")
    matched = negatives == "matched"
    stance = negatives in ("stance", "stance_wide")     # stance_wide = the same recipe with a wider (respect / dignity) egalitarian pool
    egal_pool = list(TRAIN_EGALITARIAN_CONCEPTS) + (list(TRAIN_EGALITARIAN_EXTRA_CONCEPTS) if negatives == "stance_wide" else [])
    appended_build = matched or stance     # both append their negatives and the egalitarian test arm after the base registry
    subtypes = list(MISOGYNY_SUBTYPES if subtypes is None else subtypes)
    heldout = list(HELD_OUT_SUBTYPES if heldout is None else heldout)
    if frame_mode not in ("random", "fixed"):
        raise ValueError(f"frame_mode must be 'random' or 'fixed', got {frame_mode!r}")
    if not set(heldout) <= set(subtypes):
        raise ValueError("held-out subtypes must be a subset of the subtypes")
    train_subtypes = [c for c in subtypes if c not in heldout]
    if not train_subtypes:
        raise ValueError("every subtype is held out; nothing left to train P and A on")

    concept_ids = [SINGLE_PHRASE_CONCEPT] + subtypes + BENIGN_CONTROL_CONCEPTS + HATEFUL_CONTROL_CONCEPTS
    if matched:
        concept_ids += TRAIN_BENIGN_CONTROL_CONCEPTS + TRAIN_HATEFUL_CONTROL_CONCEPTS + EGALITARIAN_CONTROL_CONCEPTS
    if stance:
        concept_ids += egal_pool + EGALITARIAN_CONTROL_CONCEPTS
    pack = pack if pack is not None else load_language(language, concept_ids)
    n_frames = len(pack["frames"])
    if frame_mode == "fixed" and not 0 <= fixed_frame_id < n_frames:
        raise ValueError(f"fixed_frame_id {fixed_frame_id} out of range for {n_frames} frames")

    rng = lambda *keys: np.random.default_rng([seed] + [int(k) for k in keys])
    pools = load_pools(language)
    class_0, class_1 = pools["class_0"], pools["class_1"]
    n = len(class_0)
    reg = _Registry(language)

    # ---- 3a. neutral negatives (class_1, no instruction) ----
    neutral_ids = [
        reg.add(build_neutral_prompt(pack, s), kind="neutral", concept="", concept_text="", frame_id=-1,
                pool="class_1", statement_id=i, pair_id=-1, statement=s)
        for i, s in enumerate(class_1)
    ]

    # ---- 3b. positives: one prompt per (concept, class_0 statement) for the single phrase and EVERY subtype ----
    positive_ids = {}
    for concept in [SINGLE_PHRASE_CONCEPT] + subtypes:
        if frame_mode == "fixed":
            frames = np.full(n, fixed_frame_id)
        else:
            frames = balanced_assignment(n, n_frames, rng(1, _stable_id(concept)))
        kind = "single" if concept == SINGLE_PHRASE_CONCEPT else "subtype"
        positive_ids[concept] = [
            reg.add(build_positive_prompt(pack, int(frames[i]), concept, s), kind=kind, concept=concept,
                    concept_text=phrase(pack, concept), frame_id=int(frames[i]), pool="class_0",
                    statement_id=i, pair_id=-1, statement=s)
            for i, s in enumerate(class_0)
        ]

    # ---- 3d. fresh statements (test-only), both pools interleaved so both styles are spread through pair_ids ----
    fresh = []
    for i in range(max(len(pools["fresh_0"]), len(pools["fresh_1"]))):
        if i < len(pools["fresh_0"]):
            fresh.append((pools["fresh_0"][i], "class_0_long"))
        if i < len(pools["fresh_1"]):
            fresh.append((pools["fresh_1"][i], "class_1_long"))
    n_test = len(fresh)

    # Test frames are random in BOTH frame modes (identical test prompts across experiments).
    sub_assign = balanced_assignment(n_test, len(train_subtypes), rng(4))
    ben_assign = balanced_assignment(n_test, len(BENIGN_CONTROL_CONCEPTS), rng(5))
    hat_assign = balanced_assignment(n_test, len(HATEFUL_CONTROL_CONCEPTS), rng(6))
    sub_frames = balanced_assignment(n_test, n_frames, rng(7))
    ben_frames = balanced_assignment(n_test, n_frames, rng(8))
    hat_frames = balanced_assignment(n_test, n_frames, rng(9))

    test_rows = []
    for j, (statement, pool) in enumerate(fresh):
        sub_c = train_subtypes[sub_assign[j]]
        ben_c = BENIGN_CONTROL_CONCEPTS[ben_assign[j]]
        hat_c = HATEFUL_CONTROL_CONCEPTS[hat_assign[j]]
        arms = {
            "neutral": ("", -1, build_neutral_prompt(pack, statement)),
            "subtype": (sub_c, int(sub_frames[j]), build_positive_prompt(pack, int(sub_frames[j]), sub_c, statement)),
            "benign": (ben_c, int(ben_frames[j]), build_positive_prompt(pack, int(ben_frames[j]), ben_c, statement)),
            "hateful": (hat_c, int(hat_frames[j]), build_positive_prompt(pack, int(hat_frames[j]), hat_c, statement)),
        }
        for arm in PAIRED_ARMS:
            concept, frame_id, prompt = arms[arm]
            pid = reg.add(prompt, kind=f"test_{arm}", concept=concept, concept_text=phrase(pack, concept) if concept else "",
                          frame_id=frame_id, pool=pool, statement_id=-1, pair_id=j, statement=statement)
            test_rows.append({"pair_id": j, "arm": arm, "prompt_id": pid, "concept": concept,
                              "frame_id": frame_id, "pool": pool})

    # ---- 3e. held-out test: EVERY fresh statement with EACH held-out subtype (unseen phrases) ----
    heldout_rows = []
    for h in heldout:
        h_frames = balanced_assignment(n_test, n_frames, rng(10, _stable_id(h)))
        for j, (statement, pool) in enumerate(fresh):
            pid = reg.add(build_positive_prompt(pack, int(h_frames[j]), h, statement), kind="test_heldout", concept=h,
                          concept_text=phrase(pack, h), frame_id=int(h_frames[j]), pool=pool, statement_id=-1,
                          pair_id=j, statement=statement)
            heldout_rows.append({"pair_id": j, "concept": h, "prompt_id": pid, "frame_id": int(h_frames[j]), "pool": pool})

    n_base = len(reg.rows)     # everything registered so far is identical in the neutral and the matched build

    # ---- 3f. MATCHED / STANCE experiments only: negatives and the egalitarian arm, APPENDED after the base registry ----
    matched_ids = []
    if stance:
        # negatives = the SAME class_0 statement under an instruction naming a TRAIN egalitarian concept (balanced, random frames)
        egal_train_of = balanced_assignment(n, len(egal_pool), rng(17))
        egal_neg_frames = balanced_assignment(n, n_frames, rng(18))
        for i, s_ in enumerate(class_0):
            c_ = egal_pool[egal_train_of[i]]
            matched_ids.append(
                reg.add(build_positive_prompt(pack, int(egal_neg_frames[i]), c_, s_), kind=STANCE_KIND, concept=c_,
                        concept_text=phrase(pack, c_), frame_id=int(egal_neg_frames[i]), pool="class_0", statement_id=i,
                        pair_id=-1, statement=s_))
    if matched:
        # type (benign / hateful) balanced 100/100, then the concept within each type balanced; random frames
        kinds_slot = balanced_assignment(n, 2, rng(11))
        concept_of = [None] * n
        for t, pool_ids in ((0, TRAIN_BENIGN_CONTROL_CONCEPTS), (1, TRAIN_HATEFUL_CONTROL_CONCEPTS)):
            slots = np.where(kinds_slot == t)[0]
            for slot, a in zip(slots, balanced_assignment(len(slots), len(pool_ids), rng(12 + t))):
                concept_of[slot] = pool_ids[a]
        neg_frames = balanced_assignment(n, n_frames, rng(14))
        for i, s_ in enumerate(class_0):
            c_ = concept_of[i]
            kind = "matched_benign" if kinds_slot[i] == 0 else "matched_hateful"
            matched_ids.append(
                reg.add(build_positive_prompt(pack, int(neg_frames[i]), c_, s_), kind=kind, concept=c_,
                        concept_text=phrase(pack, c_), frame_id=int(neg_frames[i]), pool="class_0", statement_id=i,
                        pair_id=-1, statement=s_))
    if appended_build:
        egal_assign = balanced_assignment(n_test, len(EGALITARIAN_CONTROL_CONCEPTS), rng(15))
        egal_frames = balanced_assignment(n_test, n_frames, rng(16))
        for j, (statement, pool) in enumerate(fresh):
            c_ = EGALITARIAN_CONTROL_CONCEPTS[egal_assign[j]]
            pid = reg.add(build_positive_prompt(pack, int(egal_frames[j]), c_, statement), kind=f"test_{EXTRA_ARM}",
                          concept=c_, concept_text=phrase(pack, c_), frame_id=int(egal_frames[j]), pool=pool,
                          statement_id=-1, pair_id=j, statement=statement)
            test_rows.append({"pair_id": j, "arm": EXTRA_ARM, "prompt_id": pid, "concept": c_,
                              "frame_id": int(egal_frames[j]), "pool": pool})

    # ---- 3c. training datasets (strictly interleaved pos/neg, never shuffled) ----
    #      neutral build: negative i = class_1 statement i with no instruction
    #      matched build: negative i = the SAME class_0 statement i with a control-concept instruction
    #      stance build : negative i = the SAME class_0 statement i with an EGALITARIAN instruction (the opposite stance)
    train_rows = []

    def add_dataset(name, pos_prompt_ids, pos_concepts):
        for i in range(n):
            train_rows.append({"dataset": name, "row_idx": 2 * i, "prompt_id": pos_prompt_ids[i], "label": 1.0,
                               "concept": pos_concepts[i], "pool": "class_0", "statement_id": i})
            if appended_build:
                train_rows.append({"dataset": name, "row_idx": 2 * i + 1, "prompt_id": matched_ids[i], "label": 0.0,
                                   "concept": reg.rows[matched_ids[i]]["concept"], "pool": "class_0", "statement_id": i})
            else:
                train_rows.append({"dataset": name, "row_idx": 2 * i + 1, "prompt_id": neutral_ids[i], "label": 0.0,
                                   "concept": "", "pool": "class_1", "statement_id": i})

    add_dataset(SINGLE_NAME, positive_ids[SINGLE_PHRASE_CONCEPT], [SINGLE_PHRASE_CONCEPT] * n)
    for concept in subtypes:
        add_dataset(subtype_name(concept), positive_ids[concept], [concept] * n)

    # pooled: each class_0 statement gets exactly ONE NON-held-out subtype (balanced, seeded). The prompt is
    # re-used from the registry, so this adds no new prompts.
    assign = balanced_assignment(n, len(train_subtypes), rng(2))
    pooled_concepts = [train_subtypes[a] for a in assign]
    add_dataset(POOLED_NAME, [positive_ids[c][i] for i, c in enumerate(pooled_concepts)], pooled_concepts)


    meta = {
        "seed": seed, "language": language, "frame_mode": frame_mode,
        "fixed_frame_id": int(fixed_frame_id) if frame_mode == "fixed" else None,
        "n_frames": n_frames, "frames": pack["frames"], "question_tail": pack["question_tail"],
        "subtypes": subtypes, "heldout": heldout, "train_subtypes": train_subtypes,
        "single_phrase": SINGLE_PHRASE_CONCEPT, "benign_controls": list(BENIGN_CONTROL_CONCEPTS),
        "hateful_controls": list(HATEFUL_CONTROL_CONCEPTS), "phrases": {c: phrase(pack, c) for c in concept_ids},
        "negatives": negatives, "n_base_prompts": n_base,
        "train_egalitarian_controls": list(egal_pool) if stance else [],
        "train_benign_controls": list(TRAIN_BENIGN_CONTROL_CONCEPTS) if matched else [],
        "train_hateful_controls": list(TRAIN_HATEFUL_CONTROL_CONCEPTS) if matched else [],
        "egalitarian_controls": list(EGALITARIAN_CONTROL_CONCEPTS) if appended_build else [],
        "extra_arms": [EXTRA_ARM] if appended_build else [],
    }
    return {"registry": reg.frame(), "train": pd.DataFrame(train_rows), "test": pd.DataFrame(test_rows),
            "heldout_test": pd.DataFrame(heldout_rows), "meta": meta, "pack": pack}


def registry_hash(registry: pd.DataFrame) -> str:
    h = hashlib.sha1()
    for p in registry["user_prompt"].tolist():
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# -----------------------------------------------------------------------
# 4. Validation (runs on every build; also used by the tests)
# -----------------------------------------------------------------------
def validate(tables: dict) -> None:
    registry, train, test, held = tables["registry"], tables["train"], tables["test"], tables["heldout_test"]
    meta, pack = tables["meta"], tables["pack"]
    n = N_TRAIN_PER_POOL
    subtypes, heldout, train_sub = meta["subtypes"], meta["heldout"], meta["train_subtypes"]

    assert (registry["prompt_id"].values == np.arange(len(registry))).all(), "prompt_ids must be 0..N-1 in order"
    assert registry["user_prompt"].is_unique, "registry prompts must be unique"
    assert (registry["language"] == meta["language"]).all()
    assert (registry["statement"] == registry["statement"].str.strip()).all(), "statement has stray whitespace"
    assert not registry["user_prompt"].str.endswith(("\n", " ")).any(), "prompt ends with whitespace"

    assert train["dataset"].nunique() == len(subtypes) + 2, "expected one dataset per subtype + single + pooled"
    reg = registry.set_index("prompt_id")
    frame_prefixes = [f.split("{")[0] for f in pack["frames"]]
    misogyny_texts = [phrase(pack, c) for c in [SINGLE_PHRASE_CONCEPT] + subtypes]
    for name, d in train.groupby("dataset", sort=False):
        d = d.sort_values("row_idx")
        assert len(d) == 2 * n, f"{name}: {len(d)} rows, expected {2 * n}"
        assert (d["row_idx"].values == np.arange(2 * n)).all(), f"{name}: row_idx must be 0..{2 * n - 1}"
        assert (d["label"].values[0::2] == 1.0).all() and (d["label"].values[1::2] == 0.0).all(), f"{name}: rows must alternate 1.0/0.0"
        pos, neg = d.iloc[0::2], d.iloc[1::2]
        assert pos["statement_id"].is_unique and neg["statement_id"].is_unique, f"{name}: a statement is used twice"
        assert set(reg.loc[pos["prompt_id"], "pool"]) == {"class_0"}
        for pid, concept in zip(pos["prompt_id"], pos["concept"]):
            assert reg.at[pid, "concept_text"] in reg.at[pid, "user_prompt"]
            assert reg.at[pid, "concept"] == concept
        if meta["negatives"] == "neutral":
            assert set(reg.loc[neg["prompt_id"], "pool"]) == {"class_1"}
            assert (reg.loc[neg["prompt_id"], "kind"] == "neutral").all(), f"{name}: negatives must be neutral prompts"
            assert not neg["prompt_id"].map(lambda p: any(fp in reg.at[p, "user_prompt"] for fp in frame_prefixes if fp)).any()
        else:
            # matched / stance: the SAME statement as the positive, an instruction naming a control (matched) or an
            # egalitarian (stance) concept, no misogyny phrase
            neg_kinds = MATCHED_KINDS if meta["negatives"] == "matched" else (STANCE_KIND,)
            assert set(reg.loc[neg["prompt_id"], "pool"]) == {"class_0"}
            assert reg.loc[neg["prompt_id"], "kind"].isin(neg_kinds).all(), f"{name}: negatives must be {meta['negatives']} prompts"
            assert (reg.loc[neg["prompt_id"], "statement"].values == reg.loc[pos["prompt_id"], "statement"].values).all(), \
                f"{name}: each negative must share its positive's statement"
            for pid in neg["prompt_id"]:
                text = reg.at[pid, "user_prompt"]
                assert reg.at[pid, "concept_text"] in text
                assert not any(m in text for m in misogyny_texts), f"{name}: a misogyny phrase leaked into a negative"

    # ---- pooled: only non-held-out subtypes, as evenly as possible ----
    pooled = train[train["dataset"] == POOLED_NAME].sort_values("row_idx").iloc[0::2]
    counts = pooled["concept"].value_counts()
    assert set(counts.index) == set(train_sub), "pooled must use exactly the non-held-out subtypes"
    assert counts.max() - counts.min() <= 1, "pooled: subtypes must be balanced (counts differ by at most 1)"
    assert counts.sum() == n

    # ---- held-out isolation: no held-out phrase in P's data or in the seen-phrase test ----
    for h in heldout:
        h_text = phrase(pack, h)
        pooled_texts = reg.loc[pooled["prompt_id"], "user_prompt"]
        assert not pooled_texts.str.contains(h_text, regex=False).any(), f"held-out phrase {h!r} leaked into the pooled dataset"
    assert not set(test[test["arm"] == "subtype"]["concept"]) & set(heldout), "seen-phrase test uses a held-out subtype"
    assert set(test[test["arm"] == "subtype"]["concept"]) <= set(train_sub)

    # ---- fresh statements never in training ----
    train_stmts = set(registry.loc[registry["kind"].isin(["neutral", "single", "subtype", *MATCHED_KINDS, STANCE_KIND]), "statement"])
    test_stmts = set(registry.loc[registry["kind"].str.startswith("test_"), "statement"])
    assert test_stmts and not (train_stmts & test_stmts), "test statements leaked from the training pools"

    # ---- paired test ----
    per_pair = test.groupby("pair_id")["arm"].apply(lambda a: tuple(a))
    expected_arms = PAIRED_ARMS + tuple(meta["extra_arms"])
    assert (per_pair == expected_arms).all(), f"every pair needs the arms {expected_arms} in order"
    n_pairs = test["pair_id"].nunique()
    assert n_pairs >= 2 * MIN_FRESH_PER_POOL

    # ---- held-out test: every fresh statement x every held-out subtype ----
    assert len(held) == n_pairs * len(heldout), "held-out test must cover every fresh statement with every held-out subtype"
    assert not held.duplicated(["pair_id", "concept"]).any()
    assert set(held["concept"]) == set(heldout)
    assert (held.groupby("concept")["pair_id"].nunique() == n_pairs).all()

    # ---- matched / stance experiments: appended prompts, balance, no train/test control leakage ----
    n_base = meta["n_base_prompts"]
    appended = registry.iloc[n_base:]
    all_neg_kinds = list(MATCHED_KINDS) + [STANCE_KIND]
    if meta["negatives"] in ("matched", "stance", "stance_wide"):
        stance_build = meta["negatives"] in ("stance", "stance_wide")
        neg_kinds = (STANCE_KIND,) if stance_build else MATCHED_KINDS
        assert set(appended["kind"]) <= set(neg_kinds) | {f"test_{EXTRA_ARM}"}, "only this build's negatives and the egalitarian arm may be appended"
        assert not registry.iloc[:n_base]["kind"].isin(all_neg_kinds + [f"test_{EXTRA_ARM}"]).any(), "base prompts must not contain negatives of an appended build"
        mk = registry[registry["kind"].isin(neg_kinds)]
        assert len(mk) == n and (mk["prompt_id"] >= n_base).all()
        if stance_build:
            c = mk["concept"].value_counts()
            assert c.max() - c.min() <= 1, "stance: egalitarian training concepts must be balanced"
            train_controls = set(meta["train_egalitarian_controls"])
            test_controls = set(meta["benign_controls"]) | set(meta["hateful_controls"]) | set(meta["egalitarian_controls"])
            assert set(mk["concept"]) == train_controls and not (train_controls & test_controls), "train egalitarian concepts must differ from every test control"
        else:
            by_type = mk["kind"].value_counts()
            assert abs(by_type.get("matched_benign", 0) - by_type.get("matched_hateful", 0)) <= 1, "half benign, half hateful negatives"
            for kind, g in mk.groupby("kind"):
                c = g["concept"].value_counts()
                assert c.max() - c.min() <= 1, f"{kind}: control concepts must be balanced"
            train_controls = set(meta["train_benign_controls"]) | set(meta["train_hateful_controls"])
            test_controls = set(meta["benign_controls"]) | set(meta["hateful_controls"]) | set(meta["egalitarian_controls"])
            assert set(mk["concept"]) == train_controls and not (train_controls & test_controls), "train controls must differ from every test control"
        egal = test[test["arm"] == EXTRA_ARM]
        assert len(egal) == n_pairs and set(egal["concept"]) == set(meta["egalitarian_controls"])
        neutral_stmt = reg.loc[test[test["arm"] == "neutral"].set_index("pair_id")["prompt_id"], "statement"].values
        egal_stmt = reg.loc[egal.set_index("pair_id").loc[test[test["arm"] == "neutral"]["pair_id"], "prompt_id"], "statement"].values
        assert (neutral_stmt == egal_stmt).all(), "the egalitarian arm must use the same statements as the other arms"
    else:
        assert n_base == len(registry) and not meta["extra_arms"], "neutral build: nothing is appended"

    # ---- frames ----
    pos_rows = registry[registry["kind"].isin(["single", "subtype"])]
    if meta["frame_mode"] == "random":
        fc = pos_rows.groupby("concept")["frame_id"].value_counts().unstack(fill_value=0).reindex(columns=range(meta["n_frames"]), fill_value=0)
        assert (fc.max(axis=1) - fc.min(axis=1) <= 1).all(), "frames must be balanced within each concept"
    else:
        assert (pos_rows["frame_id"] == meta["fixed_frame_id"]).all(), "fixed frame mode: every training positive must use the fixed frame"


# -----------------------------------------------------------------------
# 5. Save / report
# -----------------------------------------------------------------------
def save_all(tables: dict) -> dict:
    out = dataset_dir()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        m = tables["meta"]
        changed = [k for k in ("language", "frame_mode", "seed") if old.get(k) != m[k]]
        if changed:
            print(f"[dataset_builder] WARNING: experiment {experiment.get_experiment()!r} was built with different "
                  f"{changed}; overwriting it. Use --experiment <new tag> to keep both. Activations extracted from "
                  f"the old build will be rejected as stale.")

    registry = tables["registry"]
    registry.to_csv(out / "prompts.csv", index=False)
    tables["train"].to_csv(out / "train_datasets.csv", index=False)
    tables["test"].to_csv(out / "paired_test.csv", index=False)
    tables["heldout_test"].to_csv(out / "heldout_test.csv", index=False)

    manifest = {
        "experiment": experiment.get_experiment(),
        "registry_sha1": registry_hash(registry),
        "n_prompts": int(len(registry)),
        "base_registry_sha1": registry_hash(registry.iloc[:tables["meta"]["n_base_prompts"]]),
        "n_train_datasets": int(tables["train"]["dataset"].nunique()),
        "rows_per_train_dataset": int(2 * N_TRAIN_PER_POOL),
        "n_test_pairs": int(tables["test"]["pair_id"].nunique()),
        "n_heldout_pairs": int(len(tables["heldout_test"])),
        **tables["meta"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def print_frame_grid(language: str = None) -> None:
    """Print every frame filled with every concept phrase so the grammar can be read in one go."""
    pack = load_language(language or LANGUAGE)
    for fid, frame in enumerate(pack["frames"]):
        print(f"\n--- frame {fid}: {frame}")
        for c in [SINGLE_PHRASE_CONCEPT] + MISOGYNY_SUBTYPES + BENIGN_CONTROL_CONCEPTS + HATEFUL_CONTROL_CONCEPTS:
            print("   ", frame.format(concept=phrase(pack, c)))


def print_report(tables: dict, manifest: dict) -> None:
    reg, test = tables["registry"], tables["test"]
    print("\n" + "=" * 70)
    print(f"  Dataset build report   experiment={manifest['experiment']}  language={manifest['language']}  "
          f"frame_mode={manifest['frame_mode']}  negatives={manifest['negatives']}")
    print("=" * 70)
    print(f"  unique prompts           : {manifest['n_prompts']}")
    print(f"  prompts by kind          : {reg['kind'].value_counts().to_dict()}")
    print(f"  training datasets        : {manifest['n_train_datasets']} x {manifest['rows_per_train_dataset']} rows (interleaved, unshuffled)")
    print(f"  pooled / averaged vectors: built from {len(manifest['train_subtypes'])} subtypes")
    print(f"  held-out subtypes        : {manifest['heldout']}")
    print(f"  paired test (seen)       : {manifest['n_test_pairs']} fresh statements x {len(PAIRED_ARMS) + len(manifest['extra_arms'])} arms")
    if manifest["negatives"] in ("stance", "stance_wide"):
        print(f"  stance negatives         : {int((reg['kind'] == STANCE_KIND).sum())} (same statement as the positive, EGALITARIAN instruction)")
        print(f"  training egalitarian     : {manifest['train_egalitarian_controls']}")
        print(f"  egalitarian test arm     : {manifest['egalitarian_controls']}")
        print(f"  reusable base prompts    : the first {manifest['n_base_prompts']} (identical to the neutral build); "
              f"{manifest['n_prompts'] - manifest['n_base_prompts']} new prompts to extract")
    if manifest["negatives"] == "matched":
        print(f"  matched negatives        : {int((reg['kind'].isin(MATCHED_KINDS)).sum())} (same statement as the positive, control-concept instruction)")
        print(f"  training controls        : {manifest['train_benign_controls']} | {manifest['train_hateful_controls']}")
        print(f"  egalitarian test arm     : {manifest['egalitarian_controls']}")
        print(f"  reusable base prompts    : the first {manifest['n_base_prompts']} (identical to the neutral build); "
              f"{manifest['n_prompts'] - manifest['n_base_prompts']} new prompts to extract")
    print(f"  held-out test (unseen)   : {manifest['n_heldout_pairs']} pairs "
          f"({manifest['n_test_pairs']} statements x {len(manifest['heldout'])} phrases)")
    print(f"  registry sha1            : {manifest['registry_sha1'][:12]}...")
    print("\n  example positive         :", repr(reg[reg['kind'] == 'subtype'].iloc[0]['user_prompt']))
    if manifest["negatives"] in ("matched", "stance", "stance_wide"):
        d0 = tables["train"][tables["train"]["dataset"] == SINGLE_NAME].sort_values("row_idx")
        print("  example pair (dataset single__misogyny, row pair 0):")
        print("      +", repr(reg.at[int(d0.iloc[0]["prompt_id"]), "user_prompt"]))
        print("      -", repr(reg.at[int(d0.iloc[1]["prompt_id"]), "user_prompt"]))
    else:
        print("  example negative         :", repr(reg[reg['kind'] == 'neutral'].iloc[0]['user_prompt']))
    print("\n  paired test, pair 0:")
    for _, r in test[test["pair_id"] == 0].iterrows():
        print(f"    {r['arm']:<8}", repr(reg.at[r['prompt_id'], 'user_prompt']))
    print("=" * 70 + "\n")


def main(seed: int = None, language: str = None, frame_mode: str = None, fixed_frame_id: int = None,
         show_frames: bool = False, negatives: str = None) -> dict:
    if show_frames:
        print_frame_grid(language)
    tables = build_all(seed=seed, language=language, frame_mode=frame_mode, fixed_frame_id=fixed_frame_id, negatives=negatives)
    validate(tables)
    manifest = save_all(tables)
    print_report(tables, manifest)
    print(f"[dataset_builder] wrote prompts.csv, train_datasets.csv, paired_test.csv, heldout_test.csv, manifest.json -> {dataset_dir()}")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the misogyny steering datasets.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--language", default=None, help=f"language pack in data/languages/ (default {LANGUAGE})")
    parser.add_argument("--frame-mode", default=None, choices=["random", "fixed"],
                        help="random (default): one random frame per training positive; fixed: one frame for all training positives")
    parser.add_argument("--fixed-frame-id", type=int, default=None, help=f"frame used in fixed mode (default {FIXED_FRAME_ID})")
    parser.add_argument("--negatives", default=None, choices=list(NEGATIVES_CHOICES),
                        help="neutral (default): negatives = the question with no instruction; matched: same statement, control-concept instruction "
                             "(+ egalitarian test arm); stance: same statement, EGALITARIAN instruction = the authors' opposite-stance recipe (+ egalitarian test arm)")
    parser.add_argument("--print-frames", action="store_true", help="print every frame x concept phrase to check the grammar")
    parser.add_argument("--export-fresh", action="store_true", help="(re)derive data/languages/en/fresh_*.txt from the _long files and exit")
    parser.add_argument("--init-language", default=None, metavar="CODE", help="create data/languages/<CODE>/ from the English pack and exit")
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    if args.export_fresh:
        export_fresh_lists()
    elif args.init_language:
        init_language(args.init_language)
    else:
        experiment.apply_experiment_arg(args)
        main(seed=args.seed, language=args.language, frame_mode=args.frame_mode, fixed_frame_id=args.fixed_frame_id,
             show_frames=args.print_frames, negatives=args.negatives)
