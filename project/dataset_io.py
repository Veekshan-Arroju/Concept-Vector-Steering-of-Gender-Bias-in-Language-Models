"""
Read-side helpers for the tables written by dataset_builder.py.

Only pandas is needed here (no torch, no tokenizer), so tests and analysis scripts can use it freely.
Everything is read from dataset/processed/<experiment>/ (see experiment.py). The concept lists (which subtypes
exist, which are held out) come from manifest.json, NOT from code constants, so a build with extra concepts or a
different held-out set is followed automatically by every downstream script.

Terminology
-----------
registry     every unique prompt (prompts.csv); activations are stored per prompt_id
train set    one of the interleaved 400-row datasets (train_datasets.csv)
eval sets    "main"    : fresh statements with a SEEN subtype phrase  (neutral = 0 vs misogyny = 1)
             "heldout" : fresh statements with each HELD-OUT subtype phrase (unseen by P and A)
arms         the same fresh statement rendered 4 ways (neutral / subtype / benign / hateful)
"""

import json

import pandas as pd

from experiment import dataset_dir
from raw_data import subtype_name

ARMS = ("neutral", "subtype", "benign", "hateful")   # always present; the matched experiment adds "egalitarian" (see arm_names)
EVAL_SETS = ("main", "heldout")

_CACHE = {}


def _path(name: str):
    path = dataset_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python dataset_builder.py` (with the same --experiment) first.")
    return path


def _read_csv(name: str) -> pd.DataFrame:
    path = _path(name)
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _CACHE:
        _CACHE[key] = pd.read_csv(path, keep_default_na=False)
    return _CACHE[key].copy()


def load_registry() -> pd.DataFrame:
    reg = _read_csv("prompts.csv")
    assert (reg["prompt_id"].values == range(len(reg))).all(), "prompts.csv must be ordered by prompt_id"
    return reg


def load_manifest() -> dict:
    path = _path("manifest.json")
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _CACHE:
        with open(path, encoding="utf-8") as f:
            _CACHE[key] = json.load(f)
    return json.loads(json.dumps(_CACHE[key]))


def load_train_table() -> pd.DataFrame:
    return _read_csv("train_datasets.csv")


def load_test_table() -> pd.DataFrame:
    return _read_csv("paired_test.csv")


def load_heldout_table() -> pd.DataFrame:
    return _read_csv("heldout_test.csv")


# -----------------------------------------------------------------------
# Concept lists (from the manifest)
# -----------------------------------------------------------------------
def subtype_concepts() -> list:
    return list(load_manifest()["subtypes"])


def train_subtype_concepts() -> list:
    """Subtypes P and A are built from (everything except the held-out ones)."""
    return list(load_manifest()["train_subtypes"])


def heldout_concepts() -> list:
    return list(load_manifest()["heldout"])


def subtype_vector_names(which: str = "all") -> list:
    """Vector names of the per-subtype vectors: which = 'all' | 'train' | 'heldout'."""
    pick = {"all": subtype_concepts, "train": train_subtype_concepts, "heldout": heldout_concepts}[which]
    return [subtype_name(c) for c in pick()]


def arm_names() -> tuple:
    """Arms of the paired test in this experiment: the four base arms plus any extra arm (e.g. 'egalitarian')."""
    return ARMS + tuple(load_manifest().get("extra_arms", []))


def concept_for_vector(vector_name: str):
    """The subtype concept id behind a per-subtype vector, or None for S / P / A."""
    return {subtype_name(c): c for c in subtype_concepts()}.get(vector_name)


def list_train_datasets() -> list:
    return list(dict.fromkeys(load_train_table()["dataset"].tolist()))


def get_train_set(dataset_name: str) -> tuple:
    """
    (prompt_ids, labels) of one training dataset, in the stored (interleaved, unshuffled) order.
    Do not re-order: split_train_val relies on this order.
    """
    table = load_train_table()
    d = table[table["dataset"] == dataset_name].sort_values("row_idx")
    if d.empty:
        raise KeyError(f"Unknown training dataset {dataset_name!r}. Known: {list_train_datasets()}")
    return d["prompt_id"].astype(int).tolist(), d["label"].astype(float).tolist()


# -----------------------------------------------------------------------
# Evaluation sets
# -----------------------------------------------------------------------
def get_arm_table(set_name: str = "main", concept: str = None) -> pd.DataFrame:
    """
    One row per (fresh statement, misogyny phrase) with the prompt_id of each arm.
    columns: pair_id, pool, subtype_concept, neutral, subtype, benign, hateful (+ egalitarian in the matched experiment)

    set_name "main"   : each fresh statement once, with one SEEN subtype (the paired test).
    set_name "heldout": each fresh statement once per HELD-OUT subtype (the unseen-phrase test). The neutral / benign /
                        hateful ids repeat across the phrases of one statement, which is what makes it a paired design.
    `concept` keeps only the rows whose misogyny arm used that subtype.
    """
    if set_name not in EVAL_SETS:
        raise KeyError(f"Unknown eval set {set_name!r}; use one of {EVAL_SETS}")
    t = load_test_table()
    arms = arm_names()
    wide = t.pivot(index="pair_id", columns="arm", values="prompt_id")[list(arms)].astype(int)
    if set_name == "main":
        meta = t[t["arm"] == "subtype"].set_index("pair_id")[["pool", "concept"]].rename(columns={"concept": "subtype_concept"})
        out = meta.join(wide).reset_index()
    else:
        h = load_heldout_table()
        order = {c: i for i, c in enumerate(heldout_concepts())}
        h = h.assign(_o=h["concept"].map(order)).sort_values(["pair_id", "_o"])
        base = wide.drop(columns=["subtype"]).reset_index()
        out = (h.rename(columns={"concept": "subtype_concept", "prompt_id": "subtype"})
                 .merge(base, on="pair_id")[["pair_id", "pool", "subtype_concept", *arms]])
    out = out.sort_values("pair_id", kind="stable").reset_index(drop=True)
    if concept is not None:
        out = out[out["subtype_concept"] == concept].reset_index(drop=True)
        if out.empty:
            raise KeyError(f"No pairs in eval set {set_name!r} use the subtype {concept!r}")
    return out


def get_eval_set(set_name: str = "main", concept: str = None) -> tuple:
    """
    Labelled evaluation set: for every pair, [neutral prompt (label 0), misogyny prompt (label 1)].
    Exactly 1:1 balanced, so AUROC / accuracy code works unchanged. Returns (prompt_ids, labels).
    """
    arms = get_arm_table(set_name, concept)
    ids, labels = [], []
    for _, r in arms.iterrows():
        ids += [int(r["neutral"]), int(r["subtype"])]
        labels += [0.0, 1.0]
    return ids, labels


# -----------------------------------------------------------------------
# Inner train/val split (moved here from linear_probe.py, logic unchanged)
#    Adapted from paper's utils.split_indices() (random_split=False default)
# -----------------------------------------------------------------------
def split_train_val(
    N: int,
    frac: float = 0.2,
    max_val_count: int = 256,
) -> tuple:
    """
    Carve an inner validation fold from N training samples for λ-tuning.

    Directly adapted from utils.split_indices() in the paper.
    Uses a deterministic (non-random) split so results are reproducible
    without extra seeding — the first n_train items become the inner training
    set, the remainder form the validation set.

    The inner-val set is used ONLY to select the ridge-regression λ.
    It never influences which test predictions are computed.

    Args:
        N             : Total number of training samples available
        frac          : Fraction to hold out for validation (paper default 0.2)
        max_val_count : Hard cap so we never waste too many samples on val
                        (paper default 256)

    Returns: (train_indices, val_indices) — both are range objects
    """
    # How many samples to hold out — paper's formula exactly
    n_val   = min(int(frac * N), max_val_count)
    n_train = N - n_val
    # Paper bumps n_train to next even number (supports paired contrastive data)
    n_train = n_train + (n_train % 2)

    # Deterministic slice: first n_train -> inner-train, rest -> inner-val
    train_indices = range(n_train)
    val_indices   = range(n_train, N)  # may be slightly < n_val if bumped

    return train_indices, val_indices
