"""
Static content for the misogyny steering dataset (v5).

Design follows the authors' benchmark datasets (Beaglehole et al., arXiv:2502.03708, `pca_*_dataset`
builders in utils.py):

    positive (label 1) : <instruction naming the concept> + <question tail> + statement  (class_0 pool)
    negative (label 0) :                                   <question tail> + statement  (class_1 pool)

What lives where
----------------
* HERE (code):  the MASTER lists of concept ids (subtypes, held-out subtypes, controls) and the vector names.
  Concept ids are the English phrases and never change between languages.
* data/languages/<lang>/language.json:  everything that is language-specific, i.e. the instruction frames, the
  question tail, and the phrase used for every concept id in that language.
* data/languages/<lang>/{class_0,class_1,fresh_0,fresh_1}.txt:  the statements (line-aligned across languages).

To add a concept: add its id to MISOGYNY_SUBTYPES below, add a phrase for it in every language.json, rebuild.
To add a language: `python dataset_builder.py --init-language hi`, translate the files, build with `--language hi`.

Nothing here touches pandas, torch or a tokenizer.
"""

import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
LANGUAGES_DIR = PROJECT_ROOT / "data" / "languages"


# -----------------------------------------------------------------------
# 1. Master concept lists (canonical ids = English phrases)
# -----------------------------------------------------------------------
MISOGYNY_SUBTYPES = [
    # ambivalent sexism and hierarchy
    "hostile sexism",
    "benevolent sexism",
    "gender essentialism",
    "male superiority",
    # entitlement and control
    "sexual entitlement",
    "entitlement to women's labor",
    "policing women's behavior",
    "domestic subordination of women",
    # shaming and blame
    "slut-shaming",
    "purity culture",
    "victim-blaming",
    "sexual double standards",
    # reduction
    "sexual objectification",
    "infantilization of women",
    "dehumanization of women",
    # contempt and exclusion
    "contempt for career women",
    "anti-feminism",
    "doubting women's competence",
    "distrust of women",
    # online ideology
    "incel ideology",
]

# Withheld from the pooled vector (P) and the averaged vector (A) so that S, P and A can be compared on
# wordings none of them has seen. Their own per-subtype vectors are still trained (reference / "oracle").
# One from each of four different groups above.
HELD_OUT_SUBTYPES = [
    "slut-shaming",
    "dehumanization of women",
    "doubting women's competence",
    "incel ideology",
]

SINGLE_PHRASE_CONCEPT = "misogyny"   # the broad single phrase (the "S" vector)

# TEST-ONLY control concepts (never used to train a vector).
BENIGN_CONTROL_CONCEPTS = ["minimalism", "stoicism", "environmentalism"]
HATEFUL_CONTROL_CONCEPTS = ["xenophobia", "racial supremacy", "ableism"]

# Used only by the "matched" experiment (dataset_builder.py --negatives matched):
# * TRAIN controls are the concepts named in the NEGATIVE training prompts (same statement, same kind of instruction as the
#   positive, but a control concept). They are deliberately DIFFERENT from the test controls above, so a vector cannot learn
#   the test controls by name. The hateful ones avoid gender and sexuality, so "other hatred" stays free of the misogyny topic.
# * EGALITARIAN controls are a TEST-ONLY fifth arm: same topic as misogyny (women) with the OPPOSITE stance. A vector for a
#   misogynistic stance should not fire on them; a vector that only detects "about women" will.
TRAIN_BENIGN_CONTROL_CONCEPTS = ["gardening", "chess", "classical music"]
TRAIN_HATEFUL_CONTROL_CONCEPTS = ["religious bigotry", "classism", "ageism"]
EGALITARIAN_CONTROL_CONCEPTS = ["gender equality", "respect for women", "feminism"]

# Used only by the "stance" experiment (dataset_builder.py --negatives stance), the analogue of the authors' liberal-vs-conservative
# dataset (arXiv:2502.03708, SI B.2): the NEGATIVE of each pair is the same statement under an instruction that names the OPPOSITE
# stance (egalitarian), so the instruction format cancels and the learned direction is "misogynistic stance minus egalitarian
# stance". These TRAIN concepts are different from the TEST egalitarian arm above (nothing is learned by name), they stay on the
# topic of women (so the contrast is stance, not topic), and none of them contains a misogyny phrase or a test phrase.
TRAIN_EGALITARIAN_CONCEPTS = ["anti-sexism", "women's empowerment", "women's rights", "equal pay for women",
                              "women's autonomy", "women's education"]

# Used only by --negatives stance_wide (pool = TRAIN_EGALITARIAN_CONCEPTS + these). Motivation: the v3 vector, whose NEGATIVE sign reliably
# pushed generations toward respect / dignity / equal-treatment wording on prompts it was not aligned with, was trained against 176
# egalitarian concepts, 30 of them about respect or dignity (Respect for ..., Mutual respect, Women's dignity); the v5 stance pool has six and none
# is about respect. These are generic respect / dignity names taken from that list. Left out on purpose: names that match the topic of an
# evaluation prompt (working women, political / professional / academic respect, homemakers, married / single / childfree / young women, ...)
# and anything containing a held-out test phrase (respect for women, gender equality, feminism).
TRAIN_EGALITARIAN_EXTRA_CONCEPTS = ["mutual respect", "women's dignity", "respectful relationships", "online respect", "respect for girls",
                                    "respect for femininity", "respect for autonomy", "respect for consent", "respect for boundaries",
                                    "respect for older women", "respect for survivors", "dignity for all genders", "gender-neutral respect",
                                    "medical respect", "legal respect"]

assert len(set(MISOGYNY_SUBTYPES)) == len(MISOGYNY_SUBTYPES)
assert set(HELD_OUT_SUBTYPES) <= set(MISOGYNY_SUBTYPES), "held-out subtypes must be subtypes"
_groups = [MISOGYNY_SUBTYPES, [SINGLE_PHRASE_CONCEPT], BENIGN_CONTROL_CONCEPTS, HATEFUL_CONTROL_CONCEPTS,
           TRAIN_BENIGN_CONTROL_CONCEPTS, TRAIN_HATEFUL_CONTROL_CONCEPTS, EGALITARIAN_CONTROL_CONCEPTS,
           TRAIN_EGALITARIAN_CONCEPTS, TRAIN_EGALITARIAN_EXTRA_CONCEPTS]
assert len({c for g in _groups for c in g}) == sum(len(g) for g in _groups), "concept groups must not overlap (no train/test control leakage)"


def all_concept_ids() -> list:
    return [c for g in _groups for c in g]


# -----------------------------------------------------------------------
# 2. Language packs
# -----------------------------------------------------------------------
def _read_pack(language: str) -> str:
    path = LANGUAGES_DIR / language / "language.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No language pack at {path}. For a new language run `python dataset_builder.py --init-language {language}` "
            f"and translate the files it creates."
        )
    return path.read_text(encoding="utf-8")


def load_language(language: str = "en", concept_ids: list = None) -> dict:
    """
    Load and validate data/languages/<language>/language.json.
    Returns a dict with: language, frames, question_tail, instruction_separator, phrases.
    `concept_ids` defaults to the master lists above; pass your own to build with a different concept set.
    """
    pack = json.loads(_read_pack(language))
    if pack.get("language") != language:
        raise ValueError(f"language.json says language={pack.get('language')!r} but lives in the {language!r} folder")
    frames = pack.get("frames") or []
    if not frames or any("{concept}" not in f for f in frames):
        raise ValueError(f"{language}: 'frames' must be a non-empty list and every frame needs a {{concept}} slot")
    if "{statement}" not in pack.get("question_tail", ""):
        raise ValueError(f"{language}: 'question_tail' needs a {{statement}} slot")
    pack.setdefault("instruction_separator", "  ")
    missing = [c for c in (concept_ids or all_concept_ids()) if c not in pack.get("phrases", {})]
    if missing:
        raise ValueError(f"{language}: language.json has no phrase for concept id(s): {missing}")
    return pack


def phrase(pack: dict, concept_id: str) -> str:
    return pack["phrases"][concept_id]


def build_positive_prompt(pack: dict, frame_id: int, concept_id: str, statement: str) -> str:
    """Instruction (frame filled with the concept phrase) + question tail + statement."""
    instruction = pack["frames"][frame_id].format(concept=phrase(pack, concept_id))
    return instruction + pack["instruction_separator"] + pack["question_tail"].format(statement=statement)


def build_neutral_prompt(pack: dict, statement: str) -> str:
    """No concept at all: just the question tail + statement."""
    return pack["question_tail"].format(statement=statement)


# -----------------------------------------------------------------------
# 3. Naming helpers (dataset / vector names)
# -----------------------------------------------------------------------
def slugify(text: str) -> str:
    """'entitlement to women's labor' -> 'entitlement_to_womens_labor'."""
    s = text.lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


SUBTYPE_PREFIX = "subtype__"
SINGLE_NAME = "single__misogyny"
POOLED_NAME = "pooled__misogyny"          # trained on the NON-held-out subtypes only
AVERAGED_NAME = "averaged__misogyny"      # derived: mean of the NON-held-out subtype vectors (the "A" of the experiment)
AVERAGED_ALL_NAME = "averaged_all__misogyny"   # derived: mean of ALL subtype vectors (for final use)


def subtype_name(concept: str) -> str:
    return SUBTYPE_PREFIX + slugify(concept)


SUBTYPE_NAMES = [subtype_name(c) for c in MISOGYNY_SUBTYPES]
assert len(set(SUBTYPE_NAMES)) == len(MISOGYNY_SUBTYPES), "subtype slugs must be unique"


def train_dataset_for_vector(vector_name: str) -> str:
    """
    Which training dataset a vector's inner-train / inner-val split comes from (used to calibrate probes and
    fit monitors). The averaged vectors have no dataset of their own, so they borrow the pooled one.
    """
    if vector_name in (AVERAGED_NAME, AVERAGED_ALL_NAME):
        return POOLED_NAME
    return vector_name
