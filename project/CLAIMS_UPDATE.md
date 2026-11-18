# CLAIMS_UPDATE: belief claims vs counter-claims (one factor: what the concept is)

New files only (nothing overwritten): claims_dataset.py, tests/test_claims_dataset.py, this note.

Why: across v3 and every v5 variant no vector produced coherent misogynistic output on prompts outside its training topics. v3's one coherent
sexist assertion came on a prompt whose topic was a training concept. Category labels ("hostile sexism") may encode a topic or a persona style,
not the content of an assertion. Here the concept is a short claim and the negative is the topic-matched counter-claim.

Design: 20 pairs (stereotyped claim / egalitarian counter-claim on the SAME topic), topics DISJOINT from the 12 evaluation prompts (tests check the
shared words are only function words and "women"); 200 positives and 200 negatives on the same class_0 statements; frames drawn independently.
The dataset is named pooled__misogyny so every downstream script works unchanged. Raw read (no chat template, no assistant header) as in stance_raw.

Run order (server):
  python claims_dataset.py --experiment claims
  python raw_read_train.py --source-experiment claims --dataset pooled__misogyny --experiment claims_raw --compare old_v3 stance_raw stance_wide_raw
  python calibrate_steering.py --vector pooled__misogyny --experiment claims_raw --n-prompts 12 --prompt-offset 0 --max-coef 0.9 --tag all12

Prediction written BEFORE the run (so it can fail): judged on the positive side at |eps| 0.4-0.8, at least 10 % of the answers are misogynistic AND coherent
(stance_wide_raw / stance_raw: about 0 %), and the effect is not confined to one prompt (at least 3 of the 12 prompts have such an answer).
If that fails, the claims pool does not induce either; the remaining factors are the read position inside an ANSWER (activations read while the model
writes a sexist answer, not while it reads an instruction) and the steering site.
