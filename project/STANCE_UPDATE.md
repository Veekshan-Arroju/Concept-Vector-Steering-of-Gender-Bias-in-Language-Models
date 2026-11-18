# STANCE_UPDATE: what changed and why

Files in this update
- raw_data.py                       TRAIN_EGALITARIAN_CONCEPTS (6 concepts, disjoint from the test egalitarian arm and from every misogyny phrase)
- data/languages/en/language.json   phrases for those concepts (add them to every other language pack too)
- dataset_builder.py                `--negatives stance` (build, validate, report); neutral / matched builds are unchanged
- activation_extraction.py          message text only (`--extend-from main` works for matched and stance)
- steering_eval.py                  default sweep +-0.8..0; regime warnings; nothing else changed
- coef_calibration.py (new)         collapse threshold, working window, regime warnings, directionality (no torch)
- calibrate_steering.py (new)       sweeps |epsilon| per sign, finds the collapse point, optional random-direction control and judged evaluation
- directionality_check.py (new)     reads a saved stance_logit_test pickle; no GPU
- tests/test_stance_and_calibration.py (new)   13 tests, CPU, no torch; with tests/test_dataset.py: 38 passed

Verified here: dataset build/validation (stance, matched, neutral), registry prefix identity with the neutral build (so --extend-from main is valid),
the directionality check on your saved v5 stance pickles, the degeneration heuristic on the texts of your old screenshots, all unit tests above.
NOT verified here (no GPU, no model weights): calibrate_steering.py and the steering_eval.py change were syntax- and import-checked only;
no RFM vector was trained on the stance data; nothing says yet that the stance vector steers.

Facts behind the diagnosis
- rfm.py is byte-identical in v3 and v5; hook_model, sign orientation and the additive perturbation are the same.
- v3 and v5 vectors are all unit norm, and nearly orthogonal at every layer (mean |cos| about 0.04), so they are different directions.
- Paper, Llama-3.1-8B, all blocks, unit vectors: epsilon 0.55-0.8; above that "nonsensical, repetitive outputs". v3: 0.5 near baseline, 0.7 shifted,
  0.8+ collapsed. v5: layers 23-26 only at +-1..2 (no effect) and +-4..12 (gibberish); all layers at +-2.5..3 (collapse at +2.5, +3).
  Typical hidden-state norm is about 10-35 per layer, so +-2.5..3 is a 10-30 % push per layer.
- v5 pooled vector (matched), stance_logit_test, all layers, relative epsilon: stance shift +5.56 at +0.1 and +5.14 at -0.1; +7.27 at both +0.4 and -0.4.
  Same sign for both signs = disruption, not a direction (directionality_check.py: 0 of 3 pairs opposite). At +-0.1 the effect is larger than
  random directions (z about 2.7-3.0, the minimum p with 20 random directions is 0.05); at +-0.4 it equals them (p 0.48 / 0.86).

Added after reading the handoff document (2026-10-28)
- Units. The all-layer logit test used coefficients relative to each layer's typical hidden-state norm at the extraction token (149, 35, 28, ... median 9.5
  excluding layer -1, 1.4 at -31). Relative 0.1 / 0.2 / 0.4 is therefore a median of about 0.95 / 1.9 / 3.8 absolute (up to 3.5 / 7 / 14 at layer -2),
  against the paper's 0.55-0.8 absolute with unit vectors on all blocks. All logit-test configurations (layers -23..-26 at +-1..3, all layers at relative
  +-0.1..0.4, layers -7..-10 at +-2..6) were at or above the magnitude where generation collapses, so "same sign for both signs" is established there, not
  at the paper's scale. Step 0 in README.md re-tests the existing matched vector at absolute +-0.2..0.8 on all layers; it costs no retraining.
- Checked and NOT supported: that v5's vectors are incoherent across layers. Adjacent-layer cosine of the 31-layer vectors is 0.68 (v5 matched pooled) vs
  0.73 (v3), about 0.9 for layers -2..-13 in both, against 0.01 for random vectors. Both are coherent; they are different directions (cos old/new about 0.04).
- Stance vs misandry negatives (the handoff's open decision, matched2): they answer different questions. Egalitarian negatives give "hostile stance toward women
  minus pro-women stance" (the paper's recipe for stance concepts, and the v3 contrast); misandry negatives give "women as target minus men as target" with the
  hostility cancelled, which addresses the handoff's finding that the monitor fires on misandry. A mixed negative set is possible but is not built here.
