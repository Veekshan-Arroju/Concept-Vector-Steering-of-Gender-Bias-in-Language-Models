# Results index

Every file was produced on the first server (Llama-3.1-8B-Instruct, fp16, 2 x NVIDIA L4) and copied unchanged. The file names inside `judged_sonnet`, `judged_haiku` and `misogyny_verdicts` are the names the analysis scripts wrote, and the scripts read the labels back from them. Counts and ranges below were computed from the files themselves.

**Columns.** Generation files: `direction, coef, prompt, response, degenerate, refusal_heuristic`. `direction` is `learned` or `random<seed>`; `coef` is the signed epsilon (+ toward the misogynistic training class); `degenerate` is the repetition heuristic. Judged files add the verdicts and `judge_explanation`: `misogynistic, coherent, refusal_or_hedge` (misogyny judge) or `stereotype, demeaning, coherent` (stereotype judge).

## 1. Stem benchmark, the main result (`stem_benchmark/`)

Generation-only steering, 80 stems (40 about women, 40 matched about men), 80 new tokens. Sonnet is the main judge; Haiku is the second judge.

**Final summary tables**

| File | Rows | Content |
| --- | --- | --- |
| `stem_benchmark/stem_rates_final.csv` | 68 | Stereotype rates by vector, direction, group, sign and strength band (baseline 0.2; steered 0.4 to 0.8) with 95% Wilson intervals |
| `stem_benchmark/stem_contrasts_final.csv` | 25 | Paired per-stem contrasts (+ minus -) with permutation and sign-test p-values: 3 learned vectors x 2 groups, and 19 random directions on women stems |
| `stem_benchmark/stem_rank.csv` | 3 | Randomization test: each learned vector against the 19 random directions (women stems) |
| `stem_benchmark/first_pass_stem_rates.csv` | 28 | FIRST full judging run, before the 16 extra random directions existed (one random direction); superseded, kept because the report quotes it in the pre-registration history |
| `stem_benchmark/first_pass_stem_contrasts.csv` | 7 | same remark |

**Per-answer files, Sonnet** (`stem_benchmark/judged_sonnet/`, 22 files, 6,720 answers). `claims_raw__stems__stems.csv`, `old_v3__stems__stems.csv`, `stance_raw__stems__stems.csv`: the learned vectors, 640 answers each (80 stems x strengths 0.2 to 0.8 at both signs). `claims_raw__stems_random__stems.csv` (seed 0), `..._random1` and `..._random2`: random directions on the 40 women stems with strengths 0.2 to 0.8 (320 answers each). `..._random3` to `..._random18`: random directions with strengths 0.4, 0.6 and 0.8 (240 answers each).

| File | Answers | Prompts | Epsilon (values) | Direction |
| --- | --- | --- | --- | --- |
| `old_v3__stems__stems.csv` | 640 | 80 | -0.8 to +0.8 (8) | learned |
| `claims_raw__stems__stems.csv` | 640 | 80 | -0.8 to +0.8 (8) | learned |
| `stance_raw__stems__stems.csv` | 640 | 80 | -0.8 to +0.8 (8) | learned |
| `claims_raw__stems_random__stems.csv` | 320 | 40 | -0.8 to +0.8 (8) | random0 |
| `claims_raw__stems_random1__stems.csv` | 320 | 40 | -0.8 to +0.8 (8) | random1 |
| `claims_raw__stems_random2__stems.csv` | 320 | 40 | -0.8 to +0.8 (8) | random2 |
| `claims_raw__stems_random3__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random3 |
| `claims_raw__stems_random4__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random4 |
| `claims_raw__stems_random5__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random5 |
| `claims_raw__stems_random6__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random6 |
| `claims_raw__stems_random7__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random7 |
| `claims_raw__stems_random8__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random8 |
| `claims_raw__stems_random9__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random9 |
| `claims_raw__stems_random10__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random10 |
| `claims_raw__stems_random11__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random11 |
| `claims_raw__stems_random12__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random12 |
| `claims_raw__stems_random13__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random13 |
| `claims_raw__stems_random14__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random14 |
| `claims_raw__stems_random15__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random15 |
| `claims_raw__stems_random16__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random16 |
| `claims_raw__stems_random17__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random17 |
| `claims_raw__stems_random18__stems.csv` | 240 | 40 | -0.8 to +0.8 (6) | random18 |

**Per-answer files, Haiku** (`stem_benchmark/judged_haiku/`, 4 files, 2,240 answers): `claims_raw__stems__stems.csv`, `old_v3__stems__stems.csv`, `stance_raw__stems__stems.csv` (640 each) and `claims_raw__stems_random__stems.csv` (random seed 0, 320). A few answers have no Haiku verdict (the judge's output could not be parsed); the statistics skip them.

## 2. Misogyny-judge material

| File | Rows | Prompts | Epsilon | Content |
| --- | --- | --- | --- | --- |
| `misogyny_verdicts/claims_raw__all12__judged.csv` | 120 | 12 | -0.90 to +0.90 (10) | same, claims_raw |
| `misogyny_verdicts/old_v3__generated__judged.csv` | 72 | 12 | +0.20 to +1.20 (6) | old v3, generation-only steering, 12 prompts, positive sign; Sonnet misogyny verdicts for every answer |
| `misogyny_verdicts/old_v3__offtopic__judged.csv` | 80 | 8 | -0.90 to +0.90 (10) | Sonnet misogyny verdicts for the old v3 off-topic sweep (prompts 4-11), only |epsilon| 0.2, 0.5, 0.7, 0.8, 0.9 were judged |
| `misogyny_verdicts/stance__offtopic__judged.csv` | 80 | 8 | -0.90 to +0.90 (10) | same, stance (header read) |
| `misogyny_verdicts/stance_raw__generated__judged.csv` | 84 | 12 | +0.20 to +1.40 (7) | same, stance_raw |
| `misogyny_verdicts/stance_raw__offtopic__judged.csv` | 80 | 8 | -0.90 to +0.90 (10) | same, stance_raw |
| `misogyny_verdicts/stance_wide_raw__all12__judged.csv` | 120 | 12 | -0.90 to +0.90 (10) | Sonnet misogyny verdicts for the stance_wide_raw sweep (prompts 0-11), same five strengths |
| `opinion_probes/judged_claims_raw_opinion.csv` | 72 | 12 | +0.20 to +1.20 (6) | claims_raw, POSITIVE sign; same design |
| `opinion_probes/judged_claims_raw_opinion_neg.csv` | 72 | 12 | -1.20 to -0.20 (6) | claims_raw, NEGATIVE sign; same design |
| `opinion_probes/judged_old_v3_opinion.csv` | 72 | 12 | +0.20 to +1.20 (6) | old v3, POSITIVE sign; 12 opinion and completion prompts; generation-only steering; Sonnet misogyny judge |
| `opinion_probes/judged_old_v3_opinion_neg.csv` | 72 | 12 | -1.20 to -0.20 (6) | old v3, NEGATIVE sign; same design |
| `opinion_probes/judged_stance_raw_opinion.csv` | 72 | 12 | +0.20 to +1.20 (6) | stance_raw, POSITIVE sign; same design |
| `opinion_probes/judged_stance_raw_opinion_neg.csv` | 72 | 12 | -1.20 to -0.20 (6) | stance_raw, NEGATIVE sign; same design |

## 3. All-positions sweeps, generations only (`all_positions_sweeps/`)

| File | Rows | Prompts | Epsilon | Directions | Content |
| --- | --- | --- | --- | --- | --- |
| `all_positions_sweeps/all12_claims_raw.csv` | 216 | 12 | -0.90 to +0.90 (18) | learned | claims_raw; prompts 0-11; all-positions; learned only |
| `all_positions_sweeps/all12_stance_wide_raw.csv` | 216 | 12 | -0.90 to +0.90 (18) | learned | stance_wide_raw; prompts 0-11; all-positions; learned only |
| `all_positions_sweeps/calibration_matched.csv` | 280 | 4 | -1.20 to +1.20 (24) | learned, random0, random1 | matched vector (v5, header read, control-concept negatives); prompts 0-3; all-positions steering; learned plus random seeds 0 and 1 |
| `all_positions_sweeps/calibration_oldv3.csv` | 80 | 4 | -1.00 to +1.00 (20) | learned | old v3 vector; prompts 0-3; learned only |
| `all_positions_sweeps/calibration_stance.csv` | 272 | 4 | -1.20 to +1.20 (24) | learned, random0, random1 | stance vector (header read, egalitarian negatives); same design as the matched sweep |
| `all_positions_sweeps/calibration_stance_raw.csv` | 92 | 4 | -1.10 to +1.20 (23) | learned | stance_raw (raw read); prompts 0-3; learned only; the sweep stops after consecutive collapsed strengths |
| `all_positions_sweeps/offtopic_oldv3.csv` | 144 | 8 | -0.90 to +0.90 (18) | learned | old v3; prompts 4-11; all-positions; learned only |
| `all_positions_sweeps/offtopic_stance.csv` | 144 | 8 | -0.90 to +0.90 (18) | learned | stance (header read); prompts 4-11; all-positions; learned only |
| `all_positions_sweeps/offtopic_stance_raw.csv` | 144 | 8 | -0.90 to +0.90 (18) | learned | stance_raw; prompts 4-11; all-positions; learned only |

The verdicts for the off-topic and all12 sweeps are in `misogyny_verdicts/`, where only five strengths per file were judged; the sweeps here contain every answer.

## Not in this folder

Still only on the first server (`/DATA/deeksha/Veekshan/mtp_v5`): the logit-probe pickles (`results/.../stance_logit__*`), the collapse-threshold JSON files (`calibration__*.json`), the vectors (`concept_vectors/`), the activations, and the logs (`logs/`, the source of the run times in the report). The per-answer stem-benchmark files above already contain every generated answer, so the raw `calibration__*stems*.csv` files are not needed.

## Recompute the tables (seconds, CPU only, no API key, no GPU)

From the folder `mtp_v5`, with Python and pandas:

```
python judge_csv.py --from-judged --out /tmp/out ../results/opinion_probes/*.csv
python judge_csv.py --from-judged --out /tmp/out ../results/misogyny_verdicts/*.csv
python stereotype_eval.py --from-judged --out /tmp/out ../results/stem_benchmark/judged_sonnet/*.csv
python stereotype_eval.py --from-judged --out /tmp/out ../results/stem_benchmark/judged_haiku/*.csv
```

Checked against the numbers printed on the server: the Sonnet files reproduce `stem_rates_final.csv` (68 rows), `stem_contrasts_final.csv` (25 rows) and `stem_rank.csv` (3 rows) exactly, including the 19-direction randomization test; the Haiku files reproduce the printed second-judge table (7 rows); the opinion-probe tables (6 files) and the misogyny-verdict tables (7 files) match the printed ones.
