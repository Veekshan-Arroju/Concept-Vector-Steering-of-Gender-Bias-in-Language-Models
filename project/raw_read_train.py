"""
Read-position ablation: ONE factor changed, everything else as in v5.

v5 reads activations at the ASSISTANT HEADER that follows the chat-templated prompt. After an instruction like "Take on the role of
someone who endorses hostile sexism" that state encodes what the model is about to DO (decline / push back), and steering with the
resulting vector produced refusals ("I can't provide information on how to discriminate against women"). The v3 run read the last
token of the RAW prompt text (no chat template, no header, no BOS): before any response is planned, and its positive direction
produced sexist assertions, not refusals.

This script keeps the dataset (same prompts, same labels, same order, same RFM, same sign orientation, same 31 layers) and changes
only the position and format of the read:

    python raw_read_train.py --source-experiment stance --dataset pooled__misogyny --experiment stance_raw --limit 8   # dry run
    python raw_read_train.py --source-experiment stance --dataset pooled__misogyny --experiment stance_raw             # extract + train + save

It extracts ONLY the prompts of that one training dataset (400 for pooled__misogyny), does NOT touch the stored activations of any
experiment, and saves the vector to concept_vectors/<model_slug>/<--experiment>/<dataset>.pkl, so afterwards
    python calibrate_steering.py --vector pooled__misogyny --experiment stance_raw
    python stance_logit_test.py  --vector pooled__misogyny --experiment stance_raw ...
work unchanged. If an `old_v3` vector and the source experiment's header-read vector exist, the cosines to the new vector are printed.
"""

import argparse
import gc
import os
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from dotenv import load_dotenv

import experiment
from activation_extraction import DEFAULT_BATCH_SIZE, DEFAULT_CHUNK_SIZE, _extract_chunks, select_layers
from concept_vector_training import learn_concept_vectors, load_concept_vectors, save_concept_vectors, vector_exists
from dataset_io import get_train_set, load_registry
from model_utils import add_model_args, load_causal_lm
from raw_read_utils import dry_run_indices, label_summary, raw_prompts

load_dotenv()


def _tail(tokenizer, text: str, n: int = 6) -> list:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode([i]) for i in ids[-n:]]


def _cosine_report(model_id: str, dataset: str, new_dirs: dict, other_experiments: list) -> None:
    for exp_name in other_experiments:
        experiment.set_experiment(exp_name)
        try:
            if not vector_exists(model_id, dataset):
                print(f"  [cosine] {exp_name}: no {dataset} vector, skipped")
                continue
            other, _, method = load_concept_vectors(model_id, dataset)
        except Exception as e:   # a report must never kill a finished training run
            print(f"  [cosine] {exp_name}: could not load ({e})")
            continue
        layers = [l for l in sorted(new_dirs, reverse=True) if l in other]
        cs = [float(torch.nn.functional.cosine_similarity(new_dirs[l].reshape(-1).float(), other[l].reshape(-1).float(), dim=0)) for l in layers]
        pick = [l for l in (-2, -8, -12, -16, -20, -24, -28) if l in layers]
        print(f"  [cosine] new vector vs {exp_name} ({method}): mean {sum(cs) / len(cs):+.3f}; "
              + " ".join(f"L{l}:{cs[layers.index(l)]:+.2f}" for l in pick))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a concept vector from activations read at the end of the RAW prompt (read-position ablation).")
    ap.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    ap.add_argument("--source-experiment", required=True, help="experiment that holds the dataset (prompts.csv, train_datasets.csv), e.g. stance")
    ap.add_argument("--dataset", default="pooled__misogyny", help="training dataset / vector name")
    ap.add_argument("--experiment", default=None, help="experiment the vector is saved under (default: <source>_raw)")
    ap.add_argument("--method", default="rfm", choices=["rfm", "mean_difference"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--add-bos", action="store_true", help="prepend <|begin_of_text|> (v3 did NOT; default off = the v3 setting)")
    ap.add_argument("--limit", type=int, default=None, help="DRY RUN: extract only N prompts, print positions/shapes/norms, train and save nothing")
    ap.add_argument("--compare", nargs="*", default=["old_v3"], help="experiments whose same-named vector to compare with (default: old_v3 and the source experiment)")
    add_model_args(ap)
    args = ap.parse_args()
    if not args.model_id:
        raise ValueError("pass --model-id or set MODEL_ID in .env")
    out_experiment = args.experiment or f"{args.source_experiment}_raw"
    if out_experiment == args.source_experiment:
        raise ValueError("--experiment must differ from --source-experiment (this script never overwrites an existing experiment's vectors)")

    # ---- the dataset: same prompts / labels / order as the normal pipeline ----
    experiment.set_experiment(args.source_experiment)
    ids, labels = get_train_set(args.dataset)
    summary = label_summary(labels)
    prompts = raw_prompts(load_registry(), ids, bos_text="<|begin_of_text|>" if args.add_bos else None)
    print(f"[raw_read] dataset {args.dataset} of experiment {args.source_experiment}: {summary}")
    if not summary["interleaved"]:
        raise ValueError("training rows are not interleaved pos/neg: refusing to continue (split_train_val relies on it)")
    print(f"[raw_read] example RAW prompt (positive row):\n{prompts[0]!r}\n[raw_read] example RAW prompt (negative row):\n{prompts[1]!r}\n")

    model, tokenizer = load_causal_lm(args.model_id, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    layers = select_layers(model)
    print(f"[raw_read] token that will be READ (last tokens of the raw prompt): pos {_tail(tokenizer, prompts[0])} | neg {_tail(tokenizer, prompts[1])}")
    print(f"[raw_read] reading {len(layers)} layers ({layers[0]} ... {layers[-1]}); chat template: NO | assistant header: NO | BOS: {'yes' if args.add_bos else 'no'}")

    if args.limit is not None:
        idx = dry_run_indices(len(prompts), args.limit)
        per_layer, elapsed = _extract_chunks([prompts[i] for i in idx], model, tokenizer, layers, args.batch_size, args.chunk_size)
        sample = per_layer[layers[0]] if -16 not in per_layer else per_layer[-16]
        print(f"[raw_read] DRY RUN: {len(idx)} prompts, activations {tuple(sample.shape)}, row norms at layer -16: "
              f"{[round(float(x), 1) for x in sample.float().norm(dim=1)[:6]]}; {elapsed / len(idx):.2f} s per prompt. Nothing trained or saved.")
        return

    t0 = time.perf_counter()
    per_layer, elapsed = _extract_chunks(prompts, model, tokenizer, layers, args.batch_size, args.chunk_size)
    print(f"[raw_read] extracted {len(prompts)} prompts in {elapsed / 60:.1f} min")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- the same learner as concept_vector_training.py (RFM, inner split, sign orientation on the full training set) ----
    directions, signs = learn_concept_vectors(per_layer, labels, method=args.method)
    experiment.set_experiment(out_experiment)
    save_concept_vectors(directions, signs, args.model_id, args.method, vector_name=args.dataset,
                         extra={"dataset": args.dataset, "layers": sorted(directions.keys()), "read_position": "raw_end_of_prompt",
                                "chat_template": False, "bos": bool(args.add_bos), "source_experiment": args.source_experiment})
    print(f"[raw_read] done in {(time.perf_counter() - t0) / 60:.1f} min -> experiment {out_experiment!r}")
    _cosine_report(args.model_id, args.dataset, directions, list(dict.fromkeys(list(args.compare) + [args.source_experiment])))


if __name__ == "__main__":
    main()
