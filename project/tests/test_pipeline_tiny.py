"""
End-to-end smoke test with a tiny RANDOM Llama and a toy tokenizer (no downloads, CPU).

It runs the real code path of every stage inside a throw-away experiment tag ("pytest_pipe"):
    dataset build (with held-out subtypes) -> chat template -> activation extraction -> activation store (+ stale check)
    -> concept vectors (mean-difference for all 22, RFM for one) -> averaged vectors (16-subtype A, 20-subtype A_all) + similarity
    -> linear probe / direction probe / aggregate monitor / four-arm control eval on the SEEN and HELD-OUT sets
    -> compare_vectors (S vs P vs A on unseen phrases, bootstrap CIs)
    -> steering hooks -> steering_eval with a mock judge

The weights are random, so the NUMBERS mean nothing; the test checks wiring, shapes, row alignment, sign conventions,
holdout isolation, file naming/namespacing and that nothing is applied twice.

Run:  python tests/test_pipeline_tiny.py      (or: pytest tests/test_pipeline_tiny.py)
"""
import _path  # noqa: F401
import json
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pytest
import raw_data
import torch

import _tiny
import activation_extraction as ae
import activation_store as store
import anti_misogyny_steering as ams
import compare_vectors as cmp
import concept_vector_training as cvt
import control_eval
import dataset_builder as db
import dataset_io as dio
import direction_projection_probe as dpp
import experiment
import linear_probe as lp
import projection_monitor as pm
import prompting
import steering_eval as se
from raw_data import AVERAGED_ALL_NAME, AVERAGED_NAME, HELD_OUT_SUBTYPES, POOLED_NAME, SINGLE_NAME, subtype_name

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "test/tiny-random-llama"
SLUG = MODEL_ID.replace("/", "_").replace("-", "_")
EXP = "pytest_pipe"
_STATE = {}


def _setup():
    if _STATE:
        return _STATE
    experiment.set_experiment(EXP)
    db.main()                                   # fresh dataset files in dataset/processed/pytest_pipe
    reg = dio.load_registry()
    tok = _tiny.make_tokenizer(reg["user_prompt"].tolist())
    model = _tiny.make_llama(tok)
    _STATE.update(tokenizer=tok, model=model)
    return _STATE


def _cleanup():
    for d in ["activations", "concept_vectors", "probe_results", "direction_probe_results", "monitor_results",
              "control_results", "comparison_results", "pilot_results", "profile_results", "results"]:
        shutil.rmtree(ROOT / d / SLUG, ignore_errors=True)
    for e in (EXP, EXP + "_m", EXP + "_x", EXP + "_fm"):
        shutil.rmtree(ROOT / "dataset" / "processed" / e, ignore_errors=True)
    experiment.set_experiment("main")


# -----------------------------------------------------------------------
def test_chat_template_position_and_single_bos():
    s = _setup()
    tok = s["tokenizer"]
    formatted = prompting.format_chat_prompt(tok, "hello there")
    assert formatted.startswith("<|begin_of_text|>") and formatted.endswith("<|end_header_id|>")   # .strip() removed the "\n\n"
    ids = tok(formatted, add_special_tokens=False)["input_ids"]
    assert ids.count(tok.bos_token_id) == 1 and ids[0] == tok.bos_token_id
    assert prompting.describe_extraction_position(tok, "hello there")[-1] == "<|end_header_id|>"
    assert ams.format_chat_prompt is prompting.format_chat_prompt, "steering must use the shared formatter"


def test_extraction_store_alignment_namespacing_and_stale_check():
    s = _setup()
    model, tok = s["model"], s["tokenizer"]
    per_layer = ae.extract_registry_activations(model, tok, MODEL_ID, chunk_size=700)
    n = len(dio.load_registry())
    assert n == 6144
    assert sorted(per_layer) == [-3, -2, -1] and all(t.shape == (n, 32) for t in per_layer.values())

    meta = store.check_fresh(MODEL_ID)
    assert meta["n_prompts"] == n and meta["layers"] == [-3, -2, -1] and meta["experiment"] == EXP and meta["language"] == "en"
    assert meta["last_tokens_example"]["neutral"][-1] == "<|end_header_id|>"
    assert store.store_dir(MODEL_ID).parts[-2:] == (SLUG, EXP), "activations must live under <slug>/<experiment>"

    # row i of the store == prompt_id i, regardless of chunking
    ids = [0, 5, 17, 250, 2000, n - 1]
    rows = store.load_prompt_rows(MODEL_ID, ids, layers=[-1])[-1]
    again = ae.extract_registry_activations(model, tok, MODEL_ID, hidden_layers=[-1], chunk_size=64, limit=n)[-1][ids]
    assert torch.allclose(rows.float(), again.float(), atol=1e-3), "store rows are misaligned with prompt ids"

    # dry run (--limit): representative prompts, nothing saved, the store keeps its full contents
    dry = ae.extract_registry_activations(model, tok, MODEL_ID, hidden_layers=[-1], limit=40, chunk_size=64)
    assert dry[-1].shape[0] <= 40 and dry[-1].shape[1] == 32
    assert store.check_fresh(MODEL_ID)["n_prompts"] == n

    # the 200 neutral negatives are shared by every training dataset (extracted once)
    a, la = store.load_train_activations(MODEL_ID, SINGLE_NAME, layers=[-1])
    b, lb = store.load_train_activations(MODEL_ID, subtype_name("hostile sexism"), layers=[-1])
    assert la == lb and torch.equal(a[-1][1::2], b[-1][1::2]) and not torch.equal(a[-1][0::2], b[-1][0::2])

    # stale activations are refused
    meta_path = store.store_dir(MODEL_ID) / "meta.json"
    good = meta_path.read_text()
    bad = json.loads(good)
    bad["registry_sha1"] = "0" * 40
    meta_path.write_text(json.dumps(bad))
    try:
        store.load_train_activations(MODEL_ID, SINGLE_NAME)
        raise AssertionError("stale activations were not detected")
    except RuntimeError as e:
        assert "Stale" in str(e)
    finally:
        meta_path.write_text(good)

    # verify_activations: clean store -> no FAIL; a corrupted layer file is caught
    import verify_activations as va
    levels = va.verify(MODEL_ID)
    assert "FAIL" not in levels and levels.count("OK") >= 7, levels
    lp_path = store.layer_path(MODEL_ID, -1)
    original = torch.load(lp_path, weights_only=True).clone()
    broken = original.clone()
    broken[5, 3] = float("nan")
    broken[7] = 0.0
    torch.save(broken, lp_path)
    try:
        assert "FAIL" in va.verify(MODEL_ID), "NaN / all-zero rows were not detected"
    finally:
        torch.save(original, lp_path)
    assert "FAIL" not in va.verify(MODEL_ID)

    # another experiment tag sees nothing: outputs are namespaced
    experiment.set_experiment(EXP + "_other")
    try:
        store.load_meta(MODEL_ID)
        raise AssertionError("another experiment must not see these activations")
    except FileNotFoundError:
        pass
    finally:
        experiment.set_experiment(EXP)


def test_vectors_holdout_averages_similarity_and_signs():
    cvt.main(model_id=MODEL_ID, method="mean_difference")
    all_sub, train_sub, held_sub = (dio.subtype_vector_names(w) for w in ("all", "train", "heldout"))
    assert len(all_sub) == 20 and len(train_sub) == 16 and len(held_sub) == 4
    for name in [SINGLE_NAME, POOLED_NAME, AVERAGED_NAME, AVERAGED_ALL_NAME, *all_sub]:
        assert cvt.vector_exists(MODEL_ID, name), name
    assert cvt.vector_path(MODEL_ID, SINGLE_NAME).parts[-3:-1] == (SLUG, EXP)

    A, _, method = cvt.load_concept_vectors(MODEL_ID, AVERAGED_NAME)
    A_all, _, _ = cvt.load_concept_vectors(MODEL_ID, AVERAGED_ALL_NAME)
    assert method.startswith("average(16 subtypes") and sorted(A) == [-3, -2, -1]
    comps = pickle.load(open(cvt.vector_path(MODEL_ID, AVERAGED_NAME), "rb"))["components"]
    comps_all = pickle.load(open(cvt.vector_path(MODEL_ID, AVERAGED_ALL_NAME), "rb"))["components"]
    assert comps == train_sub and not set(comps) & set(held_sub), "A must not contain the held-out phrases"
    assert comps_all == all_sub
    for l, v in A.items():
        assert v.shape == (1, 32) and abs(float(v.norm()) - 1.0) < 1e-4
        assert abs(float(A_all[l].norm()) - 1.0) < 1e-4 and not torch.allclose(v, A_all[l]), "A (16) and A_all (20) must differ"

    # A really is the normalised mean of exactly the 16 non-held-out vectors; A_all of all 20
    for names, got in ((train_sub, A), (all_sub, A_all)):
        manual = torch.stack([cvt.load_concept_vectors(MODEL_ID, n)[0][-1][0] for n in names]).mean(0)
        assert torch.allclose(manual / manual.norm(), got[-1][0], atol=1e-5)

    # sign convention: positive projection == misogyny prompt, on the TRAINING rows
    acts, labels = store.load_train_activations(MODEL_ID, POOLED_NAME)
    y = np.array(labels)
    for l, v in A.items():
        p = (acts[l].float() @ v[0]).numpy()
        assert p[y == 1].mean() > p[y == 0].mean(), f"layer {l}: vector points the wrong way"

    rep = pickle.load(open(Path(cvt.CONCEPT_VECTORS_DIR) / SLUG / EXP / "_similarity.pkl", "rb"))
    for l, r in rep.items():
        assert -1.0 <= r["A_member_min_cos"] <= r["A_member_mean_cos"] <= 1.0 + 1e-6
        for k in ("S_held_mean", "P_held_mean", "A_held_mean"):
            assert -1.0 <= r[k] <= 1.0, k

    # resume logic: a partial-layer vector must NOT count as a finished full run
    tmp0 = Path(tempfile.mkdtemp())
    try:
        cvt.train_one_vector(MODEL_ID, SINGLE_NAME, "mean_difference", layers=[-2], save_dir=tmp0)
        assert cvt.vector_covers(MODEL_ID, SINGLE_NAME, [-2], tmp0)
        assert not cvt.vector_covers(MODEL_ID, SINGLE_NAME, [-1, -2, -3], tmp0), "partial vector was treated as complete"
        assert cvt.vector_covers(MODEL_ID, SINGLE_NAME, [-2], tmp0, method="mean_difference")
        assert not cvt.vector_covers(MODEL_ID, SINGLE_NAME, [-2], tmp0, method="rfm"), "a mean-difference vector was treated as RFM"
    finally:
        shutil.rmtree(tmp0, ignore_errors=True)

    # RFM path (CPU, tiny d) on one dataset / one layer, saved to a temp dir
    tmp = Path(tempfile.mkdtemp())
    try:
        d, sg = cvt.train_one_vector(MODEL_ID, SINGLE_NAME, "rfm", layers=[-2], save_dir=tmp)
        assert list(d) == [-2] and d[-2].shape == (1, 32) and sg[-2][0] in (1.0, -1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_training_one_vector_does_not_relabel_or_rebuild_the_averaged_vectors():
    sim = cvt.exp_dir(cvt.CONCEPT_VECTORS_DIR, MODEL_ID) / "_similarity.pkl"
    avg, avg_all = cvt.vector_path(MODEL_ID, AVERAGED_NAME), cvt.vector_path(MODEL_ID, AVERAGED_ALL_NAME)
    before = (avg.read_bytes(), avg_all.read_bytes(), sim.read_bytes())
    assert cvt.load_concept_vectors(MODEL_ID, AVERAGED_NAME)[2] == "average(16 subtypes, mean_difference)"

    # RFM for the single-phrase vector on ONE layer: the averaged vectors and the similarity report stay byte-identical
    cvt.main(model_id=MODEL_ID, vectors=[SINGLE_NAME], method="rfm", layers=[-2])
    assert cvt.load_concept_vectors(MODEL_ID, SINGLE_NAME)[2] == "rfm"
    assert (avg.read_bytes(), avg_all.read_bytes(), sim.read_bytes()) == before, "training S must not touch the derived files"
    assert cvt.load_concept_vectors(MODEL_ID, AVERAGED_NAME)[2] == "average(16 subtypes, mean_difference)"

    # one SUBTYPE vector retrained with RFM -> the averaged vectors are rebuilt and the label records both methods
    one = dio.subtype_vector_names("train")[0]
    cvt.main(model_id=MODEL_ID, vectors=[one], method="rfm")
    assert cvt.load_concept_vectors(MODEL_ID, AVERAGED_NAME)[2] == "average(16 subtypes, mixed: mean_difference+rfm)"
    assert cvt.load_concept_vectors(MODEL_ID, AVERAGED_ALL_NAME)[2] == "average(20 subtypes, mixed: mean_difference+rfm)"

    # back to the original state for the tests that follow
    cvt.main(model_id=MODEL_ID, vectors=[SINGLE_NAME, one], method="mean_difference")
    assert cvt.load_concept_vectors(MODEL_ID, AVERAGED_NAME)[2] == "average(16 subtypes, mean_difference)"
    assert cvt.load_concept_vectors(MODEL_ID, SINGLE_NAME)[2] == "mean_difference"


def test_probes_monitor_and_controls_run_on_seen_and_heldout_sets():
    for eval_set in ("main", "heldout"):
        r = lp.main(MODEL_ID, eval_set=eval_set)
        assert sorted(r) == [-3, -2, -1] and all(0.0 <= v["auroc"] <= 1.0 for v in r.values())
        r = dpp.main(MODEL_ID, AVERAGED_NAME, eval_set=eval_set)
        assert all(0.0 <= v["auroc"] <= 1.0 for v in r.values())
        m = pm.main(MODEL_ID, AVERAGED_NAME, eval_set=eval_set)
        assert 0.0 <= m["auroc"] <= 1.0
        c = control_eval.main(MODEL_ID, AVERAGED_NAME, eval_set=eval_set)
        for k in ["pair_acc_subtype", "pair_acc_benign", "pair_acc_hateful", "auroc_subtype_vs_benign",
                  "auroc_subtype_vs_hateful", "specificity"]:
            assert 0.0 <= c[-1][k] <= 1.0, k
    best = max(v["auroc"] for v in dpp.main(MODEL_ID, AVERAGED_NAME, eval_set="main").values())
    assert best > 0.55, f"direction probe is at chance on the paired test (best AUROC {best})"

    # "own phrase": a training subtype on the seen test, a held-out subtype on the held-out test
    assert sorted(dpp.main(MODEL_ID, subtype_name("hostile sexism"), eval_scope="own", eval_set="main")) == [-3, -2, -1]
    assert sorted(dpp.main(MODEL_ID, subtype_name("incel ideology"), eval_scope="own", eval_set="heldout")) == [-3, -2, -1]
    assert sorted(control_eval.main(MODEL_ID, subtype_name("incel ideology"), eval_scope="own", eval_set="heldout")) == [-3, -2, -1]
    try:   # a training subtype has no pairs in the held-out set
        dpp.main(MODEL_ID, subtype_name("hostile sexism"), eval_scope="own", eval_set="heldout")
        raise AssertionError("should have raised")
    except KeyError:
        pass

    for sub in ["probe_results", "direction_probe_results", "monitor_results", "control_results"]:
        files = list((ROOT / sub / SLUG / EXP).glob("*.pkl"))
        assert any("heldout" in f.name for f in files) and any("main" in f.name for f in files), sub


def test_compare_vectors_helpers_are_correct_on_synthetic_data():
    W = {"always": np.ones((50, 4), bool), "never": np.zeros((50, 4), bool)}
    out = cmp.cluster_bootstrap(W, n_boot=200, seed=0)
    assert out["acc"]["always"] == (1.0, 1.0, 1.0) and out["acc"]["never"] == (0.0, 0.0, 0.0)
    assert out["diff"][("always", "never")] == (1.0, 1.0, 1.0)
    rng = np.random.default_rng(0)
    half = rng.random((218, 4)) < 0.5
    pt, lo, hi = cmp.cluster_bootstrap({"half": half}, n_boot=500, seed=1)["acc"]["half"]
    assert lo <= pt <= hi and 0.01 < hi - lo < 0.12
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.3])
    assert cmp.pair_wins(p).tolist() == [True, False, False]
    wins = np.array([True, False, True, True])
    Wm = cmp.win_matrix(wins, np.array([7, 7, 9, 9]), np.array(["a", "b", "a", "b"]), ["a", "b"])
    assert Wm.tolist() == [[True, False], [True, True]]


def test_choose_layer_min_over_arms_picks_the_best_worst_case():
    w = lambda p, n=100: np.arange(n) < int(p * n)                       # bool array with win rate p
    arms = {"neutral": {-1: w(1.0), -2: w(0.8), -3: w(0.9)},
            "benign": {-1: w(1.0), -2: w(0.8), -3: w(0.9)},
            "hateful": {-1: w(0.5), -2: w(0.8), -3: w(0.7)}}              # layer -1 is perfect except one arm
    assert cmp.choose_layer_min_over_arms(arms) == -2                    # best worst-case (0.8), not the best average
    full_tie = {"a": {-1: w(0.9), -2: w(0.9)}, "b": {-1: w(0.9), -2: w(0.9)}}
    assert cmp.choose_layer_min_over_arms(full_tie) == -2                # nothing separates them: the first (most negative) layer
    same_worst = {"a": {-1: w(0.9), -2: w(0.9)}, "b": {-1: w(1.0), -2: w(0.9)}}
    assert cmp.choose_layer_min_over_arms(same_worst) == -1              # same worst case (0.9): the higher MEAN wins


def test_layer_profile_helpers_and_run():
    import layer_profile as lp
    assert lp.contiguous_runs([-26, -25, -24, -20]) == [[-26, -25, -24], [-20]]
    assert lp.contiguous_runs([-24]) == [[-24]] and lp.contiguous_runs([]) == []
    row = {"pair_acc_subtype": 0.9, "auroc_subtype_vs_benign": 1.0, "auroc_subtype_vs_hateful": 0.7, "pair_acc_benign": 0.0}
    assert lp.orderings(row) == {"neu": 0.9, "ben": 1.0, "hat": 0.7} and lp.worst(row) == 0.7
    row["auroc_subtype_vs_egalitarian"] = 0.4
    assert lp.worst(row) == 0.4 and "egl" in lp.orderings(row)

    r = lp.run_profile(MODEL_ID, SINGLE_NAME, threshold=0.0)             # threshold 0: every layer passes
    assert sorted(r["seen"]) == sorted(r["unseen"]) == [-3, -2, -1] and r["passing_on_seen"] == [-3, -2, -1]
    assert r["runs"] == [[-3, -2, -1]] and r["best_seen_layer"] in (-1, -2, -3)
    r2 = lp.run_profile(MODEL_ID, SINGLE_NAME, threshold=1.01)           # impossible threshold: nothing passes
    assert r2["passing_on_seen"] == [] and r2["runs"] == []
    out = lp.main(MODEL_ID, SINGLE_NAME, threshold=0.5)
    assert (ROOT / "profile_results" / SLUG / EXP / f"layer_profile__{SINGLE_NAME}.pkl").exists() and out["vector"] == SINGLE_NAME


def test_compare_vectors_on_the_tiny_model():
    r = cmp.compare(MODEL_ID, n_boot=200, seed=0)
    held = HELD_OUT_SUBTYPES
    assert r["heldout"] == held and r["n_statements"] == 218
    for n in (SINGLE_NAME, POOLED_NAME, AVERAGED_NAME, AVERAGED_ALL_NAME, "oracle (own-phrase vectors)"):
        pt, lo, hi = r["heldout_acc"][n]
        assert 0.0 <= lo <= pt <= hi <= 1.0, n
    assert all(l in (-3, -2, -1) for l in r["chosen_layer"].values())
    assert (AVERAGED_NAME, POOLED_NAME) in r["diff"] or (POOLED_NAME, AVERAGED_NAME) in r["diff"]
    assert set(r["per_phrase_acc"][SINGLE_NAME]) == set(held)
    # the layer is chosen on the SEEN test, never on the held-out test
    for n in (SINGLE_NAME, POOLED_NAME, AVERAGED_NAME):
        seen = r["per_layer_seen_acc"][n]
        assert r["chosen_layer"][n] == max(sorted(seen), key=lambda l: seen[l])
    forced = cmp.compare(MODEL_ID, vector_names=[SINGLE_NAME, AVERAGED_NAME], n_boot=50, force_layer=-2)
    assert set(forced["chosen_layer"].values()) == {-2}
    out = cmp.main(MODEL_ID, n_boot=50, against="neutral")
    assert (ROOT / "comparison_results" / SLUG / EXP / "compare_unseen_phrases__neutral.pkl").exists() and out["heldout"] == held

    # --against: the control arms (the neutral comparison saturates on real models, these do not)
    for arm in ("benign", "hateful"):
        ra = cmp.compare(MODEL_ID, n_boot=50, against=arm)
        assert ra["against"] == arm and ra["n_statements"] == 218
        assert 0.0 <= ra["heldout_acc"][SINGLE_NAME][0] <= 1.0
    both = cmp.main(MODEL_ID, n_boot=30, against="all")
    assert set(both) == {"neutral", "benign", "hateful"}
    # --against all uses ONE layer per vector in every table, chosen on the seen test (best worst-case over the arms)
    layers_per_table = [both[a]["chosen_layer"] for a in both]
    assert layers_per_table[0] == layers_per_table[1] == layers_per_table[2]
    assert all(r["layer_rule"] == "min over arms on the seen test" for r in both.values())
    per_arm = cmp.main(MODEL_ID, n_boot=20, against="hateful", select="per-arm")
    assert per_arm["layer_rule"] == "per arm on the seen test"
    assert cmp.main(MODEL_ID, n_boot=20, against="hateful", force_layer=-2)["layer_rule"] == "forced"
    for arm in both:
        assert (ROOT / "comparison_results" / SLUG / EXP / f"compare_unseen_phrases__{arm}.pkl").exists()
    try:
        cmp.score_vectors(MODEL_ID, {SINGLE_NAME: cvt.load_concept_vectors(MODEL_ID, SINGLE_NAME)[0]}, against="nonsense")
        raise AssertionError("an unknown reference arm must be rejected")
    except ValueError:
        pass


def test_pilot_contrasts_helpers_and_run():
    import pilot_contrasts as pc
    # folds: disjoint, together cover every statement exactly once, deterministic per seed
    items = np.arange(218)
    folds = pc.make_folds(items, 5, seed=0)
    flat = np.concatenate(folds)
    assert len(folds) == 5 and sorted(flat.tolist()) == items.tolist() and max(map(len, folds)) - min(map(len, folds)) <= 1
    assert all(np.array_equal(a, b) for a, b in zip(folds, pc.make_folds(items, 5, seed=0)))
    assert not all(np.array_equal(a, b) for a, b in zip(folds, pc.make_folds(items, 5, seed=1)))

    # fit_direction: unit norm, equals the normalised mean difference, averages several reference arms
    g = torch.Generator().manual_seed(0)
    sub, r1, r2 = torch.randn(30, 8, generator=g) + 1.0, torch.randn(30, 8, generator=g), torch.randn(30, 8, generator=g)
    d = pc.fit_direction(sub, [r1])
    assert abs(float(d.norm()) - 1.0) < 1e-5 and torch.allclose(d, (sub.mean(0) - r1.mean(0)) / (sub.mean(0) - r1.mean(0)).norm(), atol=1e-6)
    both = pc.fit_direction(sub, [r1, r2])
    assert torch.allclose(both, (sub.mean(0) - (r1.mean(0) + r2.mean(0)) / 2) / (sub.mean(0) - (r1.mean(0) + r2.mean(0)) / 2).norm(), atol=1e-6)

    # the full pilot on the tiny model
    res = pc.run_pilot(MODEL_ID, layers=[-1, -2, -3], n_folds=5, seed=0)
    assert sorted(res) == [-3, -2, -1]
    for l, per in res.items():
        assert set(k for k in per if not k.startswith("_")) == set(pc.CONTRASTS)
        for c in pc.CONTRASTS:
            for k in ("seen", "unseen"):
                assert set(per[c][k]) == set(pc.REFS) and all(0.0 <= v <= 1.0 for v in per[c][k].values())
            assert 0.0 <= pc.specificity(per[c]) <= 1.0
        assert all(-1.0 <= v <= 1.0 for v in per["_cosines"].values())
    assert res == pc.run_pilot(MODEL_ID, layers=[-1, -2, -3], n_folds=5, seed=0), "the pilot must be deterministic"
    out = pc.main(MODEL_ID, layers=[-2], n_folds=5)
    assert (ROOT / "pilot_results" / SLUG / EXP / "pilot_contrasts.pkl").exists() and sorted(out) == [-2]


def test_steering_hooks_and_eval_with_mock_judge():
    s = _setup()
    model, tok = s["model"], s["tokenizer"]
    directions = ams.load_directions(MODEL_ID, AVERAGED_NAME)

    inputs = tok(prompting.format_chat_prompt(tok, "What do you think?"), return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        base = model(**inputs).logits
        hooks = ams.hook_model(model, directions, list(directions), 5.0)
        steered = model(**inputs).logits
        ams.clear_hooks(hooks)
        after = model(**inputs).logits
    assert not torch.allclose(base, steered, atol=1e-4), "hook had no effect"
    assert torch.allclose(base, after), "hooks were not removed"

    seen_prompts = []

    def judge(prompt):
        seen_prompts.append(prompt)
        return '{"misogynistic": 0, "coherent": 1, "refusal_or_hedge": 0, "explanation": "mock"}'

    prompts = se.load_eval_prompts(n=2)
    result = se.run_steering_eval(model, tok, directions, prompts, [-0.2, 0.0, 0.2], max_new_tokens=6,
                                  judge_fn=judge, rubric="RUBRIC", progress=False)
    assert len(result["rows"]) == 6 and [e["coef"] for e in result["summary"]] == [-0.2, 0.0, 0.2]
    assert all(e["coherent"] == 1.0 and e["misogynistic"] == 0.0 for e in result["summary"])
    assert "begin_of_text" not in [r["response"] for r in result["rows"] if r["coef"] == 0.0][0], "response must not echo the prompt"
    assert not any("language with code" in p for p in seen_prompts), "English runs must not add a language note"

    se.run_steering_eval(model, tok, directions, prompts[:1], [0.0], max_new_tokens=3, judge_fn=judge,
                         rubric="RUBRIC", progress=False, language="hi")
    assert "language with code 'hi'" in seen_prompts[-1], "non-English runs tell the judge the language"

    nj = se.run_steering_eval(model, tok, directions, prompts[:1], [0.0], max_new_tokens=4, judge_fn=None, progress=False)
    assert nj["rows"][0]["misogynistic"] is None and "refusal_heuristic" in nj["rows"][0]

    stem = se.save_eval(result, AVERAGED_NAME, MODEL_ID)
    assert Path(f"{stem}.csv").exists() and Path(f"{stem}.pkl").exists()
    assert Path(stem).parts[-3:-1] == (SLUG, EXP), "results must live under <slug>/<experiment>"

    # the degeneration check is recorded per row and averaged per coefficient
    assert all("degenerate_heuristic" in r for r in result["rows"]) and all("degenerate_heuristic" in e for e in result["summary"])
    import csv as _csv
    assert "degenerate_heuristic" in next(_csv.DictReader(open(f"{stem}.csv", encoding="utf-8")))

    # random control: same layers and shapes, unit norm, deterministic per seed, different per seed, ~orthogonal to the real vector
    rnd = se.random_unit_directions(directions, seed=0)
    assert sorted(rnd) == sorted(directions) and all(rnd[l].shape == directions[l].shape for l in directions)
    assert all(abs(float(rnd[l].norm()) - 1.0) < 1e-5 for l in rnd)
    assert all(torch.equal(rnd[l], se.random_unit_directions(directions, 0)[l]) for l in rnd)
    assert not any(torch.equal(rnd[l], se.random_unit_directions(directions, 1)[l]) for l in rnd)
    assert all(abs(float((rnd[l][0] * directions[l][0]).sum())) < 0.9 for l in rnd)
    rr = se.run_steering_eval(model, tok, rnd, prompts[:1], [0.0, 2.0], max_new_tokens=4, judge_fn=None, progress=False)
    assert len(rr["rows"]) == 2
    stem2 = se.save_eval(rr, AVERAGED_NAME + "__random0__tagged", MODEL_ID)
    assert Path(f"{stem2}.csv").exists() and Path(f"{stem2}.csv") != Path(f"{stem}.csv"), "tagged runs must not overwrite each other"

    # the samples viewer works on the saved file, and recomputes the check for files saved before it existed
    import show_steering_samples as sss
    rows = sss.load_rows(f"{stem}.csv")
    assert len(rows) == 6 and {e["coef"] for e in sss.coef_summary(rows)} == {-0.2, 0.0, 0.2}
    old = Path(f"{stem}.csv").with_name("old_format.csv")
    with open(old, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["prompt", "coef", "response", "refusal_heuristic"])
        w.writeheader()
        w.writerow({"prompt": "q", "coef": 4.0, "response": "graz " * 40, "refusal_heuristic": 0})
        w.writerow({"prompt": "q", "coef": 0.0, "response": "Women are as capable as men in every role, and pay should reflect skills, not gender.", "refusal_heuristic": 0})
    summ = {e["coef"]: e["degenerate"] for e in sss.coef_summary(sss.load_rows(str(old)))}
    assert summ == {0.0: 0.0, 4.0: 1.0}
    old.unlink()


def test_stance_logit_test_statements_statistics_and_run():
    import json
    import subprocess
    import sys
    import tempfile
    import stance_logit_test as slt

    # ---- statements ----
    claims = slt.default_claims()
    cats = [c["category"] for c in claims]
    assert len(claims) == 120 and len({c["text"] for c in claims}) == 120
    assert {k: cats.count(k) for k in set(cats)} == {k: 20 for k in ("sexist", "egalitarian", "disability", "immigrant", "fact_true", "fact_false")}
    assert slt.load_claims(None) == claims
    sexist = [c["text"] for c in claims if c["category"] == "sexist"]
    egal = [c["text"] for c in claims if c["category"] == "egalitarian"]
    assert [t.split(" at ")[1] for t in sexist] == [t.split(" at ")[1] for t in egal], "sexist and egalitarian items must use the same skills"
    with tempfile.TemporaryDirectory() as d:
        good, bad = Path(d) / "good.json", Path(d) / "bad.json"
        good.write_text(json.dumps(claims[:3]), encoding="utf-8")
        bad.write_text(json.dumps([{"text": "no category"}]), encoding="utf-8")
        assert slt.load_claims(str(good)) == claims[:3]
        try:
            slt.load_claims(str(bad))
            raise AssertionError("a claim without a category must be rejected")
        except ValueError:
            pass
        out = Path(d) / "dump.json"
        done = subprocess.run([sys.executable, "stance_logit_test.py", "--write-claims", str(out)], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0 and json.loads(out.read_text(encoding="utf-8")) == claims

    # ---- the optional 'reversed' category and the gender-specificity metric ----
    rclaims = slt.default_claims(include_reversed=True)
    rcats = [c["category"] for c in rclaims]
    assert len(rclaims) == 140 and rcats.count("reversed") == 20 and slt.load_claims(None, include_reversed=True) == rclaims
    reff = np.array([2.0 if c == "sexist" else -2.0 if c == "egalitarian" else 1.5 if c == "reversed" else 0.1 for c in rcats])
    rs = slt.summary_numbers(reff, rcats)
    assert abs(rs["gender_specificity"] - 0.5) < 1e-9 and abs(rs["stance_shift"] - 4.0) < 1e-9
    assert "gender_specificity" not in slt.summary_numbers(np.zeros(120), cats)

    # ---- statistics on synthetic effects: sexist +2, egalitarian -2, the rest +0.1 ----
    eff = np.array([2.0] * 20 + [-2.0] * 20 + [0.1] * 80)
    s = slt.summary_numbers(eff, cats)
    assert abs(s["stance_shift"] - 4.0) < 1e-9 and abs(s["specificity"] - 1.9) < 1e-9
    assert slt.category_means(eff, cats)["egalitarian"] == -2.0
    # empirical p against random directions: with k random values the smallest p is 1/(k+1)
    far = slt.null_summary(10.0, [0.1, -0.1, 0.2, -0.2, 0.0])
    assert abs(far["p"] - 1 / 6) < 1e-12 and far["z"] > 10 and far["n"] == 5
    inside = slt.null_summary(0.05, [0.1, -0.1, 0.2, -0.2, 0.0])
    assert inside["p"] > 0.5
    assert slt.null_summary(5.0, []) is None
    assert abs(slt.null_summary(10.0, [float(x) for x in np.linspace(-0.3, 0.3, 19)])["p"] - 1 / 20) < 1e-12   # 19 randoms can just reach 0.05
    lo, hi = slt.bootstrap_ci(np.array([1.0, 1.0, 1.0, 1.0]))
    assert lo == hi == 1.0
    vals = np.arange(20.0)
    lo, hi = slt.bootstrap_ci(vals)
    assert lo < vals.mean() < hi and slt.bootstrap_ci(vals) == (lo, hi), "deterministic"

    # ---- the real scoring on the tiny model ----
    s0 = _setup()
    model, tok = s0["model"], s0["tokenizer"]
    directions = ams.load_directions(MODEL_ID, AVERAGED_NAME)
    texts = [c["text"] for c in claims[:24]]
    base = slt.score_claims(model, tok, texts)
    assert base.shape == (24,) and np.isfinite(base).all()
    assert np.allclose(base, slt.score_claims(model, tok, texts, batch_size=5), atol=1e-3), "left padding must not change the scores"
    assert np.allclose(base, slt.score_claims(model, tok, texts, directions, list(directions), 0.0)), "coefficient 0 is the baseline"
    steered = slt.score_claims(model, tok, texts, directions, list(directions), 5.0)
    assert not np.allclose(base, steered, atol=1e-4), "the vector must change the scores"
    assert np.allclose(base, slt.score_claims(model, tok, texts), atol=1e-6), "hooks must be removed afterwards"

    res = slt.run_stance_test(model, tok, directions, claims, list(directions), [-1.0, 2.0], seeds=[0, 1], batch_size=16, progress=False)
    assert sorted(res["effect"]) == [-1.0, 2.0] and sorted(res["random"]) == [0, 1] and len(res["baseline"]) == 120
    assert all(len(v) == 120 for v in res["effect"].values())
    assert not np.allclose(res["random"][0][2.0], res["random"][1][2.0]), "different seeds give different random controls"
    assert np.allclose(res["effect"][2.0], slt.score_claims(model, tok, [c["text"] for c in claims], directions, list(directions), 2.0, 16) - res["baseline"], atol=1e-4)
    rows = slt.tabulate(res)
    assert [r["coef"] for r in rows] == [-1.0, 2.0] and all(r["null_specificity"]["n"] == 2 for r in rows)
    assert all(r["gender_specificity"] is None for r in rows), "no reversed statements, no gender metric"
    resr = slt.run_stance_test(model, tok, directions, rclaims, list(directions), [2.0], seeds=[0, 1], batch_size=16, progress=False)
    rowr = slt.tabulate(resr)[0]
    assert rowr["gender_specificity"] is not None and rowr["null_gender_specificity"]["n"] == 2 and "reversed" in rowr["means"]
    slt.print_table(slt.tabulate(resr))
    assert all(r["null_specificity"]["p"] >= 1 / 3 for r in rows), "2 random directions can never show significance"
    slt.print_table(rows)
    stem = slt.save_result(res, rows, AVERAGED_NAME, MODEL_ID, tag="t")
    assert Path(f"{stem}.pkl").exists() and Path(stem).parts[-3:-1] == (SLUG, EXP)


def test_relative_coefficients_and_the_gpu_batch_scripts():
    import os
    import shutil
    import subprocess
    import tempfile
    import stance_logit_test as slt

    # ---- relative coefficients: coef * layer norm * unit vector, for the learned vector AND the random control ----
    s0 = _setup()
    model, tok = s0["model"], s0["tokenizer"]
    layers = [-1, -2, -3]
    norms = slt.layer_norms(MODEL_ID, layers)
    manual = {l: float(store._load_tensor(store.layer_path(MODEL_ID, l)).float().norm(dim=1).median()) for l in layers}
    assert norms == manual and all(v > 0 for v in norms.values())
    directions = ams.load_directions(MODEL_ID, AVERAGED_NAME)
    scaled = slt.scale_directions(directions, {-1: 2.0, -2: 3.0})
    assert torch.allclose(scaled[-1], directions[-1] * 2) and torch.allclose(scaled[-2], directions[-2] * 3) and torch.equal(scaled[-3], directions[-3])
    claims = slt.default_claims()[:30]
    uniform = {l: 2.0 for l in directions}
    a = slt.run_stance_test(model, tok, directions, claims, list(directions), [1.0], seeds=[0], batch_size=10, progress=False, scales=uniform)
    b = slt.run_stance_test(model, tok, directions, claims, list(directions), [2.0], seeds=[0], batch_size=10, progress=False)
    assert np.allclose(a["effect"][1.0], b["effect"][2.0], atol=1e-4), "relative coefficient 1 at scale 2 must equal absolute coefficient 2"
    assert np.allclose(a["random"][0][1.0], b["random"][0][2.0], atol=1e-4), "the random control must get the same scales"
    assert a["scales"] == uniform and b["scales"] is None

    # ---- GPU selector and batch script, with a fake nvidia-smi and a fake python ----
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake = tmp / "bin"
        fake.mkdir()
        smi = fake / "nvidia-smi"

        def set_free(*vals):
            smi.write_text("#!/usr/bin/env bash\nprintf '%s\\n' " + " ".join(str(v) for v in vals) + "\n")
            smi.chmod(0o755)

        env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}

        def pick():
            r = subprocess.run(["bash", str(ROOT / "scripts" / "pick_gpus.sh")], env=env, capture_output=True, text=True)
            return r.returncode, r.stdout.strip()

        set_free(20000, 11000); assert pick() == (0, "CUDA_VISIBLE_DEVICES=0")
        set_free(11000, 22000); assert pick() == (0, "CUDA_VISIBLE_DEVICES=1")
        set_free(18000, 19000); assert pick() == (0, "CUDA_VISIBLE_DEVICES=1"), "the GPU with the most room"
        set_free(17000, 3000); assert pick() == (0, "CUDA_VISIBLE_DEVICES=0"), "the limit is inclusive"
        set_free(12000, 11000); assert pick() == (0, "MAX_GPU_GIB=9 CUDA_VISIBLE_DEVICES=0,1")
        set_free(16999, 10000); assert pick() == (0, "MAX_GPU_GIB=9 CUDA_VISIBLE_DEVICES=0,1")
        set_free(5000, 11000); assert pick() == (1, "")
        set_free(9999, 9999); assert pick() == (1, "")

        proj = tmp / "proj"
        (proj / "scripts").mkdir(parents=True)
        for f in ("pick_gpus.sh", "stance_batch.sh"):
            shutil.copy(ROOT / "scripts" / f, proj / "scripts" / f)
        py = fake / "python"
        py.write_text('#!/usr/bin/env bash\necho "CVD=$CUDA_VISIBLE_DEVICES MAXG=${MAX_GPU_GIB:-} ARGS=$*" >> calls.txt\n')
        py.chmod(0o755)
        benv = {**env, "BATCH_WAIT_SECONDS": "0", "BATCH_MAX_ATTEMPTS": "2"}

        set_free(12000, 11000)
        subprocess.run(["bash", "scripts/stance_batch.sh"], cwd=proj, env=benv, capture_output=True, text=True, timeout=60)
        calls = (proj / "calls.txt").read_text().splitlines()
        assert len(calls) == 2
        assert "CVD=0,1" in calls[0] and "MAXG=9" in calls[0] and "--relative" in calls[0] and "--tag allrel" in calls[0] and "stance_logit_test.py" in calls[0]
        assert "--layers -7 -8 -9 -10" in calls[1] and "--tag content" in calls[1] and "--relative" not in calls[1]
        log = (proj / "logs" / "stance_batch.log").read_text()
        assert "starting logs/stance_allayers.log" in log and "finished logs/stance_content.log (exit 0)" in log and "batch done" in log

        (proj / "calls.txt").unlink()
        set_free(3000, 3000)                                               # no room anywhere: it must wait, then give up, and run nothing
        subprocess.run(["bash", "scripts/stance_batch.sh"], cwd=proj, env=benv, capture_output=True, text=True, timeout=60)
        assert not (proj / "calls.txt").exists()
        assert "gave up waiting for GPUs for logs/stance_allayers.log" in (proj / "logs" / "stance_batch.log").read_text()


def test_frame_generalization_prompts_statistics_and_run():
    import frame_generalization_test as fg
    import layer_profile as lp
    from raw_data import all_concept_ids, load_language

    pack = load_language("en", all_concept_ids())
    # ---- the new frames really are new ----
    fg.check_frames_are_new(fg.NEW_FRAMES, pack["frames"])
    for bad in (pack["frames"][0], "Take on the role of a person who loves {concept}."):       # an old frame, and one that starts like an old frame
        try:
            fg.check_frames_are_new([bad], pack["frames"])
            raise AssertionError(f"{bad!r} must be rejected")
        except ValueError:
            pass
    assert len(fg.NEW_FRAMES) == 8 and all("{concept}" in f for f in fg.NEW_FRAMES)

    # ---- balanced assignment and prompt building ----
    b = fg.balanced(40, 8, seed=1)
    assert sorted(np.bincount(b)) == [5] * 8 and np.array_equal(b, fg.balanced(40, 8, seed=1)) and not np.array_equal(b, fg.balanced(40, 8, seed=2))
    assert max(np.bincount(fg.balanced(41, 8, seed=1))) - min(np.bincount(fg.balanced(41, 8, seed=1))) <= 1
    statements = [f"Statement number {i}." for i in range(40)]
    concepts = {"subtype_heldout": ["slut-shaming", "incel ideology"], "benign": ["minimalism", "stoicism", "environmentalism"]}
    prompts = fg.build_prompts(statements, pack, concepts, seed=0)
    assert set(prompts) == {"neutral", "subtype_heldout", "benign"} and all(len(v) == 40 for v in prompts.values())
    for j, s in enumerate(statements):                                   # every arm wraps the SAME statement, in the same position
        assert all(prompts[a][j].endswith(f"Statement: {s}") for a in prompts)
    assert all("Take on" not in p and "Act as" not in p for a in ("subtype_heldout", "benign") for p in prompts[a])
    prefixes = [f.split("{concept}")[0] for f in fg.NEW_FRAMES]
    assert all(any(p.startswith(px) for px in prefixes) and "  What are your thoughts" in p for a in ("subtype_heldout", "benign") for p in prompts[a])
    assert all(not any(px in p for px in prefixes) for p in prompts["neutral"]), "the neutral arm has no instruction"
    assert all(any(c in p for c in concepts["subtype_heldout"]) for p in prompts["subtype_heldout"])
    assert sorted(np.bincount([concepts["benign"].index(next(c for c in concepts["benign"] if c in p)) for p in prompts["benign"]])) == [13, 13, 14]
    assert prompts == fg.build_prompts(statements, pack, concepts, seed=0) and prompts != fg.build_prompts(statements, pack, concepts, seed=1)

    # ---- the concept sets of this experiment (the tiny experiment has no egalitarian arm) ----
    arms_here = fg.concepts_for_this_experiment()
    assert list(arms_here) == ["subtype_heldout", "subtype_seen", "benign", "hateful"] and len(arms_here["subtype_heldout"]) == 4 and len(arms_here["subtype_seen"]) == 16
    with experiment.using(EXP + "_fm"):
        db.main(negatives="matched")
        assert list(fg.concepts_for_this_experiment())[-1] == "egalitarian"

    # ---- statistics on synthetic activations: a direction that satisfies the four orderings against one that does not ----
    g = torch.Generator().manual_seed(777777)           # not 0: random direction number 0 uses seed 0 and would be the planted vector itself
    d, n = 64, 60
    real = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=1)
    other = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=1)
    base = {a: torch.randn(n, d, generator=g) * 0.5 for a in ("neutral", "subtype_heldout", "subtype_seen", "benign", "hateful", "egalitarian")}
    for a in ("subtype_heldout", "subtype_seen"):
        base[a] = base[a] + 3.0 * real[0]
    hid = {a: {-1: t} for a, t in base.items()}
    good = fg.run_frame_test(hid, {-1: real}, n_random=30)
    assert lp.worst(good["heldout"][-1]) > 0.95 and lp.worst(good["seen"][-1]) > 0.95 and abs(good["null_p"][-1] - 1 / 31) < 1e-12
    bad = fg.run_frame_test(hid, {-1: other}, n_random=30)
    assert lp.worst(bad["heldout"][-1]) < 0.8 and bad["null_p"][-1] > 0.3
    assert good == fg.run_frame_test(hid, {-1: real}, n_random=30), "deterministic"
    fg.print_table(good, reference={-1: 0.99})

    # ---- the real reading on the tiny model ----
    s0 = _setup()
    model, tok = s0["model"], s0["tokenizer"]
    directions = ams.load_directions(MODEL_ID, AVERAGED_NAME)
    reg = dio.load_registry()
    stm = reg[reg["kind"] == "test_neutral"].sort_values("pair_id")["statement"].tolist()[:24]
    pr = fg.build_prompts(stm, pack, arms_here, seed=0)
    hidden = fg.read_arm_hidden(model, tok, pr, list(directions), batch_size=8)
    assert set(hidden) == set(pr) and all(hidden[a][-2].shape == (24, 32) for a in hidden)
    res = fg.run_frame_test(hidden, directions, n_random=4)
    assert sorted(res["heldout"]) == sorted(directions) == sorted(res["null_p"]) and res["n_random"] == 4
    assert all(p >= 1 / 5 for p in res["null_p"].values())
    assert fg.load_reference(MODEL_ID, AVERAGED_NAME) is None
    ref = fg.load_reference(MODEL_ID, SINGLE_NAME)
    assert ref is not None and sorted(ref) == [-3, -2, -1]


def test_gender_swap_prompts_statistics_and_run():
    import gender_swap_test as gs
    from frame_generalization_test import NEW_FRAMES
    from raw_data import all_concept_ids, load_language

    pack = load_language("en", all_concept_ids())
    # ---- the pairs: the misogyny side is a known phrase, the misandry side was never used anywhere ----
    assert len(gs.SWAP_PAIRS) == 6
    known = set(all_concept_ids())
    assert all(a in known for a, _ in gs.SWAP_PAIRS) and not any(b in known for _, b in gs.SWAP_PAIRS)
    assert not any(b in pack["phrases"] for _, b in gs.SWAP_PAIRS)
    extended = gs.pack_with_swaps(pack)
    assert all(b in extended["phrases"] for _, b in gs.SWAP_PAIRS) and "misandry" not in pack["phrases"], "the original pack must not be changed"
    held = set(dio.heldout_concepts())
    assert {"dehumanization of women", "doubting women's competence"} <= held and held >= {a for a, _ in gs.SWAP_PAIRS[:2]}

    # ---- prompts: the two members of a pair differ ONLY in the target gender ----
    statements = [f"Statement number {i}." for i in range(20)]
    pr = gs.build_swap_prompts(statements, pack, seed=0)
    assert len(pr) == 2 * 6 * 2 and all(len(v) == 20 for v in pr.values())
    assert pr == gs.build_swap_prompts(statements, pack, seed=0) and pr != gs.build_swap_prompts(statements, pack, seed=1)
    train_prefixes = [f.split("{concept}")[0] for f in pack["frames"]]
    new_prefixes = [f.split("{concept}")[0] for f in NEW_FRAMES]
    for fs, prefixes in (("training", train_prefixes), ("new", new_prefixes)):
        for i, (mis, man) in enumerate(gs.SWAP_PAIRS):
            for a, b, s in zip(pr[f"{fs}|{i}|mis"], pr[f"{fs}|{i}|man"], statements):
                assert b.replace(man, mis) == a and a != b, "only the target gender may differ"
                assert any(a.startswith(px) for px in prefixes) and a.endswith(f"Statement: {s}")
    import re
    assert all(not re.search(r"\bmen\b|\bmen's\b|misandry", p) for k, v in pr.items() if k.endswith("|mis") for p in v)
    assert all(re.search(r"\bmen\b|\bmen's\b|misandry", p) for k, v in pr.items() if k.endswith("|man") for p in v)

    # ---- statistics on synthetic activations ----
    g = torch.Generator().manual_seed(777777)
    d, n = 64, 30
    real = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=1)
    other = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=1)

    def make(shift):
        h = {}
        for fs in gs.FRAME_SETS:
            for i in range(6):
                man = torch.randn(n, d, generator=g) * 0.5
                h[f"{fs}|{i}|man"] = {-1: man}
                h[f"{fs}|{i}|mis"] = {-1: man + shift * real[0] + 1.0 * torch.randn(n, d, generator=g)}
        return h
    sep = gs.run_swap_test(make(5.0), {-1: real}, n_random=30)
    assert sep["training"][-1]["pooled"] > 0.99 and sep["new"][-1]["pooled"] > 0.99 and len(sep["training"][-1]["pairs"]) == 6
    assert abs(sep["training"][-1]["null"]["p"] - 1 / 31) < 1e-12
    blind = gs.run_swap_test(make(0.0), {-1: real}, n_random=30)
    assert 0.35 < blind["training"][-1]["pooled"] < 0.65 and blind["training"][-1]["null"]["p"] > 0.2, "no difference must look like 0.5"
    flipped = gs.run_swap_test(make(-5.0), {-1: real}, n_random=30)
    assert flipped["training"][-1]["pooled"] < 0.01
    gs.print_table(sep)

    # ---- the real reading on the tiny model ----
    s0 = _setup()
    model, tok = s0["model"], s0["tokenizer"]
    directions = ams.load_directions(MODEL_ID, AVERAGED_NAME)
    reg = dio.load_registry()
    stm = reg[reg["kind"] == "test_neutral"].sort_values("pair_id")["statement"].tolist()[:10]
    hidden = gs.read_arm_hidden(model, tok, gs.build_swap_prompts(stm, pack, seed=0), list(directions), batch_size=10)
    assert len(hidden) == 24 and all(h[-2].shape == (10, 32) for h in hidden.values())
    res = gs.run_swap_test(hidden, directions, n_random=4)
    assert sorted(res) == ["new", "training"] and sorted(res["new"]) == sorted(directions)
    assert all(0.0 <= r["pooled"] <= 1.0 and r["null"]["p"] >= 1 / 5 for fs in res for r in res[fs].values())


def test_misandry_position_prompts_statistics_and_run():
    import misandry_position_test as mp
    import gender_swap_test as gs
    from frame_generalization_test import NEW_FRAMES, concepts_for_this_experiment
    from raw_data import all_concept_ids, load_language

    pack = load_language("en", all_concept_ids())
    concepts = dict(concepts_for_this_experiment())
    with pytest.raises(ValueError, match="egalitarian"):
        mp.build_control_prompts(["x"], pack, {k: v for k, v in concepts.items() if k != "egalitarian"}, seed=0)
    concepts.setdefault("egalitarian", list(raw_data.EGALITARIAN_CONTROL_CONCEPTS))
    statements = [f"Statement number {i}." for i in range(20)]
    pr = mp.build_control_prompts(statements, pack, concepts, seed=0)
    assert sorted(pr) == sorted(["neutral"] + [f"{fs}|{a}" for fs in gs.FRAME_SETS for a in ("benign", "hateful", "egalitarian")])
    assert all(len(v) == 20 for v in pr.values()) and pr == mp.build_control_prompts(statements, pack, concepts, seed=0)
    assert all("Statement:" not in p or p.endswith(f"Statement: {s}") for v in pr.values() for p, s in zip(v, statements))
    train_prefixes = [f.split("{concept}")[0] for f in pack["frames"]]
    new_prefixes = [f.split("{concept}")[0] for f in NEW_FRAMES]
    assert all(any(p.startswith(x) for x in train_prefixes) for a in ("benign", "hateful", "egalitarian") for p in pr[f"training|{a}"])
    assert all(any(p.startswith(x) for x in new_prefixes) for a in ("benign", "hateful", "egalitarian") for p in pr[f"new|{a}"])
    assert not any(p.startswith(x) for p in pr["neutral"] for x in train_prefixes + new_prefixes), "the neutral arm has no instruction"

    # ---- statistics on synthetic activations: misogyny shifted along the vector, misandry shifted only when asked to ----
    g = torch.Generator().manual_seed(4242)
    d, n, P = 64, 30, 6
    real = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=1)

    def make(man_shift, mis_shift=5.0):
        sw, ch = {}, {}
        for fs in gs.FRAME_SETS:
            for i in range(P):
                base = torch.randn(n, d, generator=g)
                sw[f"{fs}|{i}|mis"] = {-1: base + mis_shift * real[0] + torch.randn(n, d, generator=g)}
                sw[f"{fs}|{i}|man"] = {-1: base + man_shift * real[0] + torch.randn(n, d, generator=g)}
            for c in mp.CTRL:
                ch[mp.ctrl_key(fs, c)] = {-1: torch.randn(n, d, generator=g)}
        ch["neutral"] = {-1: torch.randn(n, d, generator=g)}
        return sw, ch
    both = mp.evaluate(*make(5.0), {-1: real}, n_random=30)
    assert both["training"][-1]["mis"]["worst"] > 0.95 and both["training"][-1]["man"]["worst"] > 0.95
    assert abs(both["new"][-1]["p_man"] - 1 / 31) < 1e-12 and len(both["new"][-1]["man"]["pair_worst"]) == P
    only = mp.evaluate(*make(0.0), {-1: real}, n_random=30)
    assert only["training"][-1]["mis"]["worst"] > 0.95 and 0.35 < only["training"][-1]["man"]["worst"] < 0.65
    assert only["training"][-1]["p_man"] > 0.2, "no shift must not look better than random"
    assert mp.beat_rates(torch.ones(2, 4), {c: torch.ones(4) for c in mp.CTRL})["worst"] == 0.5
    mp.print_table(only, 30)

    # ---- the real reading on the tiny model ----
    s0 = _setup()
    model, tok = s0["model"], s0["tokenizer"]
    directions = ams.load_directions(MODEL_ID, AVERAGED_NAME)
    reg = dio.load_registry()
    stm = reg[reg["kind"] == "test_neutral"].sort_values("pair_id")["statement"].tolist()[:10]
    sw = gs.read_arm_hidden(model, tok, gs.build_swap_prompts(stm, pack, seed=0), list(directions), batch_size=10)
    ch = gs.read_arm_hidden(model, tok, mp.build_control_prompts(stm, pack, concepts, seed=0), list(directions), batch_size=10)
    res = mp.evaluate(sw, ch, directions, n_random=4)
    assert sorted(res) == ["new", "training"]
    assert all(0.0 <= r["man"]["worst"] <= 1.0 and r["p_man"] >= 1 / 5 for fs in res for r in res[fs].values())


def test_content_monitor_statistics_and_run():
    import tempfile
    import content_monitor_test as cm
    import stance_logit_test as slt
    claims = slt.default_claims()
    cats = [c["category"] for c in claims]

    # ---- AUROC rows and random directions on synthetic scores ----
    scores = np.array([1.0 if c == "sexist" else 0.0 for c in cats]) + np.random.default_rng(0).normal(0, 0.01, len(cats))
    row = cm.auroc_row(scores, cats)
    assert set(row) == set(cm.COMPARISONS) - {"reversed"} and all(abs(v - 1.0) < 1e-9 for v in row.values()), "no 'reversed' group in the default set"
    rev_claims = slt.default_claims(include_reversed=True)
    assert len(rev_claims) == 140 and "reversed" in cm.present_comparisons([c["category"] for c in rev_claims])
    rev = [c for c in rev_claims if c["category"] == "reversed"]
    sx = [c for c in rev_claims if c["category"] == "sexist"]
    assert [t["text"].split(" at ")[1] for t in rev] == [t["text"].split(" at ")[1] for t in sx]
    assert all(r["text"].startswith("Men are generally less capable than women") for r in rev)

    # the vectorized AUROC equals sklearn's, ties included
    rng = np.random.default_rng(5)
    sc = np.round(rng.normal(size=(40, 6)), 1)                                   # rounded -> many ties
    pos_i, neg_i = np.arange(0, 15), np.arange(15, 40)
    fast = cm.auroc_columns(sc, pos_i, neg_i)
    from sklearn.metrics import roc_auc_score
    slow = [roc_auc_score(np.r_[np.ones(15), np.zeros(25)], sc[np.r_[pos_i, neg_i], j]) for j in range(6)]
    assert np.allclose(fast, slow, atol=1e-12)
    assert all(abs(v - 0.5) < 0.2 for v in cm.auroc_row(np.random.default_rng(1).normal(size=len(cats)), cats).values())
    r = cm.random_directions(32, 7, seed=3)
    assert r.shape == (7, 32) and torch.allclose(r.norm(dim=1), torch.ones(7), atol=1e-5)
    assert torch.equal(r, cm.random_directions(32, 7, seed=3)) and not torch.equal(r, cm.random_directions(32, 7, seed=4))

    # ---- a direction that really separates the sexist statements is far outside the random cloud; an unrelated one is not ----
    g = torch.Generator().manual_seed(0)
    d = 64
    real = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=1)
    other = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=1)
    H = torch.randn(len(cats), d, generator=g) * 0.5
    H[torch.tensor([c == "sexist" for c in cats])] += 3.0 * real[0]
    res = cm.run_content_test({-1: H}, {-1: real}, cats, n_random=50)
    assert all(res[-1]["auroc"][k] > 0.99 for k in res[-1]["auroc"]) and "reversed" not in res[-1]["auroc"]
    assert abs(res[-1]["null"]["all_others"]["p"] - 1 / 51) < 1e-12 and res[-1]["null"]["all_others"]["n"] == 50
    assert cm.significant_layers(res, "all_others", p=0.05) == [-1] and cm.significant_layers(res, "reversed") == []
    res2 = cm.run_content_test({-1: H}, {-1: other}, cats, n_random=50)
    assert res2[-1]["null"]["all_others"]["p"] > 0.1, "an unrelated direction must not look significant"
    assert res == cm.run_content_test({-1: H}, {-1: real}, cats, n_random=50), "deterministic"
    cm.print_table(res)

    # ---- the 'reversed' group, saving and reloading the readings ----
    rcats = [c["category"] for c in rev_claims]
    Hr = torch.randn(len(rcats), d, generator=g) * 0.5
    Hr[torch.tensor([c == "sexist" for c in rcats])] += 3.0 * real[0]
    rres = cm.run_content_test({-1: Hr}, {-1: real}, rcats, n_random=50)
    assert rres[-1]["auroc"]["reversed"] > 0.99 and cm.significant_layers(rres, "reversed") == [-1]
    with tempfile.TemporaryDirectory() as tdir:
        path = Path(tdir) / "r.hidden.pt"
        cm.save_hidden(path, {-1: Hr}, rev_claims)
        h2, c2 = cm.load_hidden(str(path))
    assert c2 == rcats and torch.equal(h2[-1], Hr)
    assert cm.run_content_test(h2, {-1: real}, c2, n_random=50) == rres, "recomputing from the saved readings gives the same statistics"

    # ---- the real reading on the tiny model ----
    s0 = _setup()
    model, tok = s0["model"], s0["tokenizer"]
    directions = ams.load_directions(MODEL_ID, AVERAGED_NAME)
    hidden = cm.read_hidden_states(model, tok, claims, list(directions), batch_size=16)
    assert sorted(hidden) == sorted(directions) and all(h.shape == (120, 32) for h in hidden.values())
    again = cm.read_hidden_states(model, tok, claims[:5], list(directions), batch_size=2)
    assert all(torch.allclose(hidden[l][:5], again[l], atol=1e-3) for l in hidden), "batching must not change the readings"
    out = cm.run_content_test(hidden, directions, cats, n_random=20)
    assert sorted(out) == sorted(directions) and all(0.0 <= v["auroc"]["all_others"] <= 1.0 for v in out.values())
    assert all(v["null"]["facts"]["p"] >= 1 / 21 for v in out.values())


def test_extend_from_reuses_rows_extracts_only_new_and_enables_the_egalitarian_arm():
    import copy
    import control_eval as ce
    import pilot_contrasts as pc
    import verify_activations as va
    s = _setup()
    model, tok = s["model"], s["tokenizer"]
    M = EXP + "_m"

    # ---- refusals first: same experiment, non-prefix registry, precision mismatch ----
    try:
        ae.extend_registry_activations(model, tok, MODEL_ID, EXP)
        raise AssertionError("extending an experiment from itself must be refused")
    except ValueError as e:
        assert "DIFFERENT" in str(e)
    with experiment.using(EXP + "_x"):
        db.main(seed=999, negatives="matched")                       # other seed: its registry does NOT start with EXP's
        try:
            ae.extend_registry_activations(model, tok, MODEL_ID, EXP)
            raise AssertionError("a registry that does not extend the source must be refused")
        except ValueError as e:
            assert "does not start with" in str(e)

    with experiment.using(M):
        db.main(negatives="matched")
        m16 = copy.deepcopy(model).half()
        try:
            ae.extend_registry_activations(m16, tok, MODEL_ID, EXP)
            raise AssertionError("a precision mismatch must be refused before extracting")
        except ValueError as e:
            assert "float16" in str(e) and "float32" in str(e)

        # ---- the real extension ----
        combined = ae.extend_registry_activations(model, tok, MODEL_ID, EXP, chunk_size=128)
        meta = store.check_fresh(MODEL_ID)
        assert meta["n_prompts"] == 6562 and meta["extended_from"] == EXP and meta["n_reused"] == 6144 and meta["n_extracted"] == 418
        assert all(t.shape == (6562, 32) for t in combined.values())

        # reused rows are exactly the source rows; the new rows equal a fresh extraction of those prompts
        with experiment.using(EXP):
            src = {l: store._load_tensor(store.layer_path(MODEL_ID, l)) for l in (-1, -2, -3)}
        new_prompts = dio.load_registry()["user_prompt"].iloc[6144:].tolist()
        fresh = ae.extract_hidden_states([prompting.format_chat_prompt(tok, p) for p in new_prompts[:40]], model, tok, [-1, -2, -3], batch_size=4)
        for l in (-1, -2, -3):
            stored = store._load_tensor(store.layer_path(MODEL_ID, l))
            assert torch.equal(stored[:6144], src[l]), f"layer {l}: reused rows differ from the source"
            assert torch.allclose(stored[6144:6184].float(), fresh[l].float(), atol=1e-3), f"layer {l}: new rows are wrong"

        # ---- every later step works on the extended store ----
        levels = va.verify(MODEL_ID)
        assert "FAIL" not in levels and any(True for _ in levels), levels
        a, y = store.load_train_activations(MODEL_ID, SINGLE_NAME, layers=[-2])
        assert a[-2].shape == (400, 32) and y[:2] == [1.0, 0.0]
        acts, table = store.load_arm_activations(MODEL_ID, "heldout", layers=[-2])
        assert set(acts) == {"neutral", "subtype", "benign", "hateful", "egalitarian"} and len(table) == 872

        cvt.main(model_id=MODEL_ID, method="mean_difference")
        res = ce.main(MODEL_ID, AVERAGED_NAME, eval_set="main")
        for k in ("pair_acc_egalitarian", "auroc_subtype_vs_egalitarian", "specificity_all"):
            assert 0.0 <= res[-2][k] <= 1.0, k
        both = cmp.main(MODEL_ID, n_boot=30, against="all")
        assert set(both) == {"neutral", "benign", "hateful", "egalitarian"}
        assert cmp.compare(MODEL_ID, n_boot=30, against="egalitarian")["against"] == "egalitarian"
        # vectors trained in the OTHER experiment, scored on this experiment's data (same prompts, plus the egalitarian arm)
        res_x = ce.main(MODEL_ID, SINGLE_NAME, eval_set="heldout", vector_experiment=EXP)
        assert "auroc_subtype_vs_egalitarian" in res_x[-2] and 0.0 <= res_x[-2]["specificity_all"] <= 1.0
        rx = cmp.compare(MODEL_ID, n_boot=30, against="egalitarian", vector_experiment=EXP)
        assert rx["vector_experiment"] == EXP and rx["against"] == "egalitarian"
        both_x = cmp.main(MODEL_ID, n_boot=20, against="hateful", vector_experiment=EXP)
        assert both_x["vector_experiment"] == EXP
        assert (ROOT / "comparison_results" / SLUG / M / f"compare_unseen_phrases__hateful__vec-{EXP}.pkl").exists()
        assert (ROOT / "control_results" / SLUG / M / f"control_eval__{SINGLE_NAME}__heldout__all__vec-{EXP}.pkl").exists()
        try:
            ce.main(MODEL_ID, SINGLE_NAME, vector_experiment="does_not_exist")
            raise AssertionError("a missing vector experiment must be refused")
        except FileNotFoundError:
            pass

        pil = pc.run_pilot(MODEL_ID, layers=[-2], n_folds=5)
        assert {"vs_egalitarian", "vs_all_controls", "vs_mixed_all", "vs_mixed"} <= set(pil[-2]) and "egalitarian" in pil[-2]["vs_neutral"]["unseen"]
        assert 0.0 <= pc.specificity(pil[-2]["vs_neutral"]) <= 1.0

        # a corrupted reused row is caught by verify_activations
        path = store.layer_path(MODEL_ID, -2)
        original = torch.load(path, weights_only=True).clone()
        broken = original.clone()
        broken[10, 0] += 1.0
        torch.save(broken, path)
        try:
            assert "FAIL" in va.verify(MODEL_ID), "a changed reused row was not detected"
        finally:
            torch.save(original, path)

    # the default experiment's compare keeps refusing an arm it does not have
    try:
        cmp.compare(MODEL_ID, n_boot=30, against="egalitarian")
        raise AssertionError("the egalitarian arm must not exist in the default experiment")
    except ValueError as e:
        assert "no 'egalitarian' arm" in str(e)


def test_gpu_memory_cap_builds_and_reaches_the_loader():
    from unittest import mock
    import model_utils as mu
    import os
    old = os.environ.pop("MAX_GPU_GIB", None)
    try:
        assert mu.build_max_memory("", n_gpus=2) is None and mu.build_max_memory(None, n_gpus=2) is None   # unset: no cap
        os.environ["MAX_GPU_GIB"] = "7"
        assert mu.build_max_memory(None, n_gpus=2) == {0: "7GiB", 1: "7GiB"}                            # read from the environment
    finally:
        os.environ.pop("MAX_GPU_GIB", None)
        if old is not None:
            os.environ["MAX_GPU_GIB"] = old
    assert mu.build_max_memory("9", n_gpus=2) == {0: "9GiB", 1: "9GiB"}
    assert mu.build_max_memory("9.5", n_gpus=1) == {0: "9.5GiB"}
    assert mu.build_max_memory("9", n_gpus=0) is None                      # no GPU visible: nothing to cap
    for bad in ("abc", "0", "-3"):
        try:
            mu.build_max_memory(bad, n_gpus=2)
            raise AssertionError(f"{bad!r} must be rejected")
        except ValueError:
            pass

    s = _setup()
    model, tok = s["model"], s["tokenizer"]
    for cap, expect in (({0: "9GiB", 1: "9GiB"}, True), (None, False)):
        captured = {}
        def fake(*a, **kw):
            captured.update(kw)
            return model
        with mock.patch.object(mu, "AutoConfig") as cfg, mock.patch.object(mu, "AutoTokenizer") as tk, \
                mock.patch.object(mu, "AutoModelForCausalLM") as lm, mock.patch.object(mu, "build_max_memory", return_value=cap):
            cfg.from_pretrained.return_value = mock.Mock(architectures=["LlamaForCausalLM"])
            tk.from_pretrained.return_value = tok
            lm.from_pretrained.side_effect = fake
            mu.load_causal_lm("anything")
        assert captured["device_map"] == "auto"
        assert ("max_memory" in captured) == expect and (not expect or captured["max_memory"] == cap)


def test_zz_cleanup():
    _cleanup()


if __name__ == "__main__":
    try:
        order = ["test_chat_template_position_and_single_bos", "test_extraction_store_alignment_namespacing_and_stale_check",
                 "test_vectors_holdout_averages_similarity_and_signs",
                 "test_training_one_vector_does_not_relabel_or_rebuild_the_averaged_vectors", "test_probes_monitor_and_controls_run_on_seen_and_heldout_sets",
                 "test_compare_vectors_helpers_are_correct_on_synthetic_data",
                 "test_choose_layer_min_over_arms_picks_the_best_worst_case",
                 "test_layer_profile_helpers_and_run",
                 "test_gpu_memory_cap_builds_and_reaches_the_loader",
                 "test_stance_logit_test_statements_statistics_and_run",
                 "test_content_monitor_statistics_and_run",
                 "test_gender_swap_prompts_statistics_and_run",
                 "test_misandry_position_prompts_statistics_and_run",
                 "test_frame_generalization_prompts_statistics_and_run",
                 "test_relative_coefficients_and_the_gpu_batch_scripts", "test_compare_vectors_on_the_tiny_model",
                 "test_pilot_contrasts_helpers_and_run",
                 "test_steering_hooks_and_eval_with_mock_judge",
                 "test_extend_from_reuses_rows_extracts_only_new_and_enables_the_egalitarian_arm"]
        for name in order:
            globals()[name]()
            print("PASS", name)
        print(f"\n{len(order)} pipeline tests passed")
    finally:
        _cleanup()
