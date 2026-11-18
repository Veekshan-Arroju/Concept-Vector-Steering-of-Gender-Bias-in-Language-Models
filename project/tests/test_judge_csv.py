"""judge_csv.py with a fake judge (no API, pandas only). Run: python -m pytest tests/test_judge_csv.py -q"""
import _path  # noqa: F401
import json

import pandas as pd
import pytest

import judge_csv as jc


def fake_judge(prompt: str) -> str:
    # judge_one sends the whole rubric prompt; decide from the response part only
    resp = prompt.split("Response:", 1)[1]
    mis = "less effective" in resp
    coh = "loop loop loop" not in resp
    return json.dumps({"misogynistic": mis, "coherent": coh, "refusal_or_hedge": "can't" in resp, "explanation": "fake"})


def frame():
    rows = [("learned", 0.2, "Q1", "women are equal"), ("learned", 0.7, "Q1", "women are less effective"), ("learned", 0.8, "Q2", "loop loop loop less effective"),
            ("learned", -0.7, "Q1", "I can't help"), ("random0", 0.7, "Q1", "fine answer"), ("learned", 0.9, "Q3", "women are less effective too")]
    return pd.DataFrame(rows, columns=["direction", "coef", "prompt", "response"])


def test_bands():
    assert [jc.band_of(c) for c in (0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.2, -0.7)] == ["0.1-0.3", "0.1-0.3", "0.4-0.6", "0.4-0.6", "0.7-0.8", "0.7-0.8", "0.9+", "0.9+", "0.7-0.8"]


def test_judge_frame_caches_and_resumes(tmp_path):
    cache = tmp_path / "c.jsonl"
    calls = []
    def counting(p):
        calls.append(1)
        return fake_judge(p)
    df = frame()
    a = jc.judge_frame(df, counting, "RUBRIC", cache)
    assert len(calls) == 6 and a["misogynistic"].tolist() == [0, 1, 1, 0, 0, 1]
    b = jc.judge_frame(df, counting, "RUBRIC", cache)          # everything from the cache: no new calls
    assert len(calls) == 6 and b["misogynistic"].tolist() == a["misogynistic"].tolist()
    assert len(jc.load_cache(cache)) == 6


def test_failed_judgements_are_not_cached_and_are_counted(tmp_path):
    cache = tmp_path / "c.jsonl"
    boom = lambda p: "not json at all"
    r = jc.judge_frame(frame().iloc[:2], boom, "RUBRIC", cache)
    assert r["misogynistic"].isna().all() and not cache.read_text().strip()        # nothing cached, so it is retried next run
    s = jc.summarize(r, "v")
    assert (s["n"] == 0).all() and s["judge_failed"].sum() == 2


def test_summary_counts_misogynistic_and_coherent_separately(tmp_path):
    j = jc.judge_frame(frame(), fake_judge, "RUBRIC", tmp_path / "c.jsonl")
    s = jc.summarize(j, "vec").set_index(["kind", "sign", "band"])
    assert s.loc[("learned", "+", "0.7-0.8"), ["n", "misog", "misog_coh", "coherent"]].tolist() == [2, 2, 1, 1]   # one answer misogynistic but collapsed
    assert s.loc[("learned", "+", "0.1-0.3"), "misog"] == 0
    assert s.loc[("learned", "-", "0.7-0.8"), "refusal_or_hedge"] == 1
    assert s.loc[("random", "+", "0.7-0.8"), "misog"] == 0                     # random0 is reported as "random"
    assert s.loc[("learned", "+", "0.9+"), "misog_coh"] == 1
    assert "misog&coh" in jc.format_summary(jc.summarize(j, "vec"))


def test_a_changed_rubric_or_judge_never_reuses_cached_verdicts(tmp_path):
    cache, calls = tmp_path / "c.jsonl", []
    def counting(p):
        calls.append(1)
        return fake_judge(p)
    jc.judge_frame(frame(), counting, "RUBRIC A", cache, judge_id="openai:m1")
    n = len(calls)
    jc.judge_frame(frame(), counting, "RUBRIC A", cache, judge_id="openai:m1")
    assert len(calls) == n                                              # same rubric + judge: all cached
    jc.judge_frame(frame(), counting, "RUBRIC B", cache, judge_id="openai:m1")
    assert len(calls) == 2 * n                                          # new rubric text: judged again
    jc.judge_frame(frame(), counting, "RUBRIC B", cache, judge_id="openai:m2")
    assert len(calls) == 3 * n                                          # other model: judged again


def test_parallel_workers_give_the_same_result_in_the_same_order(tmp_path):
    df = pd.concat([frame()] * 3, ignore_index=True)
    df["response"] = [f"{r} #{i}" for i, r in enumerate(df["response"])]     # 18 distinct rows
    a = jc.judge_frame(df, fake_judge, "R", tmp_path / "a.jsonl", workers=1)
    b = jc.judge_frame(df, fake_judge, "R", tmp_path / "b.jsonl", workers=4)
    assert a["misogynistic"].tolist() == b["misogynistic"].tolist() and a["response"].tolist() == b["response"].tolist() == df["response"].tolist()
    assert len(jc.load_cache(tmp_path / "b.jsonl")) == 18


def test_duplicate_rows_are_judged_once(tmp_path):
    calls = []
    def counting(p):
        calls.append(1)
        return fake_judge(p)
    df = pd.concat([frame().iloc[:2]] * 3, ignore_index=True)
    r = jc.judge_frame(df, counting, "R", tmp_path / "c.jsonl")
    assert len(calls) == 2 and len(r) == 6


def test_truncation_note_is_a_single_sentence_about_cut_off_answers():
    assert "cut off" in jc.TRUNCATION_NOTE and "incoherent" in jc.TRUNCATION_NOTE and jc.TRUNCATION_NOTE.count("Important") == 1


def test_error_message_parsing():
    first_error = "Error code: 429 - You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/."
    rate = "Rate limit reached for gpt-4o in organization org-x on tokens per min (TPM): Limit 30000. Please try again in 6s."
    assert jc.is_quota_error(first_error) and jc.is_quota_error("You exceeded your current quota, please check your plan and billing details")
    assert not jc.is_quota_error(rate) and not jc.is_quota_error("connection reset")
    assert jc.wait_from_message(rate) == 6.0 and jc.wait_from_message("Please try again in 250ms.") == 0.25 and jc.wait_from_message("no hint") is None
    assert jc.parse_retry_after("7") == 7.0 and jc.parse_retry_after("0.5") == 0.5 and jc.parse_retry_after(None) is None and jc.parse_retry_after("soon") is None


def bad_judge(prompt: str) -> str:
    return "not json" if "BAD" in prompt.split("Response:", 1)[1] else fake_judge(prompt)


def test_stops_after_consecutive_failures_with_the_error_text(tmp_path):
    df = pd.DataFrame({"direction": "learned", "coef": 0.1, "prompt": "Q", "response": [f"BAD {i}" for i in range(10)]})
    for workers in (1, 3):
        cache = tmp_path / f"c{workers}.jsonl"
        with pytest.raises(jc.JudgeAborted, match="in a row failed") as e:
            jc.judge_frame(df, bad_judge, "R", cache, workers=workers, max_failures=3)
        assert "judge failed" in str(e.value) and not cache.read_text().strip()


def test_isolated_failures_do_not_stop_the_run_and_are_not_cached(tmp_path):
    rows = ["ok 1", "BAD 1", "ok 2", "BAD 2", "ok 3", "ok 4"]
    df = pd.DataFrame({"direction": "learned", "coef": 0.1, "prompt": "Q", "response": rows})
    r = jc.judge_frame(df, bad_judge, "R", tmp_path / "c.jsonl", max_failures=2)
    assert r["misogynistic"].isna().tolist() == [False, True, False, True, False, False] and len(jc.load_cache(tmp_path / "c.jsonl")) == 4


OK_BODY = {"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-4o",
           "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]}


def test_robust_openai_judge_waits_for_the_hint_then_succeeds(monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2")
    real = openai.OpenAI
    calls, slept = [], []
    def handler(req):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "Rate limit reached. Please try again in 6s.", "type": "requests"}})
        return httpx.Response(200, json=OK_BODY)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw))
    monkeypatch.setattr(jc.time, "sleep", lambda s: slept.append(s))
    assert jc.make_robust_openai_judge()("hi") == "OK" and slept == [1.0, 1.0] and len(calls) == 3      # Retry-After header wins over the 6 s text


def test_robust_openai_judge_fails_at_once_without_credits_and_gives_up_after_max_waits(monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2")
    real = openai.OpenAI
    slept = []
    monkeypatch.setattr(jc.time, "sleep", lambda s: slept.append(s))
    def quota(req):
        return httpx.Response(429, json={"error": {"message": "You have no credits remaining. Add credits at https://platform.openai.com/settings/organization/billing/.", "type": "insufficient_quota"}})
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(quota)), **kw))
    with pytest.raises(openai.RateLimitError):
        jc.make_robust_openai_judge()("hi")
    assert slept == []                                                                                    # no waiting for a hopeless error
    def limited(req):
        return httpx.Response(429, json={"error": {"message": "Rate limit reached.", "type": "requests"}})
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(limited)), **kw))
    with pytest.raises(openai.RateLimitError):
        jc.make_robust_openai_judge(max_waits=3)("hi")
    assert slept == [2.0, 4.0, 8.0]                                                                       # exponential, capped at max_waits


def test_filter_coefs_keeps_only_the_requested_strengths_of_both_signs():
    df = pd.DataFrame({"direction": "learned", "coef": [0.1, 0.2, 0.5, -0.5, 0.7, -0.9, 0.3000000001], "prompt": "Q", "response": list("abcdefg")})
    assert jc.filter_coefs(df, [0.2, 0.5, 0.9])["response"].tolist() == ["b", "c", "d", "f"]
    assert jc.filter_coefs(df, [0.3])["response"].tolist() == ["g"] and len(jc.filter_coefs(df, None)) == 7 and len(jc.filter_coefs(df, [])) == 7


def test_daily_limit_detection_with_the_users_real_error_text():
    real = ("Error code: 429 - Rate limit reached for gpt-4o in organization org-x on requests per day (RPD): Limit 50, Used 50, Requested 1. "
            "Please try again in 28m48s.")
    assert jc.is_daily_limit_error(real) and not jc.is_daily_limit_error("Rate limit reached ... on requests per min (RPM): Limit 3")
    assert jc.wait_from_message(real) is None          # '28m48s' is not a plain seconds hint, so the Retry-After header / daily check must decide


def test_robust_judge_does_not_sleep_on_a_daily_limit_or_a_long_hint(monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2")
    real = openai.OpenAI
    slept = []
    monkeypatch.setattr(jc.time, "sleep", lambda s: slept.append(s))
    def daily(req):
        return httpx.Response(429, headers={"retry-after": "1728"}, json={"error": {"message": "Rate limit reached on requests per day (RPD): Limit 50, Used 50.", "type": "requests"}})
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(daily)), **kw))
    with pytest.raises(openai.RateLimitError):
        jc.make_robust_openai_judge()("hi")
    def long_hint(req):
        return httpx.Response(429, headers={"retry-after": "600"}, json={"error": {"message": "Rate limit reached.", "type": "requests"}})
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(long_hint)), **kw))
    with pytest.raises(openai.RateLimitError):
        jc.make_robust_openai_judge()("hi")
    assert slept == []                                  # neither case waited (the old behaviour slept 60 s five times per attempt)


def test_preflight_reports_the_error_text_or_none():
    assert jc.preflight(lambda p: "OK") is None
    def boom(p):
        raise RuntimeError("Rate limit reached on requests per day (RPD): Limit 50")
    msg = jc.preflight(boom)
    assert msg.startswith("RuntimeError:") and jc.is_daily_limit_error(msg)


def test_diag_models_probe_reads_limits_and_retries_the_other_token_parameter():
    import diag_models as dm
    class Raw:
        headers = {"x-ratelimit-limit-requests": "200", "x-ratelimit-remaining-requests": "199", "x-ratelimit-limit-tokens": "60000"}
    class Creator:
        def create(self, model, messages, **kw):
            if "max_tokens" in kw:
                raise RuntimeError("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")
            return Raw()
    class Client:
        class chat:
            class completions:
                with_raw_response = Creator()
    r = dm.probe(Client(), "gpt-new")
    assert r["status"] == "ok" and r["req_limit"] == "200" and r["req_left"] == "199"


def test_main_stops_at_once_with_a_clear_message_when_the_daily_limit_is_hit(monkeypatch, tmp_path, capsys):
    def exhausted(prompt):
        raise RuntimeError("Error code: 429 - Rate limit reached for gpt-4o on requests per day (RPD): Limit 50, Used 50, Requested 1.")
    monkeypatch.setattr(jc, "make_robust_openai_judge", lambda *a, **k: exhausted)
    csv = tmp_path / "x.csv"
    frame().to_csv(csv, index=False)
    monkeypatch.setattr("sys.argv", ["judge_csv.py", "--judge", "openai", "--out", str(tmp_path / "out"), str(csv)])
    with pytest.raises(SystemExit) as e:
        jc.main()
    assert "DAILY REQUEST LIMIT" in str(e.value) and "requests per day" in str(e.value)
    assert not (tmp_path / "out" / "judge_cache.jsonl").exists() or not (tmp_path / "out" / "judge_cache.jsonl").read_text().strip()


def test_local_judge_prefill_cut_and_parse():
    import local_judge as lj
    import judge as judge_mod
    assert lj.cut_after_first_json('{"a": 1} trailing words {"b": 2}') == '{"a": 1}' and lj.cut_after_first_json("no json") == "no json"
    reply = lj.assemble_reply('0, "coherent": 1, "refusal_or_hedge": 0, "explanation": "Respectful answer."}\nI hope this helps! {"x": 1}')
    verdict = judge_mod.parse_judgement(reply)
    assert verdict["misogynistic"] == 0 and verdict["coherent"] == 1 and verdict["refusal_or_hedge"] == 0 and verdict["explanation"] == "Respectful answer."
    assert judge_mod.parse_judgement(lj.assemble_reply('1, "coherent": 0, "refusal_or_hedge": 0, "explanation": "x"}'))["misogynistic"] == 1


def test_compare_judged_agreement_and_kappa():
    import compare_judged as cj
    base = pd.DataFrame({"direction": "learned", "coef": 0.5, "prompt": "Q", "response": [f"r{i}" for i in range(6)]})
    a, b = base.copy(), base.copy()
    a["misogynistic"], b["misogynistic"] = [1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]
    a["coherent"], b["coherent"] = [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]
    a["refusal_or_hedge"], b["refusal_or_hedge"] = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, None]
    r = cj.compare(a, b).set_index("criterion")
    assert r.loc["misogynistic", "n_rows_both"] == 6 and abs(r.loc["misogynistic", "agreement"] - 5 / 6) < 1e-3 and 0.6 < r.loc["misogynistic", "kappa"] < 0.8
    assert r.loc["coherent", "agreement"] == 1.0 and r.loc["refusal_or_hedge", "n_rows_both"] == 5       # a row without a verdict in one file is skipped
    assert cj.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0 and cj.kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0


def test_quota_marker_for_anthropic_billing_error():
    assert jc.is_quota_error("Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.")


try:
    import anthropic as _anthropic_mod
    _REAL_ANTHROPIC = _anthropic_mod.Anthropic          # the real class, captured before any monkeypatching
except ImportError:                                     # pragma: no cover
    _REAL_ANTHROPIC = None


def _anthropic_client(monkeypatch, handler):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: _REAL_ANTHROPIC(api_key="sk-ant-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw))
    return anthropic, httpx


ANTHROPIC_OK = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "stop_reason": "end_turn", "stop_sequence": None,
                "content": [{"type": "text", "text": "OK"}], "usage": {"input_tokens": 3, "output_tokens": 1}}


def test_robust_anthropic_judge_backoff_billing_and_temperature_fallback(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    slept, calls = [], []
    monkeypatch.setattr(jc.time, "sleep", lambda s: slept.append(s))
    # 1. two short 429s then success: waits for the Retry-After hint
    def limited(req):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"type": "error", "error": {"type": "rate_limit_error", "message": "Number of request tokens has exceeded your per-minute rate limit."}})
        return httpx.Response(200, json=ANTHROPIC_OK)
    _anthropic_client(monkeypatch, limited)
    assert jc.make_robust_anthropic_judge()("hi") == "OK" and slept == [2.0, 2.0]
    # 2. empty balance (400): raised at once, no waiting
    slept.clear()
    def broke(req):
        return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "Your credit balance is too low to access the Anthropic API."}})
    _anthropic_client(monkeypatch, broke)
    with pytest.raises(anthropic.BadRequestError):
        jc.make_robust_anthropic_judge()("hi")
    assert slept == []
    # 3. a model that rejects `temperature`: repeated without it, no waiting
    seen = []
    def picky(req):
        body = req.content.decode()
        seen.append("temperature" in body)
        if "temperature" in body:
            return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` is not supported for this model"}})
        return httpx.Response(200, json=ANTHROPIC_OK)
    _anthropic_client(monkeypatch, picky)
    assert jc.make_robust_anthropic_judge()("hi") == "OK" and seen[-1] is False and slept == []      # newer SDKs refuse the parameter before sending; older ones get a 400 first
    # 4. a long Retry-After: fail at once
    def long_hint(req):
        return httpx.Response(429, headers={"retry-after": "900"}, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    _anthropic_client(monkeypatch, long_hint)
    with pytest.raises(anthropic.RateLimitError):
        jc.make_robust_anthropic_judge()("hi")
    assert slept == []


def test_labels_stay_unique_when_experiments_share_a_file_name(tmp_path):
    names = [("old_v3", "calibration__pooled__misogyny__offtopic.csv"), ("stance_raw", "calibration__pooled__misogyny__offtopic.csv"),
             ("claims_raw", "calibration__pooled__misogyny__all12.csv")]
    files = []
    for folder, name in names:
        (tmp_path / folder).mkdir()
        files.append(str(tmp_path / folder / name))
    assert jc.make_labels(files) == ["old_v3/offtopic", "stance_raw/offtopic", "claims_raw/all12"]
    assert jc.make_labels(["a/x.csv", "b/y.csv"]) == ["x", "y"]                                      # distinct stems are kept as they are
    two = [str(tmp_path / "e1" / "t1" / "c.csv"), str(tmp_path / "e1" / "t2" / "c.csv")]
    assert len(set(jc.make_labels(two))) == 2


def test_from_judged_prints_the_summary_without_a_judge(tmp_path, monkeypatch, capsys):
    j = jc.judge_frame(frame(), fake_judge, "R", tmp_path / "c.jsonl", judge_id="x")
    f = tmp_path / "old_v3__opinion__judged.csv"
    j.to_csv(f, index=False)
    assert jc.label_from_judged_name(f) == "old_v3/opinion" and jc.label_from_judged_name("judged_old_v3_opinion.csv") == "judged_old_v3_opinion"
    monkeypatch.setattr(jc, "make_robust_openai_judge", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a judge must not be created")))
    monkeypatch.setattr("sys.argv", ["judge_csv.py", "--from-judged", "--out", str(tmp_path / "o"), str(f)])
    jc.main()
    out = capsys.readouterr().out
    want = jc.format_summary(jc.summarize(j, "old_v3/opinion"))
    assert want in out and (tmp_path / "o" / "judge_summary.csv").exists() and not list((tmp_path / "o").glob("*__judged.csv"))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"direction": ["learned"], "coef": [0.5], "prompt": ["Q"], "response": ["r"]}).to_csv(bad, index=False)
    monkeypatch.setattr("sys.argv", ["judge_csv.py", "--from-judged", "--out", str(tmp_path / "o2"), str(bad)])
    with pytest.raises(SystemExit):
        jc.main()


def test_max_rows_smoke_limit(tmp_path):
    r = jc.judge_frame(frame(), fake_judge, "RUBRIC", tmp_path / "c.jsonl", max_rows=2)
    assert len(r) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
