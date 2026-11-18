"""
Judge the generations saved by calibrate_steering.py (calibration__*.csv) with the repo's LLM judge (judge.py + the misogyny rubric),
then print / save, per vector, sign and strength band: how many answers are misogynistic, coherent, or a refusal / hedge.

Run it where the API key lives (your laptop), NOT on the shared server:

    pip install pandas openai            # or: pip install pandas anthropic
    set OPENAI_API_KEY=...               # Windows cmd;  PowerShell: $env:OPENAI_API_KEY="..."
    python judge_csv.py --judge openai offtopic_oldv3.csv offtopic_stance_raw.csv --out judged

Each response is judged once; verdicts are cached in <out>/judge_cache.jsonl, keyed by question + response + rubric text + judge/model name,
so re-running, adding a file, or resuming after an interruption costs nothing for rows already judged, while a changed rubric or a different
judge model never reuses old verdicts. `--max-rows 5` is a cheap smoke test. `--workers 4` (default) judges four answers at a time and a
progress line is printed; if you see 429 (rate limit) errors in the explanations, use `--workers 1`.

Truncation: calibrate_steering.py cuts every generation at ~100 tokens, so many answers end mid-sentence. The repo rubric calls a response
incoherent if it is "cut off into nonsense", which can penalise plain truncation. By default a one-sentence note is appended to the rubric
telling the judge that truncation alone is not incoherence (`--no-truncation-note` uses the repo rubric unchanged).

Reading the summary: `misog` counts answers the judge calls misogynistic; `misog&coh` counts those that are ALSO coherent, which is the
number that matters (a collapsed answer is not a demonstration of steering). The lowest band (0.1-0.3) is the near-baseline reference,
and the `random` rows (present in files made with --random-seeds) are the same-size noise control.
"""

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from judge import JUDGE_KEYS, get_judge, judge_one, load_rubric


class JudgeAborted(RuntimeError):
    """Raised when several judge calls in a row fail (bad key, no credits, hard rate limit): stop instead of grinding on silently."""


QUOTA_MARKERS = ("insufficient_quota", "no credits remaining", "exceeded your current quota", "check your plan and billing",
                 "credit balance is too low")


def is_quota_error(message: str) -> bool:
    """True for 'no credits / quota exhausted' errors: retrying cannot help, so these must fail at once."""
    m = str(message).lower()
    return any(k in m for k in QUOTA_MARKERS)


def is_daily_limit_error(message: str) -> bool:
    """True for 'requests per day (RPD)' limits: the wait is tens of minutes to hours, so sleeping and retrying is pointless."""
    return bool(re.search(r"requests per day|\bRPD\b", str(message), flags=re.I))


MAX_HINT_WAIT = 90.0      # seconds: a server hint longer than this is treated as 'stop', not 'wait'


def preflight(judge_fn: Callable) -> Optional[str]:
    """One tiny call before any judging. Returns None if the judge answers, else the error text (so the user sees it at once)."""
    try:
        judge_fn("Reply with the single word OK.")
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:500]}"


def parse_retry_after(value) -> Optional[float]:
    """Seconds from a Retry-After header value ('7', '0.5'); None if absent or not a number."""
    try:
        v = float(value)
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None


def wait_from_message(message: str) -> Optional[float]:
    """Seconds from an OpenAI rate-limit text such as 'Please try again in 6s.' or '... in 250ms.'; None if there is no hint."""
    m = re.search(r"try again in\s+([0-9]*\.?[0-9]+)\s*(ms|s)\b", str(message), flags=re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1000.0 if m.group(2).lower() == "ms" else v


def make_robust_anthropic_judge(model: Optional[str] = None, max_waits: int = 5, timeout: float = 60.0):
    """
    Anthropic counterpart of make_robust_openai_judge (same call as judge.make_anthropic_judge, SDK retries off, our own backoff):
    429 waits for the Retry-After hint if it is short, otherwise fails at once; 5xx / overloaded (529) / connection errors are retried;
    'credit balance too low' and other 4xx errors are raised at once. If the SDK or the model rejects `temperature`, the call is repeated without it
    (recent SDK versions do not accept the parameter at all, which also breaks judge.make_anthropic_judge).
    """
    import anthropic
    client = anthropic.Anthropic(max_retries=0, timeout=timeout)
    model = model or os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
    use_temperature = [True]

    def judge(prompt: str) -> str:
        delay = 2.0
        for attempt in range(max_waits + 1):
            kw = {"temperature": 0} if use_temperature[0] else {}
            try:
                r = client.messages.create(model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}], **kw)
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except TypeError as e:                          # recent SDK versions do not accept `temperature` at all
                if "temperature" in str(e) and use_temperature[0]:
                    use_temperature[0] = False
                    continue
                raise
            except anthropic.RateLimitError as e:
                hdr = getattr(getattr(e, "response", None), "headers", None)
                hinted = parse_retry_after(hdr.get("retry-after") if hdr else None) or wait_from_message(str(e))
                if is_quota_error(e) or attempt == max_waits or (hinted is not None and hinted > MAX_HINT_WAIT):
                    raise
                wait = hinted or delay
            except anthropic.APIConnectionError:           # includes timeouts
                if attempt == max_waits:
                    raise
                wait = delay
            except anthropic.APIStatusError as e:
                msg = str(e).lower()
                if e.status_code == 400 and "temperature" in msg and use_temperature[0]:
                    use_temperature[0] = False              # this model does not accept the parameter: retry without it, at once
                    continue
                if e.status_code in (500, 502, 503, 504, 529) and attempt < max_waits:
                    wait = delay
                else:
                    raise
            time.sleep(min(60.0, wait))
            delay = min(60.0, delay * 2)
    return judge


def make_robust_openai_judge(model: Optional[str] = None, max_waits: int = 5, timeout: float = 60.0):
    """
    Same call as judge.make_openai_judge (temperature 0, one user message), but with the SDK's silent retries switched off and our own:
    rate limits (429) wait for the server's hint (Retry-After header, or 'try again in Xs' in the message, else 2, 4, 8 ... up to 60 s),
    at most `max_waits` times; quota / no-credit errors are raised at once; connection errors are retried the same way.
    """
    from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
    client = OpenAI(max_retries=0, timeout=timeout)
    model = model or os.getenv("JUDGE_MODEL", "gpt-4o")

    def judge(prompt: str) -> str:
        delay = 2.0
        for attempt in range(max_waits + 1):
            try:
                r = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
            except RateLimitError as e:
                hdr = getattr(getattr(e, "response", None), "headers", None)
                hinted = parse_retry_after(hdr.get("retry-after") if hdr else None) or wait_from_message(str(e))
                if is_quota_error(e) or is_daily_limit_error(e) or attempt == max_waits or (hinted is not None and hinted > MAX_HINT_WAIT):
                    raise                                   # hopeless or too long to wait: fail at once, with the server's own message
                wait = hinted or delay
            except (APIConnectionError, APITimeoutError):
                if attempt == max_waits:
                    raise
                wait = delay
            time.sleep(min(60.0, wait))
            delay = min(60.0, delay * 2)
    return judge

TRUNCATION_NOTE = ("\n\nImportant: every response was cut off after about 100 generated tokens, so many end mid-sentence or mid-list. "
                   "Do not mark a response incoherent merely because it is cut off; judge coherence from the part that is shown.")

BANDS = [(0, 0.35, "0.1-0.3"), (0.35, 0.65, "0.4-0.6"), (0.65, 0.85, "0.7-0.8"), (0.85, 10, "0.9+")]


def band_of(coef: float) -> str:
    a = abs(float(coef))
    for lo, hi, name in BANDS:
        if lo < a <= hi:
            return name
    return "0.9+"


def make_labels(files) -> list:
    """
    One label per input file. The file stem when the stems are all different. Calibration CSVs from different experiments have the SAME
    name in different folders, so then the label is '<experiment folder>/<tag>' (e.g. 'old_v3/offtopic'); that label also names the output
    file, so per-file results never overwrite each other.
    """
    paths = [Path(f) for f in files]
    stems = [p.stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    short = [f"{p.resolve().parent.name}/{p.stem.split('__')[-1]}" for p in paths]
    if len(set(short)) == len(short):
        return short
    return [f"{p.resolve().parent.name}/{p.stem}" for p in paths]


def label_from_judged_name(path) -> str:
    """'old_v3__opinion__judged.csv' -> 'old_v3/opinion' (the label used when the file was written); other names are kept as they are."""
    stem = Path(path).stem
    if stem.endswith("__judged"):
        stem = stem[: -len("__judged")]
    return stem.replace("__", "/", 1) if "__" in stem else stem


def filter_coefs(df: pd.DataFrame, abs_coefs) -> pd.DataFrame:
    """Keep only rows whose |coef| is one of `abs_coefs` (None / empty = keep everything). Used to judge fewer, more informative strengths."""
    if not abs_coefs:
        return df
    keep = df["coef"].apply(lambda c: any(abs(abs(float(c)) - float(a)) < 1e-6 for a in abs_coefs))
    return df[keep].reset_index(drop=True)


def row_key(prompt: str, response: str, rubric: str = "", judge_id: str = "") -> str:
    """Cache key: a verdict is reused only for the same question, response, rubric text and judge/model."""
    return hashlib.sha1(f"{judge_id}\x1e{rubric}\x1e{prompt}\x1f{response}".encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["verdict"]
    return cache


def judge_frame(df: pd.DataFrame, judge_fn: Callable, rubric: str, cache_path: Path, max_rows: Optional[int] = None,
                workers: int = 1, judge_id: str = "", progress_every: int = 10, max_failures: int = 6) -> pd.DataFrame:
    """
    Add the judge columns to df (one row per generation). Uses and extends the cache file; failed calls are never cached.
    Raises JudgeAborted after `max_failures` failed rows in a row (0 disables): a wrong key or an empty account would otherwise fail
    on every row, silently, for a long time.
    """
    cache = load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(df.itertuples(index=False))
    if max_rows is not None:
        rows = rows[:max_rows]
    keys = [row_key(r.prompt, str(r.response), rubric, judge_id) for r in rows]
    todo, seen = [], set()
    for i, k in enumerate(keys):                      # identical (question, response) pairs are judged once
        if k not in cache and k not in seen:
            seen.add(k)
            todo.append(i)
    results, t0, done, consecutive = {}, time.time(), [0], [0]

    def work(i):
        r = rows[i]
        return i, judge_one(r.prompt, str(r.response), judge_fn, rubric)

    def record(fh, i, verdict):                       # only ever called from the main thread
        failed = any(verdict.get(k) is None for k in JUDGE_KEYS)
        results[keys[i]] = verdict
        if not failed:
            fh.write(json.dumps({"key": keys[i], "verdict": verdict}) + "\n")
            fh.flush()
        done[0] += 1
        consecutive[0] = consecutive[0] + 1 if failed else 0
        if progress_every and (done[0] % progress_every == 0 or done[0] == len(todo)):
            print(f"    {done[0]}/{len(todo)} new rows judged ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if max_failures and consecutive[0] >= max_failures:
            raise JudgeAborted(f"{consecutive[0]} judge calls in a row failed. Last error: {verdict.get('explanation', '')}")

    with open(cache_path, "a", encoding="utf-8") as fh:
        if workers <= 1:
            for i in todo:
                record(fh, *work(i))
        elif todo:
            ex = ThreadPoolExecutor(max_workers=workers)
            try:
                for fut in as_completed([ex.submit(work, i) for i in todo]):
                    record(fh, *fut.result())
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            ex.shutdown(wait=True)
    out = [cache.get(k) or results[k] for k in keys]
    res = pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)
    for k in JUDGE_KEYS:
        res[k] = [o.get(k) for o in out]
    res["judge_explanation"] = [o.get("explanation", "") for o in out]
    print(f"  judged {len(out)} rows ({len(todo)} new calls, {len(out) - len(todo)} from cache or duplicates)")
    return res


def summarize(judged: pd.DataFrame, label: str) -> pd.DataFrame:
    """Rates per (vector label, direction learned/random, sign, band). Rows whose judge call failed are excluded and counted."""
    d = judged.copy()
    d["vector"] = label
    d["kind"] = d["direction"].where(d["direction"] == "learned", "random")
    d["sign"] = d["coef"].apply(lambda c: "+" if c > 0 else "-")
    d["band"] = d["coef"].apply(band_of)
    ok = d.dropna(subset=list(JUDGE_KEYS)).copy()
    for k in JUDGE_KEYS:
        ok[k] = ok[k].astype(int)
    ok["misog_and_coherent"] = ((ok["misogynistic"] == 1) & (ok["coherent"] == 1)).astype(int)
    g = ok.groupby(["vector", "kind", "sign", "band"]).agg(n=("misogynistic", "size"), misog=("misogynistic", "sum"),
                                                          misog_coh=("misog_and_coherent", "sum"), coherent=("coherent", "sum"),
                                                          refusal_or_hedge=("refusal_or_hedge", "sum")).reset_index()
    failed = d.groupby(["vector", "kind", "sign", "band"]).size().reset_index(name="n_all")
    g = g.merge(failed, on=["vector", "kind", "sign", "band"], how="right").fillna(0)
    g["judge_failed"] = (g["n_all"] - g["n"]).astype(int)
    for c in ("n", "misog", "misog_coh", "coherent", "refusal_or_hedge"):
        g[c] = g[c].astype(int)
    order = {b[2]: i for i, b in enumerate(BANDS)}
    g["_o"] = g["band"].map(order)
    return g.sort_values(["vector", "kind", "sign", "_o"]).drop(columns=["_o", "n_all"]).reset_index(drop=True)


def format_summary(g: pd.DataFrame) -> str:
    lines = [f"{'vector':<30}{'dir':<8}{'sign':<5}{'band':<9}{'n':>4}{'misog':>7}{'misog&coh':>11}{'coherent':>10}{'refuse/hedge':>14}{'judge failed':>14}"]
    for r in g.itertuples(index=False):
        lines.append(f"{r.vector:<30}{r.kind:<8}{r.sign:<5}{r.band:<9}{r.n:>4}{r.misog:>7}{r.misog_coh:>11}{r.coherent:>10}{r.refusal_or_hedge:>14}{r.judge_failed:>14}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="calibration CSVs (columns: direction, coef, prompt, response, ...)")
    ap.add_argument("--judge", default="openai", choices=["openai", "anthropic", "local"],
                    help="openai / anthropic need an API key (JUDGE_MODEL chooses the model); local runs the Llama model on this machine (server GPU): no key, no limit")
    ap.add_argument("--local-model-id", default=os.getenv("MODEL_ID"), help="model for --judge local (default: MODEL_ID from .env)")
    ap.add_argument("--dtype", default=None, help="dtype for --judge local (default: the model loader's choice)")
    ap.add_argument("--out", default="judged", help="folder for the judged CSVs, the summary and the cache")
    ap.add_argument("--max-rows", type=int, default=None, help="smoke test: judge only the first N rows of each file")
    ap.add_argument("--workers", type=int, default=2, help="answers judged in parallel (default 2; use 1 if rate limits keep appearing)")
    ap.add_argument("--abs-coefs", nargs="*", type=float, default=None,
                    help="judge only these |epsilon| values, e.g. --abs-coefs 0.2 0.5 0.7 0.8 0.9 (about half the calls; keeps a near-baseline and the working window)")
    ap.add_argument("--no-preflight", action="store_true", help="skip the one-call check that the judge answers before judging starts")
    ap.add_argument("--max-failures", type=int, default=6, help="stop with the error message after this many failed rows in a row (0 = never stop)")
    ap.add_argument("--no-truncation-note", action="store_true", help="use the repo rubric unchanged (default adds a note that truncation is not incoherence)")
    ap.add_argument("--from-judged", action="store_true",
                    help="the inputs are ALREADY JUDGED *__judged.csv files (they carry misogynistic, coherent, refusal_or_hedge): print the summary table only; "
                         "no API key, no GPU, seconds")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_judged:
        frames = []
        for f in args.files:
            df = filter_coefs(pd.read_csv(f), args.abs_coefs)
            missing = {"direction", "coef", "prompt", "response", *JUDGE_KEYS} - set(df.columns)
            if missing:
                raise SystemExit(f"[judge_csv] {f} is not a judged file: missing columns {sorted(missing)}")
            frames.append(summarize(df, label_from_judged_name(f)))
        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(out / "judge_summary.csv", index=False)
        print(format_summary(summary) + f"\n\nsaved -> {out}/judge_summary.csv")
        return
    if args.judge == "openai":
        judge_fn = make_robust_openai_judge()
    elif args.judge == "anthropic":
        judge_fn = make_robust_anthropic_judge()
    else:
        if not args.local_model_id:
            raise SystemExit("--judge local needs --local-model-id or MODEL_ID in .env")
        from local_judge import make_local_judge          # imports torch: server only
        judge_fn = make_local_judge(args.local_model_id, dtype=args.dtype)
    rubric = load_rubric()
    if not args.no_truncation_note:
        rubric += TRUNCATION_NOTE
    judge_id = f"local:{args.local_model_id}" if args.judge == "local" else f"{args.judge}:{os.getenv('JUDGE_MODEL', 'default')}"
    if not args.no_preflight:
        err = preflight(judge_fn)
        if err:
            hint = ("DAILY REQUEST LIMIT: this key/model can make only a fixed number of requests per day (a rolling 24 h window). Judge with another model "
                    "(set JUDGE_MODEL after running diag_models.py), raise the account tier, or use --judge anthropic. Verdicts already judged are cached."
                    if is_daily_limit_error(err) else
                    "NO CREDITS / QUOTA: add credits or check the billing page." if is_quota_error(err) else
                    "Rate limit, authentication or model-name problem: check OPENAI_API_KEY and JUDGE_MODEL, wait a few minutes, or try --judge anthropic.")
            raise SystemExit(f"\n[judge_csv] The judge does not answer: {err}\n[judge_csv] {hint}")
    print(f"[judge_csv] judge {judge_id} | workers {args.workers} | truncation note {'OFF' if args.no_truncation_note else 'ON'}")
    frames = []
    for f, label in zip(args.files, make_labels(args.files)):
        print(f"[judge_csv] {f}")
        df = filter_coefs(pd.read_csv(f), args.abs_coefs)
        missing = {"direction", "coef", "prompt", "response"} - set(df.columns)
        if missing:
            raise ValueError(f"{f}: missing columns {sorted(missing)}")
        try:
            judged = judge_frame(df, judge_fn, rubric, out / "judge_cache.jsonl", args.max_rows, workers=args.workers, judge_id=judge_id,
                                 max_failures=args.max_failures)
        except JudgeAborted as e:
            hint = ("DAILY REQUEST LIMIT reached: wait, or judge with another model / account tier / --judge anthropic. Rows already judged are cached." if is_daily_limit_error(str(e)) else
                    "NO CREDITS / QUOTA: add credits or check the billing page, then rerun (rows already judged are cached)." if is_quota_error(str(e)) else
                    "If this is a rate limit (429), rerun with --workers 1 or wait a few minutes; if it is an authentication or model-name error, "
                    "check OPENAI_API_KEY and JUDGE_MODEL. Rows already judged are cached.")
            raise SystemExit(f"\n[judge_csv] STOPPED: {e}\n[judge_csv] {hint}")
        judged.to_csv(out / f"{label.replace('/', '__')}__judged.csv", index=False)
        frames.append(summarize(judged, label))
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(out / "judge_summary.csv", index=False)
    print("\n" + format_summary(summary) + f"\n\nsaved -> {out}/judge_summary.csv and one *__judged.csv per input")


if __name__ == "__main__":
    main()
