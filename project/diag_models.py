"""
Find which judge model your OpenAI key can actually use, and how many requests per day / minute each allows.
One tiny call per candidate model (each call counts against THAT model's own limit). No retries, nothing is saved.

    python diag_models.py                     # tries the chat models your account lists whose name suggests a general-purpose GPT
    python diag_models.py gpt-4o-mini gpt-4.1-mini        # or name the models yourself

Reading the table: `req limit` is the number of requests allowed per window (per day on the free tier: 50 means 50 per DAY); `req left` is what
remains. A model with a limit of a few hundred or more per day is usable for judge_csv.py:  set JUDGE_MODEL=<that model>  and rerun.
"""
import sys
import time

from openai import OpenAI

SKIP = ("audio", "realtime", "tts", "whisper", "transcribe", "image", "embedding", "moderation", "search", "codex", "instruct", "davinci", "babbage", "dall")


def candidate_models(client) -> list:
    ids = sorted(m.id for m in client.models.list())
    keep = [i for i in ids if i.startswith(("gpt-", "o")) and not any(k in i for k in SKIP) and not i[-4:].isdigit() and "-20" not in i]
    return keep[:12]


def probe(client, model: str) -> dict:
    msgs = [{"role": "user", "content": "Reply with OK"}]
    for kw in ({"max_tokens": 5}, {"max_completion_tokens": 16}):
        try:
            r = client.chat.completions.with_raw_response.create(model=model, messages=msgs, **kw)
            h = r.headers
            return {"model": model, "status": "ok", "req_limit": h.get("x-ratelimit-limit-requests"), "req_left": h.get("x-ratelimit-remaining-requests"),
                    "tok_limit": h.get("x-ratelimit-limit-tokens"), "note": ""}
        except Exception as e:
            msg = str(e)
            if "max_tokens" in msg or "max_completion_tokens" in msg:
                continue                                               # newer models want the other parameter name: try it
            h = getattr(getattr(e, "response", None), "headers", {}) or {}
            return {"model": model, "status": type(e).__name__, "req_limit": h.get("x-ratelimit-limit-requests"), "req_left": h.get("x-ratelimit-remaining-requests"),
                    "tok_limit": h.get("x-ratelimit-limit-tokens"), "note": msg[:110].replace("\n", " ")}
    return {"model": model, "status": "unsupported-parameters", "req_limit": None, "req_left": None, "tok_limit": None, "note": ""}


def main(argv) -> None:
    client = OpenAI(max_retries=0, timeout=30)
    models = argv or candidate_models(client)
    print(f"{'model':<28}{'status':<22}{'req limit':>10}{'req left':>10}{'tok limit':>11}  note")
    for m in models:
        r = probe(client, m)
        print(f"{r['model']:<28}{r['status']:<22}{str(r['req_limit']):>10}{str(r['req_left']):>10}{str(r['tok_limit']):>11}  {r['note']}")
        time.sleep(0.5)


if __name__ == "__main__":
    main(sys.argv[1:])
