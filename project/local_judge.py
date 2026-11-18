"""
Local judge: the same rubric, run by the Llama model that is already on the server. No API, no key, no rate limit, no cost.

    python judge_csv.py --judge local --out judged_local  calibration_*.csv        (on the server, GPU)

How it works: the rubric prompt of judge.py is wrapped in the model's chat template and the assistant turn is PRE-FILLED with
`{"misogynistic": `, so the model can only continue a JSON object; decoding is greedy; the reply is cut after the first closing brace.
Weaknesses to know: an 8B model is a weaker judge than a frontier model on borderline answers. Use it for the decision at hand (blatant
misogyny, coherence, refusals), and measure its agreement with another judge on a sample with compare_judged.py before trusting small differences.
"""
from typing import Optional

PREFILL = '{"misogynistic": '


def cut_after_first_json(text: str) -> str:
    """Keep the text up to and including the first balanced '}' (the judge may keep writing after its JSON); unchanged if there is none."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth <= 0:
                return text[: i + 1]
    return text


def assemble_reply(generated: str) -> str:
    """Prefill + what the model generated, cut after the first JSON object: the string judge.parse_judgement expects."""
    return cut_after_first_json(PREFILL + generated)


def make_local_judge(model_id: str, dtype: Optional[str] = None, max_new_tokens: int = 120):
    """Load the model once; return judge(prompt) -> reply text. Imports torch / transformers only here."""
    from model_utils import load_causal_lm
    from prompting import format_chat_prompt

    model, tokenizer = load_causal_lm(model_id, dtype=dtype)

    def judge(prompt: str) -> str:
        formatted = format_chat_prompt(tokenizer, prompt) + PREFILL
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, max_length=None, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return assemble_reply(generated)

    return judge
