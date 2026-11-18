"""
Model-agnostic plumbing on three REAL architectures (tiny random weights, no downloads):
    Llama (model.model.layers), GPT-NeoX (gpt_neox.layers), GPT-2 (transformer.h)
plus a custom model with an unusual layer-list name (found by the scanning fallback) and a tokenizer without a chat template.
"""
import _path  # noqa: F401
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

import activation_extraction as ae
import anti_misogyny_steering as ams
import model_utils as mu
import prompting
import _tiny

TEXTS = ["What are your thoughts on the following statement? \nStatement: Life is what you make of it.",
         "Speak as someone who fully endorses hostile sexism.  What are your thoughts on the following statement?",
         "Take on the role of someone who strongly believes in minimalism. Truth is objective."]
_CACHE = {}


def _tok():
    if "tok" not in _CACHE:
        _CACHE["tok"] = _tiny.make_tokenizer(TEXTS * 20)
    return _CACHE["tok"]


def _models():
    tok = _tok()
    return {"llama": _tiny.make_llama(tok), "neox": _tiny.make_neox(tok), "gpt2": _tiny.make_gpt2(tok)}


def test_layer_lookup_sizes_and_selection_on_three_architectures():
    for name, model in _models().items():
        layers = mu.get_decoder_layers(model)
        assert len(layers) == 4, name
        assert mu.get_num_layers(model) == 4 and mu.get_hidden_size(model) == 32, name
        assert ae.select_layers(model) == [-1, -2, -3], name


def test_scanning_fallback_finds_unusual_layer_lists():
    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"num_hidden_layers": 3, "hidden_size": 16})()
            self.encoder_stack = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
            self.other = nn.ModuleList([nn.Linear(16, 16)])
    assert len(mu.get_decoder_layers(Weird())) == 3

    class Hopeless(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"num_hidden_layers": 5})()
            self.lin = nn.Linear(2, 2)
    try:
        mu.get_decoder_layers(Hopeless())
        raise AssertionError("should have raised")
    except AttributeError as e:
        assert "Could not locate" in str(e)


def test_hooks_change_logits_and_are_removed_on_every_architecture():
    tok = _tok()
    for name, model in _models().items():
        ids = tok(prompting.format_chat_prompt(tok, "What do you think?"), return_tensors="pt", add_special_tokens=False)
        ids = {"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"]}
        directions = {l: torch.randn(1, 32) for l in (-1, -2, -3)}
        with torch.no_grad():
            base = model(**ids).logits
            hooks = ams.hook_model(model, directions, [-1, -2, -3], 3.0)
            steered = model(**ids).logits
            ams.clear_hooks(hooks)
            after = model(**ids).logits
        assert not torch.allclose(base, steered, atol=1e-4), f"{name}: hook had no effect"
        assert torch.allclose(base, after), f"{name}: hooks were not removed"


def test_extraction_shapes_and_position_on_every_architecture():
    tok = _tok()
    prompts = [prompting.format_chat_prompt(tok, t) for t in TEXTS]
    for name, model in _models().items():
        out = ae.extract_hidden_states(prompts, model, tok, [-1, -2, -3], batch_size=2)
        assert sorted(out) == [-3, -2, -1] and all(v.shape == (3, 32) for v in out.values()), name
        # batching/padding must not change a prompt's activation (left padding keeps the last token last)
        single = ae.extract_hidden_states([prompts[0]], model, tok, [-2], batch_size=1)[-2][0]
        assert torch.allclose(out[-2][0].float(), single.float(), atol=1e-3), f"{name}: padding changed the activation"


def test_tokenizer_without_chat_template_falls_back_to_plain_prompt():
    tok = _tiny.make_tokenizer(TEXTS * 20, with_template=False)
    assert not prompting.has_chat_template(tok)
    out = prompting.format_chat_prompt(tok, "  hello there \n")
    assert out == "<|begin_of_text|>hello there"
    ids = tok(out, add_special_tokens=False)["input_ids"]
    assert ids.count(tok.bos_token_id) == 1


def test_parse_dtype_and_extract_response_only():
    assert mu.parse_dtype("bf16") == torch.bfloat16 and mu.parse_dtype("float16") == torch.float16
    assert mu.parse_dtype(torch.float32) == torch.float32 and mu.parse_dtype("auto") in (torch.float16, torch.float32)
    try:
        mu.parse_dtype("int3")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    assert ams.extract_response_only(prompt + "Hello there<|im_end|>", prompt) == "Hello there"
    llama = "<|begin_of_text|>...<|start_header_id|>assistant<|end_header_id|>\n\nSure.<|eot_id|>"
    assert ams.extract_response_only(llama) == "Sure."
    assert ams.extract_response_only("plain text, no markers") == "plain text, no markers"


def test_gpu_helpers_are_safe_on_cpu_only_machines():
    import gpu_utils
    if not torch.cuda.is_available():
        assert gpu_utils.n_visible_gpus() == 0 and gpu_utils.describe_device() == "cpu"
        assert gpu_utils.warn_if_multi_gpu() is False


def test_check_environment_script_runs_and_parses_versions():
    import subprocess
    import sys
    import check_environment as ce
    assert ce.parse_version("4.56.1") == (4, 56) and ce.parse_version("2.14.1+cu130") == (2, 14) and ce.parse_version("garbage") == (0, 0)
    root = _path.ROOT
    out = subprocess.run([sys.executable, str(root / "check_environment.py")], capture_output=True, text=True, timeout=180, cwd=str(root))
    assert out.returncode == 0 and "SUMMARY:" in out.stdout and "== GPU ==" in out.stdout, out.stdout + out.stderr


def test_dry_run_indices_are_representative_and_include_the_longest_prompt():
    tok = _tok()
    prompts = ["short"] * 90 + ["a much much longer prompt " * 12] + ["short"] * 9      # the longest sits at index 90
    assert ae._dry_run_indices(prompts, tok, 1000) == list(range(100))                  # limit >= n -> everything
    idx = ae._dry_run_indices(prompts, tok, 10)
    assert idx == sorted(set(idx)) and len(idx) <= 10 and 90 in idx and idx[0] == 0 and idx[-1] >= 95


def test_report_placement_warns_only_when_part_of_the_model_is_offloaded():
    import contextlib
    import io
    from unittest import mock

    class Fake:
        hf_device_map = {"model.embed": 0, "model.layers.0": 0, "model.layers.1": "cpu", "lm_head": "cpu"}

    class Plain:
        pass

    assert mu.report_placement(Plain()) == {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), mock.patch("torch.cuda.is_available", return_value=True):
        counts = mu.report_placement(Fake())
    assert counts == {"0": 2, "cpu": 2} and "WARNING" in buf.getvalue() and "MUCH slower" in buf.getvalue()

    class Good:
        hf_device_map = {"model": 0, "lm_head": 0}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), mock.patch("torch.cuda.is_available", return_value=True):
        assert mu.report_placement(Good()) == {"0": 2}
    assert "WARNING" not in buf.getvalue()


def test_prompting_is_the_single_source_of_truth_for_the_template():
    assert ams.format_chat_prompt is prompting.format_chat_prompt


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} model-agnostic tests passed")
