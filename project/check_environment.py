"""
Read-only check of the machine before running the MTP pipeline. It changes nothing.

    cd mtp_v5
    export CUDA_DEVICE_ORDER=PCI_BUS_ID      # make CUDA's GPU numbers match nvidia-smi's
    export CUDA_VISIBLE_DEVICES=1            # a FREE GPU, chosen from nvidia-smi
    python check_environment.py

It reports [OK] / [WARN] / [FAIL] / [INFO] lines for: Python and virtual environment, required packages, the GPU you pinned
(name, free memory), disk space, RAM, the Hugging Face cache, and internet access to huggingface.co.

It deliberately does NOT query GPU memory when several GPUs are visible and none is pinned, because that would create a
context on every GPU, including one that someone else is using.
"""

import importlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REQUIRED = ["torch", "transformers", "accelerate", "sklearn", "pandas", "numpy", "dotenv", "tqdm"]
OPTIONAL = {
    "tokenizers": "comes with transformers (the tests need it)",
    "torcheval": "used by rfm.py if installed",
    "pytest": "only for the tests",
    "openai": "only for steering_eval.py --judge openai",
    "anthropic": "only for steering_eval.py --judge anthropic",
}
MIN_GPU_FREE_GB = 20.0    # Llama-3.1-8B in float16: about 16 GB of weights + 1-2 GB working memory (an estimate)
MIN_DISK_GB = 30.0        # about 16 GB for the model files + activations + room to spare
MIN_RAM_GB = 20.0

_levels = []


def report(level: str, msg: str) -> None:
    print(f"[{level:<4}] {msg}")
    _levels.append(level)


def parse_version(v: str) -> tuple:
    """'4.56.1' -> (4, 56); anything unparsable -> (0, 0)."""
    try:
        parts = v.replace("+", ".").split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 0)


def free_gb(path: Path) -> float:
    p = Path(path).expanduser()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def check_python() -> None:
    report("INFO", f"python: {sys.executable}  (version {sys.version.split()[0]})")
    in_venv = sys.prefix != sys.base_prefix
    report("OK" if in_venv else "WARN",
           "running inside a virtual environment" if in_venv else
           "NOT inside a virtual environment (activate the project's venv first)")
    if os.environ.get("CONDA_DEFAULT_ENV"):
        report("INFO", f"conda environment also active: {os.environ['CONDA_DEFAULT_ENV']} (fine as long as the python path above is the venv's)")
    if sys.version_info < (3, 9):
        report("FAIL", "Python 3.9 or newer is required")


def check_packages() -> dict:
    mods = {}
    for name in REQUIRED:
        try:
            m = importlib.import_module(name)
            mods[name] = m
            report("OK", f"{name:<13} {getattr(m, '__version__', '')}")
        except Exception as e:
            report("FAIL", f"{name:<13} missing or broken ({type(e).__name__}: {e}) -> pip install -r requirements.txt")
    for name, why in OPTIONAL.items():
        try:
            m = importlib.import_module(name)
            report("OK", f"{name:<13} {getattr(m, '__version__', '')}  (optional)")
        except Exception:
            report("INFO", f"{name:<13} not installed (optional: {why})")
    tf = mods.get("transformers")
    if tf is not None and parse_version(tf.__version__) < (4, 56):
        report("WARN", f"transformers {tf.__version__}: the code passes dtype= to from_pretrained, which needs a recent "
                       f"transformers (about 4.56 or newer); older versions call it torch_dtype")
    return mods


def check_gpu(mods: dict) -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    order = os.environ.get("CUDA_DEVICE_ORDER")
    report("INFO", f"CUDA_VISIBLE_DEVICES={vis!r}  CUDA_DEVICE_ORDER={order!r}")
    torch = mods.get("torch")
    if torch is None:
        return
    report("INFO", f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        report("FAIL", "torch cannot see a GPU (torch.cuda.is_available() is False): a CPU-only torch build, a driver problem, "
                       "or CUDA_VISIBLE_DEVICES points at a GPU that does not exist")
        return
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    report("INFO", f"{n} visible GPU(s): {names}")
    if n > 1 and not vis:
        report("WARN", "several GPUs are visible and none is pinned, so memory was NOT queried (that would touch every GPU, "
                       "including one that may be busy). Look at nvidia-smi, then run: export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
                       "export CUDA_VISIBLE_DEVICES=<index of a free GPU>; and re-run this check.")
        return
    if vis and order != "PCI_BUS_ID":
        report("WARN", "CUDA_DEVICE_ORDER is not PCI_BUS_ID, so CUDA's GPU numbers may differ from nvidia-smi's. "
                       "Run: export CUDA_DEVICE_ORDER=PCI_BUS_ID (before pinning) and re-run.")
    free, total = torch.cuda.mem_get_info(0)
    free_g, total_g = free / 1e9, total / 1e9
    report("INFO", f"the GPU this process would use: {names[0]} | {free_g:.1f} GB free of {total_g:.1f} GB "
                   f"(cross-check with nvidia-smi: this should be the idle GPU)")
    report("OK" if free_g >= MIN_GPU_FREE_GB else "WARN",
           f"free GPU memory {free_g:.1f} GB (Llama-3.1-8B in float16 needs about 18 GB; need >= {MIN_GPU_FREE_GB:.0f} GB free)")
    try:
        report("OK" if torch.cuda.is_bf16_supported() else "INFO",
               "bfloat16 supported" if torch.cuda.is_bf16_supported() else "bfloat16 not supported (use --dtype float16)")
    except Exception:
        pass


def hf_hub_dir() -> Path:
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def check_disk_ram_cpu() -> None:
    here = Path(__file__).resolve().parent
    g = free_gb(here)
    report("OK" if g >= MIN_DISK_GB else "WARN", f"free disk where the project lives ({here}): {g:.0f} GB (want >= {MIN_DISK_GB:.0f} GB)")
    hub = hf_hub_dir()
    g2 = free_gb(hub)
    report("OK" if g2 >= MIN_DISK_GB else "WARN",
           f"free disk where models are cached ({hub}): {g2:.0f} GB (want >= {MIN_DISK_GB:.0f} GB unless the model is already cached; "
           f"to move the cache: export HF_HOME=/path/with/space)")
    try:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        avail = mem["MemAvailable"] / 1e6
        report("OK" if avail >= MIN_RAM_GB else "WARN", f"available RAM: {avail:.0f} GB (want >= {MIN_RAM_GB:.0f} GB)")
    except Exception:
        report("INFO", "could not read /proc/meminfo (not Linux?)")
    report("INFO", f"CPU cores: {os.cpu_count()}")


def check_hf() -> None:
    hub = hf_hub_dir()
    cached = sorted(p.name[len("models--"):].replace("--", "/") for p in hub.glob("models--*")) if hub.exists() else []
    report("INFO", f"models already in the cache: {cached if cached else 'none'}")
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
        report("OK", "huggingface.co is reachable")
    except Exception as e:
        report("WARN", f"huggingface.co NOT reachable ({type(e).__name__}); fine only if every model you need is already cached")


def main() -> int:
    print("== Python ==");        check_python()
    print("\n== Packages ==");    mods = check_packages()
    print("\n== GPU ==");         check_gpu(mods)
    print("\n== Disk / RAM / CPU =="); check_disk_ram_cpu()
    print("\n== Hugging Face ==");  check_hf()
    print(f"\nSUMMARY: {_levels.count('OK')} OK, {_levels.count('WARN')} WARN, {_levels.count('FAIL')} FAIL "
          f"(INFO lines are context, not problems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
