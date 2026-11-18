"""
Small GPU helpers for shared servers (torch only; no transformers).

Two things matter on a machine where someone else may be using a GPU:
  * With several GPUs visible, device_map="auto" shards a model across ALL of them (including a GPU that is busy), and
    the RFM step silently uses cuda:0. Pin the GPU you want BEFORE starting python:
        CUDA_VISIBLE_DEVICES=1 python activation_extraction.py ...
  * Check `nvidia-smi` first to see which GPU is free.
"""

import os

import torch


def n_visible_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def warn_if_multi_gpu(what: str = "this step") -> bool:
    """Print a warning (and return True) if several GPUs are visible and none was pinned with CUDA_VISIBLE_DEVICES."""
    n = n_visible_gpus()
    if n > 1 and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"\n[gpu] WARNING: {n} GPUs are visible {names} and CUDA_VISIBLE_DEVICES is not set, so {what} may use or "
              f"spread over ALL of them, including one that someone else is using.\n"
              f"      Check `nvidia-smi`, then pin a free GPU, e.g.:  CUDA_VISIBLE_DEVICES=1 python <script>.py ...\n")
        return True
    return False


def describe_device() -> str:
    """'cuda:0 NVIDIA L4 (22.4 GB free of 23.0 GB)' or 'cpu'. Only touches device 0 of the VISIBLE devices."""
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info(0)
    return f"cuda:0 {torch.cuda.get_device_name(0)} ({free / 1e9:.1f} GB free of {total / 1e9:.1f} GB)"
