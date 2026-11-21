"""Opt-in wall-clock component timing for one content/style inference call.

Only this pipeline instance is wrapped, and methods are restored even on failure.
No latents are cached and no model outputs or inputs are changed.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from functools import wraps
import statistics
import time

import torch


COMPONENTS = ("content_vae", "style_vae", "dit", "decode")


def _synchronize(device):
    device = torch.device(device)
    if device.type == "cpu":
        return
    if device.type == "mps":
        torch.mps.synchronize()
    else:
        backend = getattr(torch, device.type, None)
        if backend is None or not hasattr(backend, "synchronize"):
            raise ValueError(f"Component timing cannot synchronize device {device}")
        backend.synchronize(device)


@contextmanager
def _wrap_method(obj, name, invoke):
    original = getattr(obj, name)
    had_instance_attribute = name in vars(obj)

    @wraps(original)
    def wrapped(*args, **kwargs):
        return invoke(original, *args, **kwargs)

    setattr(obj, name, wrapped)
    try:
        yield
    finally:
        if had_instance_attribute:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


def profile_inference(pipe, generate, device):
    """Time the unchanged call; encode order must be [content, style], without input_image."""
    durations = {name: [] for name in COMPONENTS}
    vae_inputs, style_latent = {}, {}

    def timed(name, method, *args, **kwargs):
        _synchronize(device)
        start = time.perf_counter()
        result = method(*args, **kwargs)
        _synchronize(device)
        durations[name].append(time.perf_counter() - start)
        return result

    def encode(method, image, *args, **kwargs):
        index = len(durations["content_vae"]) + len(durations["style_vae"])
        if index >= 2:
            raise ValueError("Component timing expects exactly two VAE encodes: content, then style")
        role = ("content", "style")[index]
        result = timed(role + "_vae", method, image, *args, **kwargs)
        vae_inputs[role] = {"shape": list(image.shape), "dtype": str(image.dtype)}
        if role == "style":
            style_latent.update(shape=list(result.shape), dtype=str(result.dtype),
                                bytes=result.numel() * result.element_size())
        return result

    with ExitStack() as stack:
        stack.enter_context(_wrap_method(pipe.vae, "encode", encode))
        stack.enter_context(_wrap_method(pipe.vae, "decode", lambda method, *a, **kw: timed("decode", method, *a, **kw)))
        stack.enter_context(_wrap_method(pipe.dit, "forward", lambda method, *a, **kw: timed("dit", method, *a, **kw)))
        _synchronize(device)
        start = time.perf_counter()
        result = generate()
        _synchronize(device)
        total = time.perf_counter() - start

    counts = {name: len(values) for name, values in durations.items()}
    if counts["content_vae"] != 1 or counts["style_vae"] != 1 or counts["decode"] != 1 or counts["dit"] < 1:
        raise ValueError(f"Unexpected pipeline calls for component timing: {counts}")
    seconds = {name: sum(values) for name, values in durations.items()}
    seconds["other"] = max(0.0, total - sum(seconds.values()))
    seconds["total"] = total
    return result, {"seconds": seconds, "calls": counts, "dit_forward_seconds": durations["dit"],
                    "vae_inputs": vae_inputs, "style_latent": style_latent}


def summarize_profiles(samples):
    report = {
        "method": "Wall-clock seconds with device synchronization at each component boundary; "
                  "includes profiling overhead. Warmup and failed/skipped pairs are excluded.",
        "scope": "Total covers one pipeline call, excluding model loading, reference file I/O and output saving. "
                 "VAE times cover encode/decode only. DiT is the sum of all forward calls. "
                 "Other includes preprocessing, text conditioning, scheduler work and transfers outside these calls.",
        "sample_count": len(samples), "samples": samples,
        "mean_seconds": None, "percent_of_total": None, "style_vae_cache_estimate": None,
    }
    if not samples:
        return report
    means = {name: statistics.mean(sample["seconds"][name] for sample in samples)
             for name in (*COMPONENTS, "other", "total")}
    total, saved = means["total"], means["style_vae"]
    hit_total = total - saved
    report.update(
        mean_seconds=means,
        percent_of_total={name: 100 * means[name] / total if total > 0 else None for name in (*COMPONENTS, "other")},
        style_vae_cache_estimate={
            "scope": "Ideal steady-state cache hit: subtract only style VAE encode time. "
                     "Assumes zero cache read/transfer cost; excludes first-use cache construction. "
                     "All DiT style-token computation remains. No VAE cache was used in this run.",
            "saved_seconds": saved, "estimated_total_seconds": hit_total,
            "saved_percent": 100 * saved / total if total > 0 else None,
            "estimated_speedup": total / hit_total if hit_total > 0 else None,
        },
    )
    return report


def print_profile_summary(report):
    if not report["sample_count"]:
        return
    means = report["mean_seconds"]
    print("Component timing (mean per generated image; synchronized wall time):")
    for name in (*COMPONENTS, "other"):
        print(f"  {name:12s} {means[name]:9.4f} s  {report['percent_of_total'][name]:6.2f}%")
    print(f"  {'total':12s} {means['total']:9.4f} s")
    estimate = report["style_vae_cache_estimate"]
    print(f"Ideal style VAE cache hit: {estimate['estimated_total_seconds']:.4f} s, "
          f"saving {estimate['saved_seconds']:.4f} s ({estimate['saved_percent']:.2f}%). "
          "Estimate only; no cache was used.")
