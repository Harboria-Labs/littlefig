#!/usr/bin/env python3
"""P3a: isolate FigLinear lowram dequantization and CPU RSS retention."""

import argparse
import ctypes
import gc
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import psutil
import torch

from little_fig.engine.figquant import FigQuantTensor
from little_fig.engine.linear import FigLinear


TINYLLAMA_SHAPES = {
    "q_proj": (2048, 2048),
    "mlp_proj": (5632, 2048),
}


def log(message):
    print(f"[P3a] {message}", flush=True)


def rss_mib():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def malloc_trim():
    if platform.system() != "Linux":
        return {"available": False, "called": False, "result": None}
    try:
        libc = ctypes.CDLL("libc.so.6")
        result = int(libc.malloc_trim(0))
        return {"available": True, "called": True, "result": result}
    except Exception as error:
        return {"available": False, "called": False, "error": repr(error)}


def make_layer(out_features, in_features, group_size=128, lora_r=16):
    """Create exact-shape deterministic FigQuant storage without model download."""
    numel = out_features * in_features
    n_groups = (numel + group_size - 1) // group_size
    padded = n_groups * group_size
    packed_count = (padded + 1) // 2
    generator = torch.Generator().manual_seed(out_features * 10000 + in_features)
    low = torch.randint(0, 16, (packed_count,), dtype=torch.uint8, generator=generator)
    high = torch.randint(0, 16, (packed_count,), dtype=torch.uint8, generator=generator)
    indices = low | (high << 4)
    codebook = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32)
    scales = torch.full((n_groups,), 0.02, dtype=torch.float32)
    fq = FigQuantTensor(
        indices=indices,
        codebook=codebook,
        scales=scales,
        shape=torch.Size((out_features, in_features)),
        n_groups=n_groups,
        group_size=group_size,
        numel=numel,
    )
    return FigLinear(
        in_features=in_features,
        out_features=out_features,
        fq=fq,
        lora_r=lora_r,
        lora_alpha=32,
        fast=False,
        mode="lowram",
    )


def run_case(name, shape, iterations, batch_size, sequence_length):
    out_features, in_features = shape
    gc.collect()
    before_layer = rss_mib()
    layer = make_layer(out_features, in_features)
    x = torch.randn(batch_size, sequence_length, in_features, requires_grad=True)
    after_setup = rss_mib()
    samples = []

    log(f"CASE {name}: shape={out_features}x{in_features}, iterations={iterations}")
    log(f"  RSS before layer={before_layer:.1f} MiB; after setup={after_setup:.1f} MiB")
    for iteration in range(1, iterations + 1):
        iteration_start = rss_mib()
        out = layer(x)
        after_forward = rss_mib()
        out.float().mean().backward()
        after_backward = rss_mib()
        layer.zero_grad(set_to_none=True)
        x.grad = None
        del out
        gc.collect()
        after_gc = rss_mib()
        sample = {
            "iteration": iteration,
            "rss_start_mib": iteration_start,
            "rss_after_forward_mib": after_forward,
            "rss_after_backward_mib": after_backward,
            "rss_after_gc_mib": after_gc,
        }
        samples.append(sample)
        log(
            f"  iter={iteration:02d} start={iteration_start:8.1f} "
            f"fwd={after_forward:8.1f} bwd={after_backward:8.1f} "
            f"gc={after_gc:8.1f} MiB"
        )

    before_trim = rss_mib()
    trim = malloc_trim()
    after_trim = rss_mib()
    log(f"  malloc_trim: {before_trim:.1f} -> {after_trim:.1f} MiB ({trim})")
    result = {
        "name": name,
        "shape": list(shape),
        "fp32_weight_mib": out_features * in_features * 4 / (1024 ** 2),
        "rss_before_layer_mib": before_layer,
        "rss_after_setup_mib": after_setup,
        "samples": samples,
        "rss_growth_after_gc_mib": samples[-1]["rss_after_gc_mib"] - samples[0]["rss_start_mib"],
        "rss_before_trim_mib": before_trim,
        "rss_after_trim_mib": after_trim,
        "rss_released_by_trim_mib": before_trim - after_trim,
        "malloc_trim": trim,
    }
    del layer, x
    gc.collect()
    malloc_trim()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--results-path", default=os.path.join(os.path.dirname(__file__), "lowram_allocator_results.json"))
    args = parser.parse_args()
    if min(args.iterations, args.batch_size, args.sequence_length) < 1:
        raise ValueError("iterations, batch size, and sequence length must be positive")

    started = time.time()
    baseline = rss_mib()
    results = {
        "scope": "isolated shipped FigLinear lowram forward/backward; no model, trainer, optimizer, dataset, or checkpointing",
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "baseline_rss_mib": baseline,
        "torch": torch.__version__,
        "cases": [],
    }
    log(f"Baseline RSS: {baseline:.1f} MiB")
    for name, shape in TINYLLAMA_SHAPES.items():
        results["cases"].append(
            run_case(name, shape, args.iterations, args.batch_size, args.sequence_length)
        )
    results["runtime_s"] = time.time() - started
    os.makedirs(os.path.dirname(os.path.abspath(args.results_path)), exist_ok=True)
    temporary = f"{args.results_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    os.replace(temporary, args.results_path)
    log(f"Saved -> {args.results_path}")


if __name__ == "__main__":
    main()
