#!/usr/bin/env python3
"""
Experiment: Per-Layer Codebook vs Shared (Global) Codebook

Observation: All 50 GPT-2 codebooks are within 0.019 L2 of each other.
The k-means refinement produces nearly identical results for every layer.
This is 400 wasted iterations (8 per layer × 50 layers).

Hypothesis: A single codebook computed from ALL weights together should
match per-layer quality at 50x less computation.

Test: Quantize GPT-2 three ways:
  A) Per-layer codebook (current FigQuant — 8 iters per layer)
  B) Global codebook (k-means on ALL weights concatenated)
  C) Fixed NF4 (no k-means at all — pure baseline)

Measure: MSE, cosine similarity, quantization time.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import numpy as np

def log(msg): print(f"[EXP] {msg}", flush=True)

def nf4_codebook():
    return torch.tensor([-1.0,-0.6962,-0.5251,-0.3949,-0.2844,-0.1848,-0.0911,0.0,
                          0.0796,0.1609,0.2461,0.3379,0.4407,0.5626,0.7230,1.0])

def run_kmeans(all_vals, n_iters=8):
    """Run k-means on values to produce a 16-entry codebook."""
    codebook = nf4_codebook().clone()
    for _ in range(n_iters):
        dists = (all_vals.unsqueeze(1) - codebook.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        for i in range(16):
            mask = assignments == i
            if mask.sum() > 0:
                codebook[i] = all_vals[mask].mean()
    # Keep zero representable
    codebook[codebook.abs().argmin()] = 0.0
    return codebook

def quantize_with_codebook(W, codebook, group_size=128):
    """Quantize weight matrix using a GIVEN codebook (no k-means)."""
    shape = W.shape
    numel = W.numel()
    flat = W.reshape(-1).float()
    pad = (group_size - numel % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    grouped = flat.reshape(-1, group_size)
    scales = grouped.abs().amax(dim=1).clamp(min=1e-10)
    scaled = grouped / scales.unsqueeze(1)
    # Assign to codebook
    dists = (scaled.reshape(-1).unsqueeze(1) - codebook.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1).reshape(-1, group_size)
    # Dequantize
    cb_exp = codebook.unsqueeze(0).expand(indices.shape[0], -1)
    result = torch.gather(cb_exp, 1, indices.long()) * scales.unsqueeze(1)
    return result.reshape(-1)[:numel].reshape(shape)

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    from little_fig.engine.figquant import figquant_quantize, figquant_dequantize

    log("="*60)
    log("  EXPERIMENT: Shared Codebook vs Per-Layer Codebook")
    log("="*60)

    log("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)

    # Collect all weight matrices
    weights = []
    names = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and param.numel() >= 512:
            weights.append(param.data.float())
            names.append(name)
    log(f"  {len(weights)} weight matrices")

    # ── Build GLOBAL codebook ──
    log("\nBuilding global codebook (k-means on ALL weights)...")
    t0 = time.time()
    # Sample from all weights (can't fit all in memory for pairwise distance)
    all_flat = torch.cat([w.reshape(-1) for w in weights])
    # Normalize to [-1,1] using global absmax
    global_scale = all_flat.abs().max()
    all_normalized = all_flat / global_scale
    # Subsample for k-means (1M points is enough)
    n_sample = min(1_000_000, all_normalized.numel())
    idx = torch.randperm(all_normalized.numel())[:n_sample]
    sample = all_normalized[idx]
    global_codebook = run_kmeans(sample, n_iters=16)  # more iters since it's global
    global_cb_time = time.time() - t0
    log(f"  Global codebook computed in {global_cb_time:.2f}s")
    log(f"  Codebook: {global_codebook.numpy().round(4).tolist()}")

    # ── Quantize each layer three ways and compare ──
    results = {"per_layer": [], "global": [], "nf4_fixed": []}
    times = {"per_layer": 0, "global": 0, "nf4_fixed": 0}

    nf4_cb = nf4_codebook()

    for i, (W, name) in enumerate(zip(weights, names)):
        # Method A: Per-layer FigQuant (current)
        t0 = time.time()
        q = figquant_quantize(W, group_size=128, n_iters=8)
        times["per_layer"] += time.time() - t0
        deq_pl = figquant_dequantize(q)
        mse_pl = F.mse_loss(deq_pl, W).item()
        cos_pl = F.cosine_similarity(W.flatten().unsqueeze(0), deq_pl.flatten().unsqueeze(0)).item()

        # Method B: Global codebook (no per-layer k-means)
        t0 = time.time()
        deq_gl = quantize_with_codebook(W, global_codebook, group_size=128)
        times["global"] += time.time() - t0
        mse_gl = F.mse_loss(deq_gl, W).item()
        cos_gl = F.cosine_similarity(W.flatten().unsqueeze(0), deq_gl.flatten().unsqueeze(0)).item()

        # Method C: Fixed NF4 (zero refinement)
        t0 = time.time()
        deq_nf = quantize_with_codebook(W, nf4_cb, group_size=128)
        times["nf4_fixed"] += time.time() - t0
        mse_nf = F.mse_loss(deq_nf, W).item()
        cos_nf = F.cosine_similarity(W.flatten().unsqueeze(0), deq_nf.flatten().unsqueeze(0)).item()

        results["per_layer"].append({"mse": mse_pl, "cos": cos_pl})
        results["global"].append({"mse": mse_gl, "cos": cos_gl})
        results["nf4_fixed"].append({"mse": mse_nf, "cos": cos_nf})

        if (i+1) % 10 == 0:
            log(f"  {i+1}/{len(weights)} layers done...")

    # ── Results ──
    log("\n" + "="*60)
    log("  RESULTS")
    log("="*60)

    for method in ["per_layer", "global", "nf4_fixed"]:
        avg_mse = np.mean([r["mse"] for r in results[method]])
        avg_cos = np.mean([r["cos"] for r in results[method]])
        label = {"per_layer": "Per-Layer (current)", "global": "Global Codebook", "nf4_fixed": "Fixed NF4"}[method]
        log(f"\n  {label}:")
        log(f"    Avg MSE:    {avg_mse:.6e}")
        log(f"    Avg Cosine: {avg_cos:.6f}")
        log(f"    Time:       {times[method]:.2f}s")

    # Compare
    avg_pl = np.mean([r["mse"] for r in results["per_layer"]])
    avg_gl = np.mean([r["mse"] for r in results["global"]])
    avg_nf = np.mean([r["mse"] for r in results["nf4_fixed"]])

    gl_vs_pl = (avg_gl - avg_pl) / avg_pl * 100
    gl_vs_nf = (avg_nf - avg_gl) / avg_nf * 100
    speedup = times["per_layer"] / max(times["global"], 0.01)

    log(f"\n  Global vs Per-Layer: {gl_vs_pl:+.1f}% MSE")
    log(f"  Global vs Fixed NF4: {gl_vs_nf:+.1f}% better")
    log(f"  Speed: {speedup:.1f}x faster than per-layer")

    # Per-layer comparison: how many layers does global win?
    gl_wins = sum(1 for a, b in zip(results["global"], results["per_layer"]) if a["mse"] <= b["mse"])
    log(f"  Global wins on {gl_wins}/{len(weights)} layers")

    if abs(gl_vs_pl) < 5.0:
        log(f"\n  ✅ Global codebook is within 5% of per-layer — VIABLE")
        log(f"     {speedup:.1f}x faster quantization for <{abs(gl_vs_pl):.1f}% quality cost")
    elif gl_vs_pl < 0:
        log(f"\n  ✅ Global codebook BEATS per-layer — ship it")
    else:
        log(f"\n  ❌ Global codebook is >{gl_vs_pl:.1f}% worse — not worth it")
