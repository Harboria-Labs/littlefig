#!/usr/bin/env python3
"""
Experiment P2 — FigQuant vs fixed NF4 / uniform INT4, rigorous multi-model re-test.

CLAIM UNDER TEST (paper/fig_engine.md §6.1-6.2, README, Little_Fig_Colab.ipynb):
    FigQuant beats fixed NF4 on per-layer reconstruction MSE:
      - GPT-2:      -5.3% MSE, wins 50 / 50   weight matrices  (committed: tests/v05_results.json)
      - TinyLlama:  -5.5% MSE, wins 156 / 156  linear layers    (NO committed script anywhere)
    Paper TinyLlama table: NF4 MSE 5.97e-6 -> FigQuant 5.64e-6 (= -5.5%), 156/156 wins.

WHY THIS SCRIPT EXISTS:
    tests/test_v05.py::test_figquant_vs_baselines only ever loads "gpt2" (hardcoded).
    The 156/156 TinyLlama headline has never had a committed, runnable script — the
    same gap we found for FigMeZO in P1. This script:
      1. Reproduces GPT-2 as a HARNESS SELF-CHECK against the committed JSON
         (if we match 50/50 and -5.28% there, the harness is trustworthy), and
      2. Runs the MISSING TinyLlama benchmark, reported honestly.

DESIGN (rigor):
    - Uses the SHIPPED little_fig.engine.figquant.figquant_quantize + measure_quality
      (group_size=128, n_iters=8) — the REAL algorithm, not a reimplementation.
    - NF4 and uniform-INT4 baselines are byte-identical to tests/test_v05.py, and use
      the SAME per-group absmax scaling as FigQuant (fairest possible comparison; the
      NF4 codebook is the exact set of values FigQuant refines from).
    - DETERMINISTIC: k-means starts from the fixed NF4 codebook on fixed pretrained
      weights, so ONE run suffices — no seeds. (Contrast P1, which needed 5 seeds
      because MeZO perturbations are random.)
    - MEMORY-SAFE: extract weights, free the HF model, quantize LARGEST-LAST, freeing
      each after use, so peak RAM stays ~9 GB even for TinyLlama's two 65.5M-param
      matrices (embed_tokens + lm_head). Fits a free Colab CPU runtime.
    - HONEST REPORTING: per-model win count, the MEAN and DISTRIBUTION (min / median /
      max) of per-layer MSE reduction, and an explicit list of any layers FigQuant
      LOSES on. FigQuant = "NF4 init + per-layer k-means"; k-means provably reduces
      *normalized* distortion, but the metric is *reconstruction* MSE (reweighted by
      per-group scale^2), so all-wins is a genuine empirical result, not a tautology.

USAGE:
    python benchmark/experiment_figquant_v2.py                 # gpt2 + TinyLlama
    python benchmark/experiment_figquant_v2.py --models gpt2   # just gpt2 (fast self-check)
    python benchmark/experiment_figquant_v2.py --smoke         # synthetic matrices, NO downloads
"""
import sys, os, gc, json, time, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn.functional as F
import numpy as np

from little_fig.engine.figquant import figquant_quantize, measure_quality  # SHIPPED code under test

GROUP_SIZE = 128
N_ITERS = 8
MIN_NUMEL = 1024
DEFAULT_MODELS = ["gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]

# Committed GPT-2 reference (tests/v05_results.json) — used as a harness self-check.
GPT2_REF = {"n_layers": 50, "fq_wins": 50, "mse_reduction_vs_nf4_pct": 5.2809199447595745}


def log(msg):
    print(f"[P2] {msg}", flush=True)


# ── Baselines: byte-identical to tests/test_v05.py (same NF4 quantiles, same scaling) ──
def _nf4_quantize_dequantize(tensor: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Real NF4 (QLoRA-style): fixed N(0,1)-quantile codebook, no refinement."""
    numel = tensor.numel()
    flat = tensor.reshape(-1).float()
    pad = (group_size - numel % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    grouped = flat.reshape(-1, group_size)
    n_groups = grouped.shape[0]
    scales = grouped.abs().amax(dim=1).clamp(min=1e-10)
    scaled = grouped / scales.unsqueeze(1)
    codebook = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
    ], dtype=torch.float32)
    dists = (scaled.reshape(-1).unsqueeze(1) - codebook.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1).reshape(n_groups, group_size)
    cb = codebook.unsqueeze(0).expand(n_groups, -1)
    result = torch.gather(cb, dim=1, index=indices.long()) * scales.unsqueeze(1)
    return result.reshape(-1)[:numel].reshape(tensor.shape)


def _absmax_int4_quantize_dequantize(tensor: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Real uniform absmax INT4: 16 evenly-spaced levels in [-1, 1]."""
    numel = tensor.numel()
    flat = tensor.reshape(-1).float()
    pad = (group_size - numel % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    grouped = flat.reshape(-1, group_size)
    n_groups = grouped.shape[0]
    scales = grouped.abs().amax(dim=1).clamp(min=1e-10)
    scaled = grouped / scales.unsqueeze(1)
    codebook = torch.linspace(-1.0, 1.0, 16)
    dists = (scaled.reshape(-1).unsqueeze(1) - codebook.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1).reshape(n_groups, group_size)
    cb = codebook.unsqueeze(0).expand(n_groups, -1)
    result = torch.gather(cb, dim=1, index=indices.long()) * scales.unsqueeze(1)
    return result.reshape(-1)[:numel].reshape(tensor.shape)


def _measure_quality_raw(original: torch.Tensor, dequantized: torch.Tensor) -> dict:
    o = original.reshape(-1).float()
    d = dequantized.reshape(-1).float()
    mse = F.mse_loss(d, o).item()
    cos = F.cosine_similarity(o.unsqueeze(0), d.unsqueeze(0)).item()
    snr = 10 * np.log10(o.pow(2).mean().item() / max(mse, 1e-20))
    return {"cosine_similarity": cos, "mse": mse, "snr_db": snr}


def collect_weights(model_id):
    """Load model, extract all 2D weight matrices (numel>=MIN_NUMEL) as fp32 CPU
    tensors, free the HF model, and return them largest-first so that run_model's
    pop() processes them SMALLEST-first (the two huge embed/lm_head matrices last),
    which keeps peak RAM low."""
    from transformers import AutoModelForCausalLM
    log(f"  Loading {model_id} (fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    weights = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.numel() >= MIN_NUMEL:
            weights.append((name, p.data.detach().float().cpu().clone()))
    del model
    gc.collect()
    # Sort DESCENDING so weights.pop() (from the end) yields SMALLEST-first. We process
    # small->large, freeing each after use, so by the time we reach the two 65.5M
    # embed/lm_head matrices almost everything else is gone and peak RAM stays ~9 GB.
    weights.sort(key=lambda kv: kv[1].numel(), reverse=True)
    log(f"  {len(weights)} 2D matrices >= {MIN_NUMEL} params "
        f"(largest {weights[0][1].numel():,}: {weights[0][0]})")
    return weights


def run_model(model_id, weights):
    """Quantize every matrix with FigQuant / NF4 / absmax; return honest stats."""
    tot = {m: {"cos": 0.0, "mse": 0.0, "snr": 0.0} for m in ("figquant", "nf4", "absmax")}
    n = 0
    fq_wins = 0
    reductions = []          # per-layer % MSE reduction of FigQuant vs NF4 (+ = better)
    losers = []              # layers where FigQuant did NOT beat NF4
    while weights:
        name, W = weights.pop()          # smallest remaining first (list is desc-sorted), then freed
        q_fq = figquant_quantize(W, group_size=GROUP_SIZE, n_iters=N_ITERS)
        qual_fq = measure_quality(W, q_fq)
        qual_nf4 = _measure_quality_raw(W, _nf4_quantize_dequantize(W, GROUP_SIZE))
        qual_abs = _measure_quality_raw(W, _absmax_int4_quantize_dequantize(W, GROUP_SIZE))

        for k, q in (("figquant", qual_fq), ("nf4", qual_nf4), ("absmax", qual_abs)):
            tot[k]["cos"] += q["cosine_similarity"]
            tot[k]["mse"] += q["mse"]
            tot[k]["snr"] += q["snr_db"]

        red = (qual_nf4["mse"] - qual_fq["mse"]) / qual_nf4["mse"] * 100 if qual_nf4["mse"] else 0.0
        reductions.append(red)
        if qual_fq["mse"] < qual_nf4["mse"]:
            fq_wins += 1
        else:
            losers.append({"layer": name, "reduction_pct": red})
        n += 1
        del W, q_fq
        gc.collect()

    avgs = {m: {k: v / n for k, v in tot[m].items()} for m in tot}
    mse_red = (avgs["nf4"]["mse"] - avgs["figquant"]["mse"]) / avgs["nf4"]["mse"] * 100
    mse_red_abs = (avgs["absmax"]["mse"] - avgs["figquant"]["mse"]) / avgs["absmax"]["mse"] * 100
    reductions = sorted(reductions)
    return {
        "model": model_id,
        "n_layers": n,
        "fq_wins_vs_nf4": fq_wins,
        "win_fraction": f"{fq_wins}/{n}",
        "mse_reduction_vs_nf4_pct": mse_red,
        "mse_reduction_vs_absmax_pct": mse_red_abs,
        "snr_gain_vs_nf4_db": avgs["figquant"]["snr"] - avgs["nf4"]["snr"],
        "per_layer_reduction_min_pct": reductions[0],
        "per_layer_reduction_median_pct": reductions[len(reductions) // 2],
        "per_layer_reduction_max_pct": reductions[-1],
        "n_losers": len(losers),
        "losers": losers[:10],
        "averages": avgs,
    }


def synthetic_weights():
    """Offline smoke: random matrices shaped like a mini-transformer (no downloads).
    Includes one biggish matrix to exercise the batched final-assignment path."""
    g = torch.Generator().manual_seed(0)
    shapes = [(768, 768), (768, 3072), (3072, 768), (2048, 512), (8000, 2048)]
    return [(f"synthetic.{i}.{r}x{c}", torch.randn(r, c, generator=g))
            for i, (r, c) in enumerate(shapes)]


def verdict_for(model_id, r):
    """Per-model verdict vs the paper claim (-5.3%..-5.5%, all-wins)."""
    all_win = r["fq_wins_vs_nf4"] == r["n_layers"]
    red = r["mse_reduction_vs_nf4_pct"]
    magnitude_ok = red >= 3.0                      # claim ~5.3-5.5%; allow slack
    if all_win and magnitude_ok:
        v = "REPRODUCED"
    elif all_win:
        v = f"PARTIAL (wins all {r['win_fraction']} but MSE reduction {red:.2f}% < claimed ~5.3%)"
    elif r["fq_wins_vs_nf4"] >= 0.9 * r["n_layers"]:
        v = f"MOSTLY ({r['win_fraction']}; not the claimed all-layers sweep)"
    else:
        v = f"REFUTED ({r['win_fraction']} wins, {red:+.2f}% MSE)"
    # GPT-2 also self-checks against the committed number.
    if model_id == "gpt2":
        match = (r["n_layers"] == GPT2_REF["n_layers"]
                 and r["fq_wins_vs_nf4"] == GPT2_REF["fq_wins"]
                 and abs(red - GPT2_REF["mse_reduction_vs_nf4_pct"]) < 0.3)
        v += "  [harness self-check vs committed v05_results.json: " + ("MATCH" if match else "MISMATCH") + "]"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true", help="synthetic matrices, no downloads")
    args = ap.parse_args()

    log("=" * 68)
    log("  FigQuant vs NF4 / uniform-INT4  |  group_size=128, n_iters=8, deterministic")
    log("=" * 68)

    t0 = time.time()
    results = {}

    if args.smoke:
        log("SMOKE: synthetic random matrices (no model download)")
        r = run_model("synthetic", synthetic_weights())
        results["synthetic"] = r
        log(f"  synthetic: wins {r['win_fraction']}, MSE vs NF4 {r['mse_reduction_vs_nf4_pct']:+.2f}%, "
            f"vs absmax {r['mse_reduction_vs_absmax_pct']:+.2f}%")
    else:
        models = args.models or DEFAULT_MODELS
        for mid in models:
            log("-" * 68)
            log(f"MODEL: {mid}")
            weights = collect_weights(mid)
            r = run_model(mid, weights)
            r["_verdict"] = verdict_for(mid, r)
            results[mid] = r
            log(f"  RESULT: FigQuant wins {r['win_fraction']} vs NF4")
            log(f"          mean MSE vs NF4: {r['mse_reduction_vs_nf4_pct']:+.2f}%  "
                f"(claim ~ -5.3%)  | vs absmax {r['mse_reduction_vs_absmax_pct']:+.2f}%")
            log(f"          per-layer reduction  min {r['per_layer_reduction_min_pct']:+.2f}%  "
                f"median {r['per_layer_reduction_median_pct']:+.2f}%  "
                f"max {r['per_layer_reduction_max_pct']:+.2f}%")
            log(f"          SNR gain vs NF4: {r['snr_gain_vs_nf4_db']:+.3f} dB  |  "
                f"layers FigQuant loses: {r['n_losers']}")
            log(f"          VERDICT: {r['_verdict']}")
            gc.collect()

    results["_config"] = {
        "group_size": GROUP_SIZE, "n_iters": N_ITERS, "min_numel": MIN_NUMEL,
        "metric": "per-layer reconstruction MSE (dequantized vs fp32)",
        "deterministic": True, "seeds": "n/a (single deterministic run)",
        "torch": torch.__version__, "runtime_s": round(time.time() - t0, 1),
    }
    out = os.path.join(os.path.dirname(__file__), "figquant_v2_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log("=" * 68)
    log(f"Saved -> {out}  ({results['_config']['runtime_s']}s)")


if __name__ == "__main__":
    main()
