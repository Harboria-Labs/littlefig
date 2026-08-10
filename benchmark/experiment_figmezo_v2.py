#!/usr/bin/env python3
"""
Experiment P1 — FigMeZO inverse error-shaping, rigorous re-test.

Goal: verify or refute the paper's headline claim:
    FigMeZO (shaping_strength = -0.3)  →  -18.6% loss vs standard MeZO, 3 seeds.

Why this script exists:
    The committed benchmark/experiment_figmezo.py only ever runs shaping
    strengths +0.7 and +1.0 — it NEVER tests the -0.3 setting the claim rests
    on. It also reports the perturbed train-loss estimate (L+ + L-)/2, which is
    a noisy proxy, and uses a single seed.

What this fixes:
    1. Sweeps alpha in {-0.3, 0.0, +0.7} on ONE code path (FigMeZO). At alpha=0
       the shaped noise reduces exactly to isotropic N(0,I) == standard MeZO, so
       alpha=0 is a clean paired control on the identical quantized model.
    2. Reports held-out EVAL loss (unperturbed params, no_grad) on a fixed set,
       not the perturbation estimate.
    3. Runs multiple seeds and reports mean +/- 95% CI, plus % change vs alpha=0.
    4. Paired design: for a given seed, all alphas share the same model init,
       same data order, same base perturbation directions — only shaping differs.

Usage:
    python benchmark/experiment_figmezo_v2.py                # full: 100 steps, 5 seeds
    python benchmark/experiment_figmezo_v2.py --smoke        # tiny: 8 steps, 2 seeds
    python benchmark/experiment_figmezo_v2.py --steps 60 --seeds 3
"""
import sys, os, time, gc, json, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import numpy as np
from little_fig.engine import FigModel
from little_fig.engine.tier import TrainingTier
from little_fig.engine.figmezo import FigMeZO, FigMeZOConfig

MODEL = "gpt2"
LR = 1e-5
EPSILON = 1e-3
ALPHAS = [-0.3, 0.0, 0.7]     # -0.3 = paper's claim, 0.0 = standard MeZO control, +0.7 = paper says worse
N_TRAIN = 200
N_EVAL = 32
MAX_LEN = 64


def log(msg):
    print(f"[P1] {msg}", flush=True)


def prepare_data(n_train, n_eval, max_len):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    log(f"Loading Alpaca ({n_train} train + {n_eval} eval)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(n_train + n_eval))
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token

    def encode(row):
        text = f"### Instruction:\n{row.get('instruction','')}\n\n### Response:\n{row.get('output','')}"
        enc = tok(text, truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"], "labels": enc["input_ids"].clone()}

    rows = [encode(r) for r in ds]
    return rows[:n_train], rows[n_train:n_train + n_eval]


def model_device(model):
    return next(model.model.parameters()).device


@torch.no_grad()
def eval_loss(model, eval_batches):
    """Unperturbed held-out loss — the honest metric."""
    model.model.eval()
    dev = model_device(model)
    total = 0.0
    for b in eval_batches:
        total += model(input_ids=b["input_ids"].to(dev),
                       labels=b["labels"].to(dev)).loss.item()
    return total / len(eval_batches)


def run_one(alpha, seed, train, eval_batches, steps):
    """One training run. Returns (eval_before, eval_after, train_trace)."""
    # Fresh model per run so seeds/alphas never share drifted state.
    model = FigModel.from_pretrained(
        MODEL, lora_r=16, lora_alpha=32, tier=TrainingTier.STREAMING_LORA
    )
    opt = FigMeZO(model.model, FigMeZOConfig(
        learning_rate=LR, epsilon=EPSILON, seed=seed, shaping_strength=alpha,
    ))

    ev_before = eval_loss(model, eval_batches)
    model.model.eval()
    dev = model_device(model)
    trace = []
    for step in range(steps):
        batch = train[step % len(train)]
        ids = batch["input_ids"].to(dev)
        lab = batch["labels"].to(dev)
        def forward_fn():
            return model(input_ids=ids, labels=lab).loss
        trace.append(opt.step(forward_fn))
    ev_after = eval_loss(model, eval_batches)

    del model, opt
    gc.collect()
    return ev_before, ev_after, trace


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    m = xs.mean()
    if len(xs) < 2:
        return m, 0.0
    # 95% CI via t-ish 1.96 (small n, report sem*1.96 as an honest approximation)
    sem = xs.std(ddof=1) / math.sqrt(len(xs))
    return m, 1.96 * sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    if args.smoke:
        steps, n_seeds, n_train, n_eval = 8, 2, 32, 8
    else:
        steps, n_seeds, n_train, n_eval = 100, 5, N_TRAIN, N_EVAL
    if args.steps:
        steps = args.steps
    if args.seeds:
        n_seeds = args.seeds

    log("=" * 64)
    log(f"  FigMeZO alpha sweep | steps={steps} seeds={n_seeds} alphas={ALPHAS}")
    log("=" * 64)

    train, eval_batches = prepare_data(n_train, n_eval, MAX_LEN)
    base_seeds = [42 + 100 * i for i in range(n_seeds)]

    # results[alpha] = list of eval_after over seeds; also keep deltas (after-before)
    results = {a: {"after": [], "delta": [], "before": []} for a in ALPHAS}
    t0 = time.time()

    for si, seed in enumerate(base_seeds):
        for alpha in ALPHAS:
            eb, ea, trace = run_one(alpha, seed, train, eval_batches, steps)
            results[alpha]["before"].append(eb)
            results[alpha]["after"].append(ea)
            results[alpha]["delta"].append(ea - eb)
            log(f"  seed {seed:>4} alpha {alpha:+.2f}: "
                f"eval {eb:.4f} -> {ea:.4f} (delta {ea-eb:+.4f})")

    log("\n" + "=" * 64)
    log("  RESULTS — held-out eval loss (mean +/- 95% CI over seeds)")
    log("=" * 64)

    summary = {}
    base_mean, _ = mean_ci(results[0.0]["after"])
    for alpha in ALPHAS:
        m, ci = mean_ci(results[alpha]["after"])
        dm, dci = mean_ci(results[alpha]["delta"])
        pct_vs_std = (m - base_mean) / base_mean * 100 if base_mean else float("nan")
        summary[str(alpha)] = {
            "eval_after_mean": m, "eval_after_ci95": ci,
            "delta_mean": dm, "delta_ci95": dci,
            "pct_vs_alpha0": pct_vs_std,
            "eval_after_raw": results[alpha]["after"],
        }
        tag = "  <- standard MeZO (control)" if alpha == 0.0 else ""
        log(f"  alpha {alpha:+.2f}: eval_after = {m:.4f} +/- {ci:.4f}"
            f"   ({pct_vs_std:+.1f}% vs standard){tag}")

    # Verdict on the -0.3 claim
    claim = summary.get("-0.3")
    log("\n" + "-" * 64)
    if claim:
        pct = claim["pct_vs_alpha0"]
        neg03 = np.asarray(results[-0.3]["after"])
        std0 = np.asarray(results[0.0]["after"])
        # paired difference test (same seeds)
        diff = neg03 - std0
        paired_mean, paired_ci = mean_ci(diff)
        better = paired_mean < 0 and (paired_mean + paired_ci) < 0  # CI excludes 0
        log(f"  CLAIM: alpha=-0.3 gives -18.6% vs standard MeZO.")
        log(f"  FOUND: alpha=-0.3 is {pct:+.1f}% vs standard (eval loss).")
        log(f"  Paired (per-seed) mean diff = {paired_mean:+.4f} +/- {paired_ci:.4f}")
        log(f"  VERDICT: {'SUPPORTED (CI excludes 0, direction correct)' if better else 'NOT SUPPORTED at this scale'}")
        summary["_verdict"] = {
            "claim_pct": -18.6, "found_pct": pct,
            "paired_mean_diff": paired_mean, "paired_ci95": paired_ci,
            "supported": bool(better),
        }

    summary["_config"] = {
        "steps": steps, "seeds": n_seeds, "alphas": ALPHAS,
        "lr": LR, "epsilon": EPSILON, "model": MODEL,
        "n_train": n_train, "n_eval": n_eval, "max_len": MAX_LEN,
        "runtime_s": round(time.time() - t0, 1),
        "metric": "held-out eval loss (unperturbed)",
    }

    out = os.path.join(os.path.dirname(__file__), "figmezo_v2_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  Saved -> {out}  ({summary['_config']['runtime_s']}s)")


if __name__ == "__main__":
    main()
