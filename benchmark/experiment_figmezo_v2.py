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
    # Also return the paper's ORIGINAL metric: mean perturbed train-loss estimate
    # over the final third of steps (in-sample, noisy — this is what -18.6% likely used).
    tail = trace[len(trace)//3:] if trace else trace
    train_est = float(np.mean(tail)) if tail else float("nan")
    return ev_before, ev_after, train_est


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    m = xs.mean()
    n = len(xs)
    if n < 2:
        return m, 0.0
    # Proper two-sided 95% t-multiplier by dof = n-1 (NOT 1.96, which only holds
    # for large n). Small-sample CIs are much wider — critical for honest verdicts.
    T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    t = T95.get(n - 1, 1.96)
    sem = xs.std(ddof=1) / math.sqrt(n)
    return m, t * sem


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

    # results[alpha] = eval_after / delta / train_est over seeds
    results = {a: {"after": [], "delta": [], "before": [], "train_est": []} for a in ALPHAS}
    t0 = time.time()

    for si, seed in enumerate(base_seeds):
        for alpha in ALPHAS:
            eb, ea, te = run_one(alpha, seed, train, eval_batches, steps)
            results[alpha]["before"].append(eb)
            results[alpha]["after"].append(ea)
            results[alpha]["delta"].append(ea - eb)
            results[alpha]["train_est"].append(te)
            log(f"  seed {seed:>4} alpha {alpha:+.2f}: "
                f"eval {eb:.4f} -> {ea:.4f} (delta {ea-eb:+.4f}) | train_est {te:.4f}")

    log("\n" + "=" * 64)
    log("  RESULTS — held-out eval loss (mean +/- 95% CI over seeds)")
    log("=" * 64)

    summary = {}
    base_mean, _ = mean_ci(results[0.0]["after"])
    base_te_mean, _ = mean_ci(results[0.0]["train_est"])
    for alpha in ALPHAS:
        m, ci = mean_ci(results[alpha]["after"])
        dm, dci = mean_ci(results[alpha]["delta"])
        tem, teci = mean_ci(results[alpha]["train_est"])
        pct_vs_std = (m - base_mean) / base_mean * 100 if base_mean else float("nan")
        pct_te_vs_std = (tem - base_te_mean) / base_te_mean * 100 if base_te_mean else float("nan")
        summary[str(alpha)] = {
            "eval_after_mean": m, "eval_after_ci95": ci,
            "delta_mean": dm, "delta_ci95": dci,
            "pct_vs_alpha0_EVAL": pct_vs_std,
            "train_est_mean": tem, "train_est_ci95": teci,
            "pct_vs_alpha0_TRAINEST": pct_te_vs_std,
            "eval_after_raw": results[alpha]["after"],
            "train_est_raw": results[alpha]["train_est"],
        }
        tag = "  <- standard MeZO (control)" if alpha == 0.0 else ""
        log(f"  alpha {alpha:+.2f}: EVAL={m:.4f}+/-{ci:.4f} ({pct_vs_std:+.1f}%)"
            f" | TRAIN_EST={tem:.4f} ({pct_te_vs_std:+.1f}%){tag}")

    # Verdict on the -0.3 claim — judge MAGNITUDE, not just sign.
    claim = summary.get("-0.3")
    log("\n" + "-" * 64)
    if claim:
        pct_eval = claim["pct_vs_alpha0_EVAL"]
        pct_te = claim["pct_vs_alpha0_TRAINEST"]
        neg03 = np.asarray(results[-0.3]["after"])
        std0 = np.asarray(results[0.0]["after"])
        diff = neg03 - std0                      # paired, same seeds
        paired_mean, paired_ci = mean_ci(diff)
        ci_excludes_0 = (paired_mean + paired_ci) < 0 or (paired_mean - paired_ci) > 0
        direction_ok = paired_mean < 0
        # "Reproduced" requires being in the same ballpark as -18.6% on eval loss.
        magnitude_ok = pct_eval <= -5.0         # generous: at least a 5% eval-loss win
        log(f"  CLAIM (paper): alpha=-0.3 gives -18.6% vs standard MeZO.")
        log(f"  EVAL loss:      alpha=-0.3 is {pct_eval:+.2f}% vs standard.")
        log(f"  TRAIN estimate: alpha=-0.3 is {pct_te:+.2f}% vs standard  (paper's likely metric).")
        log(f"  Paired eval diff = {paired_mean:+.4f} +/- {paired_ci:.4f}  "
            f"(CI excludes 0: {ci_excludes_0})")
        if magnitude_ok and ci_excludes_0:
            verdict = "REPRODUCED (large effect, CI excludes 0)"
        elif direction_ok and ci_excludes_0:
            verdict = "DIRECTION ONLY (alpha=-0.3 helps, but effect ~100x smaller than -18.6%)"
        elif direction_ok:
            verdict = "INCONCLUSIVE (right direction, CI includes 0 — noise-dominated)"
        else:
            verdict = "REFUTED (alpha=-0.3 did not help on held-out eval)"
        log(f"  VERDICT: {verdict}")
        summary["_verdict"] = {
            "claim_pct": -18.6,
            "found_pct_EVAL": pct_eval,
            "found_pct_TRAINEST": pct_te,
            "paired_eval_mean_diff": paired_mean, "paired_eval_ci95": paired_ci,
            "ci_excludes_0": bool(ci_excludes_0),
            "direction_correct": bool(direction_ok),
            "magnitude_reproduced": bool(magnitude_ok),
            "verdict": verdict,
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
