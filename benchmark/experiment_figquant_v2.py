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
    python benchmark/experiment_figquant_v2.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --resume --checkpoint-dir /content/drive/MyDrive/littlefig-p2
"""
import sys, os, gc, json, time, argparse, math, threading
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

def rss_gb():
    """Best-effort process RSS in GiB for Colab progress diagnostics."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except ImportError:
        return float("nan")


def system_memory_gib():
    """Return total, available, and used system memory in GiB."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        scale = 1024 ** 3
        return {
            'total_gib': memory.total / scale,
            'available_gib': memory.available / scale,
            'used_gib': (memory.total - memory.available) / scale,
            'percent': memory.percent,
        }
    except ImportError:
        return {
            'total_gib': float('nan'),
            'available_gib': float('nan'),
            'used_gib': float('nan'),
            'percent': float('nan'),
        }


# ── Baselines: byte-identical to tests/test_v05.py (same NF4 quantiles, same scaling) ──
class PeakRSSMonitor:
    '''Sample process/system memory and optionally persist a crash heartbeat.'''

    def __init__(self, interval_s=0.05, heartbeat_path=None, phase=None):
        self.interval_s = interval_s
        self.heartbeat_path = heartbeat_path
        self.phase = phase
        self.start_gb = rss_gb()
        self.peak_gb = self.start_gb
        self.end_gb = self.start_gb
        initial_system = system_memory_gib()
        self.system_total_gib = initial_system['total_gib']
        self.system_available_start_gib = initial_system['available_gib']
        self.system_available_min_gib = initial_system['available_gib']
        self.system_used_peak_gib = initial_system['used_gib']
        self._last_persisted = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _record_sample(self, persist=False):
        current_rss = rss_gb()
        system = system_memory_gib()
        self.peak_gb = max(self.peak_gb, current_rss)
        self.system_available_min_gib = min(
            self.system_available_min_gib, system['available_gib']
        )
        self.system_used_peak_gib = max(self.system_used_peak_gib, system['used_gib'])
        now = time.time()
        if self.heartbeat_path and (persist or now - self._last_persisted >= 2.0):
            _atomic_write_json(self.heartbeat_path, {
                'updated_unix_s': now,
                'phase': self.phase,
                'process_rss_current_gib': current_rss,
                'process_rss_peak_gib': self.peak_gb,
                'system_total_gib': system['total_gib'],
                'system_available_current_gib': system['available_gib'],
                'system_available_min_gib': self.system_available_min_gib,
                'system_used_current_gib': system['used_gib'],
                'system_used_peak_gib': self.system_used_peak_gib,
            })
            self._last_persisted = now

    def _sample(self):
        while not self._stop.wait(self.interval_s):
            self._record_sample()

    def __enter__(self):
        self._record_sample(persist=True)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_gb = rss_gb()
        self.peak_gb = max(self.peak_gb, self.end_gb)
        self._stop.set()
        self._thread.join()
        self._record_sample(persist=True)


def _checkpoint_path(model_id, checkpoint_dir):
    safe_model_id = ''.join(c if c.isalnum() or c in '._-' else '_' for c in model_id)
    return os.path.join(checkpoint_dir, f'figquant_v2_checkpoint_{safe_model_id}.json')


def _memory_status_path(model_id, checkpoint_dir):
    safe_model_id = ''.join(c if c.isalnum() or c in '._-' else '_' for c in model_id)
    return os.path.join(checkpoint_dir, f'figquant_v2_memory_{safe_model_id}.json')


def _atomic_write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def _new_checkpoint(model_id, total_layers, collection_stats):
    return {
        'version': 1,
        'model': model_id,
        'group_size': GROUP_SIZE,
        'n_iters': N_ITERS,
        'min_numel': MIN_NUMEL,
        'total_layers': total_layers,
        'collection_stats': collection_stats,
        'layers': {},
    }


def _load_checkpoint(path, model_id, total_layers):
    with open(path) as f:
        checkpoint = json.load(f)
    expected = {
        'model': model_id,
        'group_size': GROUP_SIZE,
        'n_iters': N_ITERS,
        'min_numel': MIN_NUMEL,
        'total_layers': total_layers,
    }
    mismatches = [
        f'{key}: checkpoint={checkpoint.get(key)!r}, current={value!r}'
        for key, value in expected.items()
        if checkpoint.get(key) != value
    ]
    if mismatches:
        raise ValueError('Checkpoint configuration mismatch: ' + '; '.join(mismatches))
    return checkpoint


def _quality_record(quality):
    return {
        'cos': float(quality['cosine_similarity']),
        'mse': float(quality['mse']),
        'snr': float(quality['snr_db']),
    }


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
    result = torch.empty_like(grouped)
    for start in range(0, n_groups, 2048):
        end = min(start + 2048, n_groups)
        chunk = scaled[start:end]
        dists = (chunk.unsqueeze(2) - codebook.reshape(1, 1, -1)).abs()
        indices = dists.argmin(dim=2)
        result[start:end] = codebook[indices] * scales[start:end].unsqueeze(1)
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
    result = torch.empty_like(grouped)
    for start in range(0, n_groups, 2048):
        end = min(start + 2048, n_groups)
        chunk = scaled[start:end]
        dists = (chunk.unsqueeze(2) - codebook.reshape(1, 1, -1)).abs()
        indices = dists.argmin(dim=2)
        result[start:end] = codebook[indices] * scales[start:end].unsqueeze(1)
    return result.reshape(-1)[:numel].reshape(tensor.shape)


def _measure_quality_raw(original: torch.Tensor, dequantized: torch.Tensor) -> dict:
    o = original.reshape(-1).float()
    d = dequantized.reshape(-1).float()
    mse = F.mse_loss(d, o).item()
    cos = F.cosine_similarity(o.unsqueeze(0), d.unsqueeze(0)).item()
    snr = 10 * np.log10(o.pow(2).mean().item() / max(mse, 1e-20))
    return {"cosine_similarity": cos, "mse": mse, "snr_db": snr}


def collect_weights(model_id, heartbeat_path=None):
    """Load model, extract all 2D weight matrices (numel>=MIN_NUMEL) as fp32 CPU
    tensors, free the HF model, and return them largest-first so that run_model's
    pop() processes them SMALLEST-first (the two huge embed/lm_head matrices last),
    which keeps peak RAM low."""
    from transformers import AutoModelForCausalLM
    collection_monitor = PeakRSSMonitor(
        heartbeat_path=heartbeat_path, phase=f'collect:{model_id}'
    )
    collection_monitor.__enter__()
    log(f"  Loading {model_id} (fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    total_model_params = sum(p.numel() for p in model.parameters())
    weights = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.numel() >= MIN_NUMEL:
            weights.append((name, p.data.detach().float().cpu().clone()))
    del model
    gc.collect()
    collection_monitor.__exit__(None, None, None)
    # Sort DESCENDING so weights.pop() (from the end) yields SMALLEST-first. We process
    # small->large, freeing each after use, so by the time we reach the two 65.5M
    # embed/lm_head matrices almost everything else is gone and peak RAM stays ~9 GB.
    weights.sort(key=lambda kv: kv[1].numel(), reverse=True)
    selected_params = sum(weight.numel() for _, weight in weights)
    largest_name, largest_weight = weights[0]
    total_model_fp32_gib = total_model_params * 4 / (1024 ** 3)
    selected_matrix_fp32_gib = selected_params * 4 / (1024 ** 3)
    collection_stats = {
        'total_model_params': total_model_params,
        'total_model_fp32_gib': total_model_fp32_gib,
        'selected_matrix_params': selected_params,
        'selected_matrix_fp32_gib': selected_matrix_fp32_gib,
        'largest_matrix': largest_name,
        'largest_matrix_params': largest_weight.numel(),
        'largest_matrix_fp32_mib': largest_weight.numel() * 4 / (1024 ** 2),
        'collection_rss_start_gb': collection_monitor.start_gb,
        'collection_rss_peak_gb': collection_monitor.peak_gb,
        'collection_rss_after_free_gb': collection_monitor.end_gb,
        'system_total_gib': collection_monitor.system_total_gib,
        'system_available_start_gib': collection_monitor.system_available_start_gib,
        'system_available_min_gib': collection_monitor.system_available_min_gib,
        'system_used_peak_gib': collection_monitor.system_used_peak_gib,
    }
    log(f"  {len(weights)} 2D matrices >= {MIN_NUMEL} params "
        f"(largest {weights[0][1].numel():,}: {weights[0][0]})")
    log(f'  SIZE: model {total_model_params:,} params = {total_model_fp32_gib:.2f} GiB FP32')
    log(f'  SIZE: selected matrices {selected_params:,} params = '
        f'{selected_matrix_fp32_gib:.2f} GiB FP32')
    log(f'  MEMORY: load/extract RSS start {collection_monitor.start_gb:.2f} GB | '
        f'peak {collection_monitor.peak_gb:.2f} GB | after model free {collection_monitor.end_gb:.2f} GB')
    return weights, collection_stats


def run_model(model_id, weights):
    """Quantize every matrix with FigQuant / NF4 / absmax; return honest stats."""
    tot = {m: {"cos": 0.0, "mse": 0.0, "snr": 0.0} for m in ("figquant", "nf4", "absmax")}
    n = 0
    fq_wins = 0
    reductions = []          # per-layer % MSE reduction of FigQuant vs NF4 (+ = better)
    losers = []              # layers where FigQuant did NOT beat NF4
    total = len(weights)
    while weights:
        name, W = weights.pop()          # smallest remaining first (list is desc-sorted), then freed
        layer_numel = W.numel()
        layer_started = time.time()
        log(f"  [{n + 1}/{total}] START {name} ({layer_numel:,} params) | RSS {rss_gb():.2f} GB")
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
        del W, q_fq, qual_fq, qual_nf4, qual_abs
        gc.collect()
        log(f"  [{n}/{total}] DONE  {name} | FigQuant vs NF4 {red:+.2f}% | "
            f"{time.time() - layer_started:.1f}s | RSS {rss_gb():.2f} GB")

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


def _fallback_collection_stats(weights):
    selected_params = sum(weight.numel() for _, weight in weights)
    largest_name, largest_weight = max(weights, key=lambda item: item[1].numel())
    current_rss = rss_gb()
    system = system_memory_gib()
    return {
        'total_model_params': selected_params,
        'total_model_fp32_gib': selected_params * 4 / (1024 ** 3),
        'selected_matrix_params': selected_params,
        'selected_matrix_fp32_gib': selected_params * 4 / (1024 ** 3),
        'largest_matrix': largest_name,
        'largest_matrix_params': largest_weight.numel(),
        'largest_matrix_fp32_mib': largest_weight.numel() * 4 / (1024 ** 2),
        'collection_rss_start_gb': current_rss,
        'collection_rss_peak_gb': current_rss,
        'collection_rss_after_free_gb': current_rss,
        'system_total_gib': system['total_gib'],
        'system_available_start_gib': system['available_gib'],
        'system_available_min_gib': system['available_gib'],
        'system_used_peak_gib': system['used_gib'],
    }


def run_model_resumable(
    model_id,
    weights,
    collection_stats=None,
    checkpoint_dir='.',
    resume=False,
    stop_after=None,
    memory_budget_gib=8.0,
):
    '''Quantize all matrices with durable per-layer metrics and peak-RSS evidence.'''
    weights.sort(key=lambda item: item[1].numel(), reverse=True)
    total = len(weights)
    collection_stats = collection_stats or _fallback_collection_stats(weights)
    checkpoint_path = _checkpoint_path(model_id, checkpoint_dir)
    heartbeat_path = _memory_status_path(model_id, checkpoint_dir)

    if resume and os.path.exists(checkpoint_path):
        checkpoint = _load_checkpoint(checkpoint_path, model_id, total)
        checkpoint.setdefault('collection_runs', []).append(collection_stats)
        completed = len(checkpoint['layers'])
        log(f'  RESUME: {completed}/{total} completed layers from {checkpoint_path}')
    else:
        checkpoint = _new_checkpoint(model_id, total, collection_stats)
        checkpoint['collection_runs'] = [collection_stats]
        if resume:
            log(f'  RESUME: no checkpoint found; starting fresh at {checkpoint_path}')
        else:
            log(f'  CHECKPOINT: saving every completed layer to {checkpoint_path}')
    _atomic_write_json(checkpoint_path, checkpoint)

    while weights:
        name, weight = weights.pop()
        if name in checkpoint['layers']:
            completed = len(checkpoint['layers'])
            log(f'  [{completed}/{total}] SKIP  {name} (checkpointed)')
            del weight
            gc.collect()
            continue

        completed_before = len(checkpoint['layers'])
        layer_numel = weight.numel()
        layer_fp32_mib = layer_numel * 4 / (1024 ** 2)
        layer_started = time.time()
        log(f'  [{completed_before + 1}/{total}] START {name} ({layer_numel:,} params, '
            f'{layer_fp32_mib:.2f} MiB FP32) | RSS {rss_gb():.2f} GB')

        with PeakRSSMonitor(
            heartbeat_path=heartbeat_path, phase=f'quantize:{name}'
        ) as layer_memory:
            quantized = figquant_quantize(weight, group_size=GROUP_SIZE, n_iters=N_ITERS)
            quality_figquant = measure_quality(weight, quantized)
            quality_nf4 = _measure_quality_raw(
                weight, _nf4_quantize_dequantize(weight, GROUP_SIZE)
            )
            quality_absmax = _measure_quality_raw(
                weight, _absmax_int4_quantize_dequantize(weight, GROUP_SIZE)
            )

        nf4_mse = quality_nf4['mse']
        reduction = ((nf4_mse - quality_figquant['mse']) / nf4_mse * 100
                     if nf4_mse else 0.0)
        runtime_s = time.time() - layer_started
        checkpoint['layers'][name] = {
            'numel': layer_numel,
            'fp32_mib': layer_fp32_mib,
            'figquant': _quality_record(quality_figquant),
            'nf4': _quality_record(quality_nf4),
            'absmax': _quality_record(quality_absmax),
            'reduction_pct': float(reduction),
            'figquant_wins': bool(quality_figquant['mse'] < nf4_mse),
            'runtime_s': round(runtime_s, 3),
            'rss_start_gb': layer_memory.start_gb,
            'rss_peak_gb': layer_memory.peak_gb,
            'rss_end_gb': layer_memory.end_gb,
            'rss_peak_delta_gb': layer_memory.peak_gb - layer_memory.start_gb,
            'system_total_gib': layer_memory.system_total_gib,
            'system_available_start_gib': layer_memory.system_available_start_gib,
            'system_available_min_gib': layer_memory.system_available_min_gib,
            'system_used_peak_gib': layer_memory.system_used_peak_gib,
        }
        _atomic_write_json(checkpoint_path, checkpoint)
        completed = len(checkpoint['layers'])

        del weight, quantized, quality_figquant, quality_nf4, quality_absmax
        gc.collect()
        log(f'  [{completed}/{total}] DONE  {name} | FigQuant vs NF4 {reduction:+.2f}% | '
            f'{runtime_s:.1f}s | RSS peak {layer_memory.peak_gb:.2f} GB | '
            f'end {rss_gb():.2f} GB | delta {layer_memory.peak_gb - layer_memory.start_gb:+.2f} GB')
        if stop_after is not None and completed >= stop_after:
            log(f'  INTENTIONAL STOP after {completed}/{total}; rerun with --resume')
            raise SystemExit(75)

    records = list(checkpoint['layers'].items())
    if len(records) != total:
        raise RuntimeError(f'Incomplete checkpoint: {len(records)}/{total} layers')

    totals = {
        method: {'cos': 0.0, 'mse': 0.0, 'snr': 0.0}
        for method in ('figquant', 'nf4', 'absmax')
    }
    reductions = []
    losers = []
    for name, record in records:
        for method in totals:
            for metric in totals[method]:
                totals[method][metric] += record[method][metric]
        reductions.append(record['reduction_pct'])
        if not record['figquant_wins']:
            losers.append({'layer': name, 'reduction_pct': record['reduction_pct']})

    layer_count = len(records)
    wins = layer_count - len(losers)
    averages = {
        method: {metric: value / layer_count for metric, value in metrics.items()}
        for method, metrics in totals.items()
    }
    mse_reduction_nf4 = (
        (averages['nf4']['mse'] - averages['figquant']['mse'])
        / averages['nf4']['mse'] * 100
    )
    mse_reduction_absmax = (
        (averages['absmax']['mse'] - averages['figquant']['mse'])
        / averages['absmax']['mse'] * 100
    )
    reductions.sort()
    collection_runs = checkpoint.get('collection_runs', [collection_stats])
    collection_peak = max(run['collection_rss_peak_gb'] for run in collection_runs)
    quantization_peak = max(record['rss_peak_gb'] for _, record in records)
    max_workspace_delta = max(record['rss_peak_delta_gb'] for _, record in records)
    run_peak = max(collection_peak, quantization_peak)
    peak_layer_name, peak_layer_record = max(
        records, key=lambda item: item[1]['rss_peak_gb']
    )
    system_available_min = min(
        [run.get('system_available_min_gib', float('inf')) for run in collection_runs]
        + [record.get('system_available_min_gib', float('inf')) for _, record in records]
    )
    within_memory_budget = run_peak <= memory_budget_gib

    return {
        'model': model_id,
        'n_layers': layer_count,
        'fq_wins_vs_nf4': wins,
        'win_fraction': f'{wins}/{layer_count}',
        'mse_reduction_vs_nf4_pct': mse_reduction_nf4,
        'mse_reduction_vs_absmax_pct': mse_reduction_absmax,
        'snr_gain_vs_nf4_db': averages['figquant']['snr'] - averages['nf4']['snr'],
        'per_layer_reduction_min_pct': reductions[0],
        'per_layer_reduction_median_pct': reductions[len(reductions) // 2],
        'per_layer_reduction_max_pct': reductions[-1],
        'n_losers': len(losers),
        'losers': losers[:10],
        'averages': averages,
        'memory_proof': {
            'total_model_params': collection_stats['total_model_params'],
            'total_model_fp32_gib': collection_stats['total_model_fp32_gib'],
            'selected_matrix_params': collection_stats['selected_matrix_params'],
            'selected_matrix_fp32_gib': collection_stats['selected_matrix_fp32_gib'],
            'largest_matrix': collection_stats['largest_matrix'],
            'largest_matrix_params': collection_stats['largest_matrix_params'],
            'largest_matrix_fp32_mib': collection_stats['largest_matrix_fp32_mib'],
            'collection_peak_rss_gb': collection_peak,
            'quantization_peak_rss_gb': quantization_peak,
            'run_peak_rss_gb': run_peak,
            'run_peak_rss_gib': run_peak,
            'peak_layer': peak_layer_name,
            'peak_layer_rss_gib': peak_layer_record['rss_peak_gb'],
            'max_layer_workspace_delta_gb': max_workspace_delta,
            'system_total_gib': collection_stats.get('system_total_gib'),
            'system_available_min_gib': system_available_min,
            'memory_budget_gib': memory_budget_gib,
            'within_memory_budget': within_memory_budget,
            'budget_headroom_gib': memory_budget_gib - run_peak,
            'budget_utilization_pct': run_peak / memory_budget_gib * 100,
            'rss_sampling_interval_s': 0.05,
            'memory_heartbeat': heartbeat_path,
        },
        'checkpoint': checkpoint_path,
    }


def synthetic_weights():
    """Offline smoke: random matrices shaped like a mini-transformer (no downloads).
    Includes one biggish matrix to exercise the batched final-assignment path."""
    g = torch.Generator().manual_seed(0)
    shapes = [(128, 128), (256, 256), (512, 128), (128, 512), (1024, 512)]
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


def main_resumable():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--smoke', action='store_true', help='synthetic matrices, no downloads')
    ap.add_argument('--resume', action='store_true', help='skip layers already saved in checkpoint')
    ap.add_argument('--checkpoint-dir', default='.', help='directory for per-model checkpoints')
    ap.add_argument('--results-path', default=None, help='durable output JSON path')
    ap.add_argument('--memory-budget-gib', type=float, default=8.0,
                    help='target peak process RSS budget (default: 8 GiB)')
    ap.add_argument('--stop-after', type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    log('=' * 68)
    log('  FigQuant vs NF4 / uniform-INT4 | resumable, deterministic, peak-RSS sampled')
    log('=' * 68)

    started = time.time()
    results = {}

    if args.smoke:
        log('SMOKE: synthetic random matrices (no model download)')
        weights = synthetic_weights()
        result = run_model_resumable(
            'synthetic', weights, checkpoint_dir=args.checkpoint_dir, resume=args.resume,
            stop_after=args.stop_after, memory_budget_gib=args.memory_budget_gib,
        )
        results['synthetic'] = result
        win_fraction = result['win_fraction']
        reduction_nf4 = result['mse_reduction_vs_nf4_pct']
        reduction_absmax = result['mse_reduction_vs_absmax_pct']
        log(f'  synthetic: wins {win_fraction}, MSE vs NF4 {reduction_nf4:+.2f}%, '
            f'vs absmax {reduction_absmax:+.2f}%')
    else:
        models = args.models or DEFAULT_MODELS
        for model_id in models:
            log('-' * 68)
            log(f'MODEL: {model_id}')
            heartbeat_path = _memory_status_path(model_id, args.checkpoint_dir)
            weights, collection_stats = collect_weights(model_id, heartbeat_path)
            result = run_model_resumable(
                model_id,
                weights,
                collection_stats=collection_stats,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                stop_after=args.stop_after,
                memory_budget_gib=args.memory_budget_gib,
            )
            result['_verdict'] = verdict_for(model_id, result)
            results[model_id] = result

            memory = result['memory_proof']
            win_fraction = result['win_fraction']
            reduction_nf4 = result['mse_reduction_vs_nf4_pct']
            reduction_absmax = result['mse_reduction_vs_absmax_pct']
            reduction_min = result['per_layer_reduction_min_pct']
            reduction_median = result['per_layer_reduction_median_pct']
            reduction_max = result['per_layer_reduction_max_pct']
            snr_gain = result['snr_gain_vs_nf4_db']
            loser_count = result['n_losers']
            selected_fp32 = memory['selected_matrix_fp32_gib']
            collection_peak = memory['collection_peak_rss_gb']
            quantization_peak = memory['quantization_peak_rss_gb']
            run_peak = memory['run_peak_rss_gb']
            budget = memory['memory_budget_gib']
            budget_headroom = memory['budget_headroom_gib']
            budget_status = 'PASS' if memory['within_memory_budget'] else 'FAIL'
            checkpoint_path = result['checkpoint']
            verdict = result['_verdict']
            log(f'  RESULT: FigQuant wins {win_fraction} vs NF4')
            log(f'          mean MSE vs NF4: {reduction_nf4:+.2f}% '
                f'| vs absmax {reduction_absmax:+.2f}%')
            log(f'          per-layer reduction min {reduction_min:+.2f}% '
                f'| median {reduction_median:+.2f}% | max {reduction_max:+.2f}%')
            log(f'          SNR gain vs NF4: {snr_gain:+.3f} dB '
                f'| layers FigQuant loses: {loser_count}')
            log(f'          MEMORY: selected FP32 {selected_fp32:.2f} GiB '
                f'| load/extract peak {collection_peak:.2f} GB '
                f'| quant peak {quantization_peak:.2f} GB | run peak {run_peak:.2f} GB')
            log(f'          MEMORY BUDGET: {budget_status} | target {budget:.2f} GiB '
                f'| headroom {budget_headroom:+.2f} GiB '
                f'| system available floor {memory["system_available_min_gib"]:.2f} GiB')
            log(f'          MEMORY PEAK LAYER: {memory["peak_layer"]} '
                f'({memory["peak_layer_rss_gib"]:.2f} GiB RSS)')
            log(f'          CHECKPOINT: {checkpoint_path}')
            log(f'          VERDICT: {verdict}')
            gc.collect()

    results['_config'] = {
        'group_size': GROUP_SIZE,
        'n_iters': N_ITERS,
        'min_numel': MIN_NUMEL,
        'metric': 'per-layer reconstruction MSE (dequantized vs fp32)',
        'deterministic': True,
        'seeds': 'n/a (single deterministic run)',
        'rss_sampling_interval_s': 0.05,
        'resume_enabled': args.resume,
        'memory_budget_gib': args.memory_budget_gib,
        'torch': torch.__version__,
        'runtime_s': round(time.time() - started, 1),
    }
    output_path = args.results_path or os.path.join(
        os.path.dirname(__file__), 'figquant_v2_results.json'
    )
    _atomic_write_json(output_path, results)
    runtime_s = results['_config']['runtime_s']
    log('=' * 68)
    log(f'Saved -> {output_path} ({runtime_s}s)')


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
    main_resumable()
