#!/usr/bin/env python3
"""P3: measure Fig Engine Tier-1 training memory on the real code path."""

import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import psutil
import torch


MODEL_FULL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_SMOKE = "gpt2"


def log(message):
    print(f"[P3] {message}", flush=True)


def rss_gib():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def system_memory():
    memory = psutil.virtual_memory()
    return {
        "system_total_gib": memory.total / (1024 ** 3),
        "system_available_gib": memory.available / (1024 ** 3),
        "system_used_gib": memory.used / (1024 ** 3),
    }


def atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class PhaseTracker:
    """Continuously sample process RSS and system memory by named phase."""

    def __init__(self, durable_callback=None, interval=0.05):
        self.interval = interval
        self.durable_callback = durable_callback
        self.current_phase = "startup"
        self.phase_stats = {}
        self.n_samples = 0
        self.overall_peak_gib = 0.0
        self.baseline_rss_gib = None
        self.system_available_min_gib = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        process_gib = rss_gib()
        system = system_memory()
        stats = self.phase_stats.setdefault(
            self.current_phase,
            {
                "rss_entry_gib": process_gib,
                "rss_peak_gib": process_gib,
                "system_available_min_gib": system["system_available_gib"],
            },
        )
        stats["rss_peak_gib"] = max(stats["rss_peak_gib"], process_gib)
        stats["system_available_min_gib"] = min(
            stats["system_available_min_gib"], system["system_available_gib"]
        )
        self.overall_peak_gib = max(self.overall_peak_gib, process_gib)
        if self.baseline_rss_gib is None:
            self.baseline_rss_gib = process_gib
        self.system_available_min_gib = min(
            self.system_available_min_gib, system["system_available_gib"]
        )
        self.n_samples += 1

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self._thread.start()

    def phase(self, label):
        self._sample()
        if self.durable_callback:
            self.durable_callback(self.report(), self.current_phase)
        gc.collect()
        self.current_phase = label
        self._sample()
        log(f"-> {label} (entry RSS {rss_gib():.3f} GiB)")

    def stop(self):
        self._sample()
        self._stop.set()
        self._thread.join()

    def report(self):
        return {
            "baseline_rss_gib": self.baseline_rss_gib,
            "overall_peak_gib": self.overall_peak_gib,
            "overall_peak_delta_gib": self.overall_peak_gib - (self.baseline_rss_gib or 0.0),
            "phase_stats": dict(self.phase_stats),
            "n_samples": self.n_samples,
            "sampling_interval_s": self.interval,
            "system_available_min_gib": self.system_available_min_gib,
        }


def write_fixed_dataset(path, n_examples, sequence_length):
    # Long, deterministic text makes each example reach max_seq_length, so the
    # measured activation footprint is not understated by short examples.
    text = ("Memory-efficient training must be measured on the real path. " * 80)
    examples = [{"text": f"example {index}: {text}"} for index in range(n_examples)]
    atomic_write_json(path, examples)
    return {"examples": n_examples, "target_sequence_length": sequence_length}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--memory-mode", choices=("lowram", "figcache", "fast"), default="lowram")
    parser.add_argument("--memory-budget-gib", type=float, default=8.0)
    parser.add_argument("--claim-memory-gib", type=float, default=0.4)
    parser.add_argument("--results-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.sequence_length < 8:
        raise ValueError("steps and batch size must be positive; sequence length must be >= 8")

    model_id = MODEL_SMOKE if args.smoke else MODEL_FULL
    n_steps = 2 if args.smoke else args.steps
    results_path = args.results_path or os.path.join(
        os.path.dirname(__file__), "figengine_8gb_results.json"
    )
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(results_path)), "figengine_8gb_fixed_dataset.json"
    )
    run = {
        "model": model_id,
        "n_steps": n_steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "memory_mode": args.memory_mode,
        "budget_gib": args.memory_budget_gib,
        "paper_claim_gib": args.claim_memory_gib,
        "status": "starting",
        "scope": "CPU process RSS for FigModel.from_pretrained plus FigTrainer Tier-1 training",
    }

    def persist(report, phase, status="in_progress"):
        run.update(report)
        run["last_completed_phase"] = phase
        run["status"] = status
        atomic_write_json(results_path, run)

    tracker = PhaseTracker(durable_callback=persist)
    tracker.start()
    started = time.time()
    log("=" * 72)
    log(f"Fig Engine memory test | {model_id} | mode={args.memory_mode} | steps={n_steps}")
    log(f"8 GiB budget={args.memory_budget_gib:.3f} | paper estimate={args.claim_memory_gib:.3f} GiB")
    log("=" * 72)

    try:
        tracker.phase("import_engine")
        from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig
        from little_fig.engine.tier import TrainingTier

        tracker.phase("model_load_and_quantize")
        load_started = time.time()
        model = FigModel.from_pretrained(
            model_id,
            lora_r=16,
            lora_alpha=32,
            tier=TrainingTier.STREAMING_LORA,
            # Avoid creating the full FP32 cache when testing figcache/lowram.
            fast=args.memory_mode == "fast",
        )
        run["model_load_and_quantize_s"] = time.time() - load_started

        tracker.phase("trainer_and_dataset_init")
        config = FigTrainingConfig(
            tier=TrainingTier.STREAMING_LORA.value,
            num_epochs=1,
            learning_rate=2e-4,
            max_seq_length=args.sequence_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            use_packing=False,
            use_pipeline=False,
            activation_checkpointing=True,
            memory_mode=args.memory_mode,
            logging_steps=1,
            save_steps=0,
            output_dir=os.path.join(os.path.dirname(os.path.abspath(results_path)), "p3_checkpoints"),
        )
        trainer = FigTrainer(model, config)
        run["dataset"] = write_fixed_dataset(
            dataset_path, n_steps * args.batch_size, args.sequence_length
        )
        trainer.load_dataset(dataset_path)
        actual_batches = len(trainer.dataloader)
        if actual_batches != n_steps:
            raise RuntimeError(f"expected {n_steps} training batches, got {actual_batches}")

        tracker.phase("training_steps")
        train_started = time.time()
        trainer.train()
        run["train_time_s"] = time.time() - train_started

        tracker.phase("post_training_idle")
        time.sleep(0.5)
        tracker.stop()
        report = tracker.report()
        run.update(report)
        run["runtime_s"] = time.time() - started
        run["within_8gib_budget"] = report["overall_peak_gib"] <= args.memory_budget_gib
        run["within_paper_estimate"] = report["overall_peak_delta_gib"] <= args.claim_memory_gib
        run["budget_headroom_gib"] = args.memory_budget_gib - report["overall_peak_gib"]
        run["paper_estimate_headroom_gib"] = args.claim_memory_gib - report["overall_peak_delta_gib"]
        worst_phase, worst_stats = max(
            report["phase_stats"].items(), key=lambda item: item[1]["rss_peak_gib"]
        )
        run["worst_phase"] = worst_phase
        run["worst_phase_peak_gib"] = worst_stats["rss_peak_gib"]
        run["verdict_8gib"] = "REPRODUCED" if run["within_8gib_budget"] else "REFUTED"
        run["verdict_paper_estimate"] = (
            "REPRODUCED" if run["within_paper_estimate"] else "NOT REPRODUCED"
        )
        run["torch"] = torch.__version__
        run["status"] = "complete"
        atomic_write_json(results_path, run)
    except BaseException as error:
        if tracker._thread.is_alive():
            tracker.stop()
        run.update(tracker.report())
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        run["traceback"] = traceback.format_exc()
        atomic_write_json(results_path, run)
        raise

    log("PER-PHASE PEAK PROCESS RSS")
    for phase, stats in run["phase_stats"].items():
        log(f"  {phase:28s} {stats['rss_peak_gib']:7.3f} GiB")
    log(f"OVERALL: {run['overall_peak_gib']:.3f} GiB | worst={run['worst_phase']}")
    log(f"8 GiB verdict: {run['verdict_8gib']}")
    log(f"incremental RSS +{run['overall_peak_delta_gib']:.3f} GiB | "
        f"~{args.claim_memory_gib:.1f} GiB paper estimate: {run['verdict_paper_estimate']}")
    log(f"Saved -> {results_path}")


if __name__ == "__main__":
    main()
