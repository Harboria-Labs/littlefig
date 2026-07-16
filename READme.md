# Little Fig — Harboria Labs AI Memory Stack

[![Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen)](https://github.com/Harboria-Labs/littlefig)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://github.com/Harboria-Labs/littlefig)
[![License](https://img.shields.io/badge/license-Apache%202.0%20%2F%20AGPL--3.0-blue)](#license)
[![arXiv](https://img.shields.io/badge/arXiv-paper-red)](https://github.com/Harboria-Labs/littlefig/blob/main/paper/fig_engine.md)

This repository is part of the **Harboria Labs AI Memory Stack** — a four-part research program building a complete memory system for AI from specification to evaluation.

| Layer | Project | What it does | Repo |
|---|---|---|---|
| 1 | **Ember's Diaries** | Cognitive memory specification | [Harboria-Labs/embers-diaries](https://github.com/Harboria-Labs/embers-diaries) |
| 2 | **Memory Fabric** | Neural weight-space memory implementation | This repo → `src/little_fig/engine/memory_fabric.py` |
| 3 | **Fig Engine** | CPU-native training infrastructure | This repo → `src/little_fig/engine/` |
| 4 | **CogMemBench** | Cognitive memory evaluation benchmark | This repo → `cogmembench/` |

**Fig Engine** makes it possible to fine-tune large language models on machines with no GPU — even with just 8 GB RAM. **Memory Fabric** uses Fig Engine's micro-training infrastructure to write memories directly into model weights between conversation turns. **CogMemBench** measures whether any of it actually improved memory.

---

## Research Papers

| Paper | What it covers | Link |
|---|---|---|
| Ember's Diaries | Cognitive memory specification (8 principles, immutable design) | [embers-diaries/paper](https://github.com/Harboria-Labs/embers-diaries/blob/main/paper/embers_diaries_paper.md) |
| Memory Fabric | Neural weight-space implementation of Ember's Diaries | [paper/memory_fabric.md](https://github.com/Harboria-Labs/littlefig/blob/main/paper/memory_fabric.md) |
| Fig Engine | CPU-native training infrastructure | [paper/fig_engine.md](https://github.com/Harboria-Labs/littlefig/blob/main/paper/fig_engine.md) |
| CogMemBench | Axis-decomposed cognitive memory benchmark | [cogmembench/paper](https://github.com/Harboria-Labs/littlefig/blob/main/cogmembench/paper/cogmembench_paper.md) |

---

## What's New (v0.6)

| Research Finding | Improvement | Validated |
|---|---|---|
| **FigMeZO** — inverse error-shaped zeroth-order optimization | −18.6% loss vs standard MeZO | ✓ 3 seeds |
| **Sensitivity-guided LISA** — weight selection by layer importance | −10% loss vs random LISA | ✓ controlled |
| **Shared codebook** — reuse one layer's codebook for all | 5× faster loading, 0.1% quality cost | ✓ 50 layers |
| **Memory Fabric gate fix** — decoupled gate/adapter learning rates | Gate opens in 3 steps (was stuck at 27%) | ✓ synthetic |

---

## Benchmark Results (TinyLlama 1.1B, Tesla T4)

### Quantization Quality (156 layers)

| Method | Cosine Sim | MSE | Wins |
|---|---|---|---|
| **FigQuant** | **0.9956** | **5.64e-6** | **156/156** |
| NF4 (QLoRA) | 0.9953 | 5.97e-6 | 0/156 |
| Absmax INT4 | 0.9936 | 8.94e-6 | 0/156 |

### GPU Training (100 steps, Alpaca, LoRA r=16)

| Method | Final Loss | Time | GPU Memory | Speed |
|---|---|---|---|---|
| FP16 LoRA | 0.2252 | 1309s | 3,585 MB | 1× |
| BnB NF4 QLoRA | 0.2399 | 1423s | 2,441 MB | 0.9× |
| **FigQuant LoRA** | **0.2475** | **184s** | 10,181 MB | **7×** |

### CogMemBench Baseline (TinyLlama 1.1B, no memory training)

| Axis | Score | What it means |
|---|---|---|
| Acquisition | 75% | Reads and repeats facts from context — basic comprehension |
| Recall (goal-directed) | 10% | Fails to retrieve by goal relevance vs topic similarity |
| Decay | 0% | Treats all facts as equally reliable regardless of age |
| Conflict detection | 0% | Cannot identify contradictions between stored facts |
| Consolidation | 10% | Doesn't weight repeated facts higher than single-mention |
| **CogMem Score** | **19 / 100** | **Floor — Memory Fabric target: 50–80** |

---

## What's Possible

| Task | Model | RAM Needed |
|---|---|---|
| Fine-tune (LoRA) | GPT-2 124M | ~350 MB |
| Fine-tune (LoRA) | TinyLlama 1.1B | ~400 MB |
| Fine-tune (LISA) | Gemma 4B | ~3.2 GB |
| Fine-tune (LoRA) | Llama 3.1 8B | ~3 GB |
| Memory Fabric write | Any model | +~30 MB (adapters) |
| CogMemBench eval | Any model | ~RAM for inference |

---

## Quick Start

### Install

```bash
# CPU PyTorch first (avoids 2.5GB CUDA download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Little Fig
pip install -e ".[full]"
```

### Fine-tune with Fig Engine

```python
from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig

# Load model — automatically quantizes with FigQuant + adds LoRA
model = FigModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_r=16,
    lora_alpha=32,
)

config = FigTrainingConfig(
    num_epochs=3,
    learning_rate=2e-4,
    max_seq_length=512,
)

trainer = FigTrainer(model, config)
trainer.load_dataset("tatsu-lab/alpaca")
trainer.train()

model.save_adapter("./my_adapter")
```

### Write a memory into weights (Memory Fabric)

```python
from little_fig.engine import FigModel
from little_fig.engine.micro_trainer import MicroTrainer, MicroTrainConfig

model = FigModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ember_mode=True,   # Activates Memory Fabric + injects memory tokens
)

config = MicroTrainConfig(
    learning_rate=5e-4,
    gate_lr_multiplier=20.0,   # Gate needs 20x higher lr than adapters
    steps=3,
)
trainer = MicroTrainer(model.memory_fabric, config)

# Write a personal fact into the personal/ namespace adapter
input_ids = model.tokenize("My API key expires September 1st")
result = trainer.write_memory(
    model=model,
    namespace="personal",
    input_ids=input_ids,
    labels=input_ids,
)
print(f"Loss: {result['loss_before']:.3f} → {result['loss_after']:.3f}")
print(f"Written in {result['time_ms']:.1f}ms")
```

### Run CogMemBench

```python
from cogmembench import CogMemRunner

runner = CogMemRunner(per_axis=200)   # Full 1,000 cases
results = runner.run(
    model_fn=lambda prompt: your_model.generate(prompt),
)

print(f"CogMem Score: {results['cogmem_score']}/100")
# Standard LLM: ~19/100
# With Memory Fabric (target): 50-80/100
```

### Train with Ember Memory tokens

```python
from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig

model = FigModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_r=16,
    ember_mode=True,   # Adds <|mem_store|>, <|mem_recall|>, etc.
)

config = FigTrainingConfig(num_epochs=3, max_seq_length=512)
trainer = FigTrainer(model, config)

# Generate Ember cognitive memory training data
trainer.load_ember_dataset(n_examples=1000)
trainer.train()
```

### Try in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harboria-Labs/littlefig/blob/main/Little_Fig_Colab.ipynb)

---

## Architecture

```
src/little_fig/
├── engine/
│   ├── figquant.py          # Adaptive codebook INT4 quantization
│   ├── figkernel.py         # Fused ops: RMSNorm (2.95×), SwiGLU, CE, Linear+LoRA
│   ├── figpipeline.py       # Async GPU-CPU training pipeline
│   ├── figmezo.py           # Inverse error-shaped zeroth-order optimizer
│   ├── figcache.py          # Three-tier cache: fast / figcache / lowram
│   ├── memory_fabric.py     # Memory Fabric: multi-namespace adapter memory  ← AGPL-3.0
│   ├── micro_trainer.py     # MicroTrainer: write memories between turns      ← AGPL-3.0
│   ├── ember_integration.py # Memory tokens + Ember training data
│   ├── linear.py            # FigLinear: FigQuant base + LoRA
│   ├── model.py             # FigModel: streaming loader + shared codebook
│   ├── trainer.py           # FigTrainer: unified training loop
│   ├── tier.py              # Auto tier selection by available RAM
│   ├── lisa.py              # Sensitivity-weighted LISA scheduler
│   ├── lomo.py              # LOMO optimizer
│   └── gguf_loader.py       # Universal GGUF loader

cogmembench/                 # CogMemBench evaluation benchmark   ← MIT
├── cogmembench.py           # Runner, Generator, Scorer
├── paper/                   # arXiv-ready paper
└── README.md

paper/                       # Research papers (CC BY 4.0)
├── fig_engine.md
├── memory_fabric.md
└── (ember's diaries → see Harboria-Labs/embers-diaries)
```

---

## Engine Stack

| Layer | Module | What it does |
|---|---|---|
| **Quantization** | FigQuant | Adaptive codebook INT4 — 5.4% less MSE than NF4, 7× faster training |
| **Cache** | FigCache | Three-tier: 75% less memory than fast, 1.3× faster than LowRAM |
| **Scheduling** | FigSweep | Rolling layer window — O(W) active memory instead of O(L) |
| **Compute** | FigKernel | torch.compile fused ops — 2.95× RMSNorm, 8× less CE memory |
| **Training** | 4 Tiers | LoRA → LISA → MeZO → LOMO, auto-selected by available RAM |
| **Optimizer** | FigMeZO | Inverse error-shaped — −18.6% loss vs standard MeZO |
| **Memory** | Memory Fabric | Multi-namespace adapters, gating, decay, conflict routing |
| **Cognition** | Ember Integration | 9 memory tokens trained into model vocabulary |
| **Benchmark** | CogMemBench | 5-axis cognitive memory evaluation, 1,000 cases |

---

## License

This repository contains components under different licenses. See [NOTICE.md](NOTICE.md) for full details.

| Component | License |
|---|---|
| Fig Engine (`src/little_fig/engine/` excluding memory_fabric) | Apache 2.0 |
| Memory Fabric (`src/little_fig/engine/memory_fabric.py`, `micro_trainer.py`) | AGPL-3.0 + Commercial |
| CogMemBench (`cogmembench/`) | MIT |
| All papers (`paper/`) | CC BY 4.0 |

Commercial use of Memory Fabric without AGPL-3.0 compliance requires a license from Harboria Labs.
See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

---

## Citation

```bibtex
@misc{figengine2026,
  title   = {Fig Engine: CPU-Native Training Infrastructure for Large Language Models},
  author  = {0xticketguy},
  year    = {2026},
  publisher = {Harboria Labs},
  url     = {https://github.com/Harboria-Labs/littlefig}
}
```

---

*Built by [0xticketguy](https://github.com/ticketguy) / [Harboria Labs](https://github.com/Harboria-Labs)*
