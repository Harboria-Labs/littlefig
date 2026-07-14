# Little Fig

**The implementation hub for Harboria Labs' cognitive memory stack.**

Little Fig combines CPU-native model training, neural weight-space memory, and memory-specific evaluation tooling. The goal is not just to fine-tune language models on smaller machines. The goal is to build models that can accumulate, revise, and reason over memory without relying only on prompt stuffing or destructive base-weight edits.

[![Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License](https://img.shields.io/badge/license-mixed-blue)]()

## Research Stack

Little Fig is Layer 3 of a four-part research program, and this repository also carries the Memory Fabric implementation and CogMemBench benchmark.

| Layer | Component | Role | Paper |
|---|---|---|---|
| 1 | Ember's Diaries | Cognitive memory specification: immutable history, confidence, conflict preservation, reflection, provenance | [`paper/embers_diaries.md`](paper/embers_diaries.md) |
| 2 | Memory Fabric | Neural weight-space memory using isolated LoRA adapter namespaces and learned gates | [`paper/memory_fabric.md`](paper/memory_fabric.md) |
| 3 | Fig Engine | CPU-native training infrastructure for quantized models and continuous adapter updates | [`paper/fig_engine.md`](paper/fig_engine.md) |
| 4 | CogMemBench | Benchmark for acquisition, goal-directed recall, decay awareness, conflict detection, and consolidation | [`paper/cogmembench.md`](paper/cogmembench.md) |

The stack is built around one claim: useful AI memory needs more than retrieval. It needs lifecycle, uncertainty, contradiction handling, consolidation, and a way to learn continuously without damaging the base model.

## What This Repo Contains

- **Fig Engine**: adaptive INT4 quantization, CPU-aware training tiers, fused kernels, FigMeZO, LISA/LoRA/LOMO support, and GPU/CPU pipeline utilities.
- **Memory Fabric**: experimental adapter-based memory architecture that keeps the pretrained model frozen while writing memories into dedicated trainable adapter space.
- **Ember integration**: memory tokens and synthetic training examples based on the Ember's Diaries cognitive memory protocol.
- **CogMemBench**: a 1,000-case benchmark for testing cognitive memory behavior across five axes.
- **Studio UI**: a local web interface for running training, memory experiments, and benchmark flows.

## Current Highlights

| Result | Status |
|---|---|
| FigQuant beats fixed NF4 reconstruction error on GPT-2 and TinyLlama weight matrices | Validated |
| FigQuant LoRA benchmark reached 7x speedup over BnB NF4 QLoRA on a TinyLlama/T4 run | Validated |
| FigMeZO reduced loss by 18.6% vs standard MeZO in controlled tests | Validated |
| Sensitivity-guided LISA improved loss by 10% vs random LISA | Validated |
| Memory Fabric cross-session validation | In progress |

See the papers for methodology, limitations, and the difference between validated results and open research questions.

## Quick Start

Install CPU PyTorch first to avoid the default CUDA wheel download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[full]"
```

Fine-tune with Fig Engine:

```python
from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig

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
```

Train with Ember memory examples:

```python
from little_fig.engine import FigModel, FigTrainer, FigTrainingConfig

model = FigModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_r=16,
    ember_mode=True,
)

trainer = FigTrainer(model, FigTrainingConfig(num_epochs=3))
trainer.load_ember_dataset(n_examples=1000)
trainer.train()
```

Run CogMemBench:

```python
from cogmembench import CogMemRunner

results = CogMemRunner().run(model_fn=your_model_fn)
print(results["score"])
```

Launch the local Studio UI:

```bash
little-fig
```

## Repository Map

```text
src/little_fig/engine/     Fig Engine, Memory Fabric, and Ember integration
src/little_fig/web/        Local Studio UI and API
cogmembench/               Benchmark runner, scorer, generator, and dataset
paper/                     Canonical research papers
benchmark/                 Experiment scripts
tests/                     Unit and validation tests
```

## Research Papers

- [`paper/embers_diaries.md`](paper/embers_diaries.md): cognitive memory specification.
- [`paper/memory_fabric.md`](paper/memory_fabric.md): neural weight-space memory implementation.
- [`paper/fig_engine.md`](paper/fig_engine.md): CPU-native training infrastructure.
- [`paper/cogmembench.md`](paper/cogmembench.md): cognitive memory benchmark.

## License

Little Fig uses component-specific licenses:

| Component | License |
|---|---|
| Ember's Diaries integration and Memory Fabric core IP | AGPL-3.0-or-later + commercial option |
| Fig Engine training infrastructure | Apache-2.0 |
| CogMemBench benchmark code and dataset | MIT |
| Research papers | CC-BY-4.0 |

See [`NOTICE.md`](NOTICE.md) for the full license map and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for commercial licensing information.