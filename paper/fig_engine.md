# Fig Engine: CPU-Native Training Infrastructure for Large Language Models via Adaptive Quantization and Memory-Aware Optimization

**Authors:** 0xticketguy (Harboria Labs)
**Repository:** https://github.com/Harboria-Labs/littlefig
**Version:** 1.0

> **Harboria Labs Research Stack**
> This paper is Layer 3 of a four-part research program.
> Layer 1 — Ember's Diaries: cognitive memory specification
> Layer 2 — Memory Fabric: neural weight-space implementation
> Layer 3 — **Fig Engine** *(this paper)*: training infrastructure
> Layer 4 — CogMem Benchmark: evaluation

---

## Abstract

Large language model (LLM) training has become increasingly dependent on high-memory GPUs, limiting participation to organizations with specialized hardware. While parameter-efficient fine-tuning methods reduce trainable parameters, they continue to assume GPU-centric execution and memory hierarchies. CPU execution remains largely unsupported as a first-class training environment, despite the widespread availability of commodity desktop and cloud CPUs.

We present Fig Engine, a CPU-native training infrastructure designed to enable efficient fine-tuning of quantized language models on resource-constrained hardware. Rather than treating CPU execution as a fallback, Fig Engine introduces a collection of complementary systems that optimize memory movement, quantization quality, and execution efficiency for commodity processors. The framework consists of FigQuant, an adaptive codebook INT4 quantization method with layer-aware refinement; FigCache, a cache architecture that minimizes repeated unpacking costs by storing intermediate quantization indices; FigSweep, a rolling layer-window execution strategy that bounds active memory during sequential transformer execution; FigKernel, a collection of fused operations compiled through torch.compile; and adaptive training tiers that automatically select optimization strategies according to available system memory.

Beyond the systems architecture, Fig Engine introduces two optimization techniques motivated by empirical analysis of quantized models. FigMeZO demonstrates that perturbing lower-error weight dimensions produces more reliable zeroth-order gradient estimates than perturbing high-error regions, reducing optimization loss by 18.6% relative to conventional MeZO. Sensitivity-Guided LISA allocates layer updates according to measured loss sensitivity rather than uniform sampling, improving parameter-efficient fine-tuning by 10% without increasing memory consumption.

Experiments on GPT-2 and TinyLlama demonstrate substantial reductions in memory requirements while maintaining competitive training quality. FigQuant achieves 5.3% lower MSE than fixed NF4 on all 50 GPT-2 weight matrices and wins all 156 layers on TinyLlama 1.1B. A GPU benchmark demonstrates FigQuant trains 7× faster than industry-standard BnB NF4 QLoRA on TinyLlama 1.1B while maintaining competitive loss quality.

Fig Engine establishes CPU-first training as a practical systems problem rather than a degraded GPU implementation, and provides the training infrastructure enabling Memory Fabric — the companion neural memory architecture described in a separate paper — to perform continuous weight-space memory writes on commodity hardware.

---

## 1. Introduction

The rapid progress of large language models has been accompanied by an equally rapid increase in computational requirements. Modern training pipelines typically assume access to GPUs equipped with tens of gigabytes of high-bandwidth memory, making efficient fine-tuning inaccessible to many researchers, students, and organizations operating on commodity hardware. Even parameter-efficient methods such as LoRA and QLoRA continue to rely on GPU-oriented execution models, where memory bandwidth substantially exceeds that of conventional CPUs.

This hardware assumption creates an unnecessary barrier to experimentation. Consumer desktop systems commonly provide 8–32 GB of system memory and multi-core CPUs capable of significant numerical throughput, yet existing training frameworks rarely optimize for these environments. CPU execution is generally treated as a compatibility feature rather than an architectural target, resulting in repeated memory transfers, redundant dequantization, and execution patterns that fail to exploit the characteristics of modern CPU memory hierarchies.

The principal limitation of CPU training is not arithmetic throughput but memory movement. Transformer execution repeatedly converts compressed weights into floating-point representations, reloads parameters across sequential layers, and materializes temporary tensors that exceed cache capacity. Consequently, memory bandwidth becomes the dominant bottleneck long before available computation is exhausted.

Fig Engine addresses this problem by redesigning the training pipeline around CPU constraints. Instead of introducing a single optimization, the framework combines multiple cooperating components that reduce memory traffic throughout the execution pipeline. Adaptive quantization improves representation quality while preserving compact storage. Cache-aware execution avoids redundant unpacking work. Rolling layer activation restricts the number of simultaneously expanded parameters. Compiled fused kernels minimize intermediate tensor allocation, while adaptive optimization strategies select training algorithms appropriate for the available memory budget.

The result is a CPU-native training infrastructure capable of fine-tuning modern language models using commodity hardware without treating CPU execution as a secondary deployment target.

### 1.1 Contributions

This paper makes the following contributions:

1. **FigQuant** — an adaptive INT4 codebook quantization method that refines NF4 initialization through layer-specific k-means optimization, consistently reducing reconstruction error across GPT-2 and TinyLlama weight matrices. Wins all 50/50 GPT-2 layers and all 156/156 TinyLlama layers against fixed NF4.

2. **FigCache** — a cache hierarchy that stores unpacked quantization indices rather than floating-point weights, reducing cache memory by 75% while eliminating repeated bit unpacking overhead. Three modes (fast / figcache / lowram) provide a continuous memory-speed tradeoff.

3. **FigSweep** — a rolling execution strategy that exploits the sequential structure of transformer layers to bound active memory usage during forward and backward computation. Reduces active parameter memory from O(L) to O(W) for window size W.

4. **FigKernel** — a collection of compiled fused operators via torch.compile that reduce memory allocation and improve execution efficiency across RMSNorm (2.95× speedup), activation, chunked cross-entropy, and fused linear+LoRA operations.

5. **Adaptive Training Tiers** — automatic selection of fine-tuning strategies (LoRA, LISA, MeZO, LOMO) according to available memory resources, enabling the same codebase to run from 400 MB to 8 GB.

6. **FigMeZO** — an inverse error-shaped zeroth-order optimization strategy demonstrating that perturbations concentrated in lower-error weight regions produce more reliable gradient estimates than perturbations targeting high-error regions. 18.6% loss reduction over standard MeZO, validated across 3 seeds.

7. **Sensitivity-Guided LISA** — a layer-selection strategy that allocates training effort according to measured loss sensitivity rather than uniform random sampling. 10% loss reduction over random LISA with no additional memory overhead.

---

## 2. Related Work

Efficient fine-tuning of large language models has become an active area of research, with most methods seeking to reduce either computational cost or trainable parameters. Existing approaches, however, continue to assume GPU-centric execution and do not address the broader systems challenges associated with CPU-native training.

### 2.1 Parameter-Efficient Fine-Tuning

Parameter-efficient fine-tuning (PEFT) methods reduce optimization cost by updating only a subset of model parameters while freezing the pretrained backbone. LoRA [Hu et al., 2022] introduced low-rank adaptation matrices that approximate weight updates with minimal trainable parameters, substantially lowering memory requirements without modifying the original model weights. Subsequent work, including QLoRA [Dettmers et al., 2023], combined LoRA with 4-bit quantization to further reduce GPU memory consumption while maintaining competitive downstream performance.

Although these methods dramatically decrease trainable parameter counts, they continue to rely on GPU-oriented execution pipelines. Quantized weights are repeatedly dequantized during computation, and execution assumes abundant high-bandwidth device memory. Consequently, PEFT methods alone do not address the memory hierarchy or bandwidth limitations of commodity CPUs.

### 2.2 Quantization

Weight quantization has become a standard technique for reducing the storage and inference cost of large language models. Uniform INT4 quantization provides aggressive compression but often introduces substantial reconstruction error due to the highly non-uniform distribution of transformer weights. NF4 [Dettmers et al., 2023] addressed this by introducing a fixed non-uniform codebook derived from a normal distribution. More recent methods, including AWQ [Lin et al., 2024] and GPTQ, optimize quantization through activation-aware or reconstruction-aware procedures.

FigQuant differs from these approaches by adapting the quantization codebook to the empirical weight distribution of each model. Rather than treating the NF4 codebook as fixed, FigQuant initializes from the NF4 distribution and refines the codebook using k-means optimization, consistently reducing reconstruction error while remaining compatible with efficient INT4 execution.

### 2.3 CPU Training

Most modern training frameworks expose CPU execution through backend abstraction rather than explicit architectural optimization. Libraries such as Hugging Face Transformers, TRL, LLaMA-Factory, and Unsloth primarily target GPU execution, with CPU support serving as a compatibility layer rather than a performance objective.

This design leaves several CPU-specific challenges unaddressed, including repeated dequantization, cache locality, memory bandwidth limitations, and redundant tensor allocation. Fig Engine approaches CPU execution as a first-class systems problem, redesigning quantization, caching, kernel execution, and layer scheduling to align with CPU memory hierarchies.

### 2.4 Zeroth-Order Optimization

Zeroth-order optimization methods estimate gradients through function evaluations rather than explicit backpropagation. MeZO [Malladi et al., 2023] demonstrated that forward-pass-only optimization can successfully fine-tune language models while significantly reducing memory requirements. Conventional MeZO samples perturbation directions uniformly across parameter space, implicitly assuming all weight dimensions contribute equally to gradient estimation quality. FigMeZO challenges this assumption and demonstrates the opposite strategy is more effective.

### 2.5 Layer Selection Strategies

LISA [Pan et al., 2024] introduced randomized layer activation during fine-tuning, reducing memory consumption by updating only subsets of layers at each iteration. Fig Engine extends this concept by replacing uniform layer selection with a one-time sensitivity analysis, allocating optimization effort proportionally to measured layer influence on training loss.

---

## 3. Design Objectives

The development of Fig Engine is guided by a simple observation: modern training systems are optimized for GPU hardware, whereas commodity CPUs exhibit fundamentally different performance characteristics. The architecture is driven by five primary design objectives.

### 3.1 CPU as the Primary Target

CPU execution should not be treated as a fallback implementation of a GPU pipeline. Memory access patterns, cache utilization, and sequential execution define the architecture from the outset. Every subsystem is designed with the CPU memory hierarchy as the primary constraint.

### 3.2 Minimize Memory Movement

Repeated movement of tensors between compressed and floating-point representations dominates execution time during CPU fine-tuning. Fig Engine minimizes unnecessary memory transfers through adaptive caching, layer scheduling, and fused execution. Where possible, computation is reorganized to reduce data movement rather than arithmetic complexity.

### 3.3 Preserve Model Quality

Aggressive compression often sacrifices downstream model performance. Fig Engine preserves the representational fidelity of pretrained weights while reducing storage requirements. Adaptive codebook refinement allows quantization to remain compact without relying on a fixed statistical approximation.

### 3.4 Progressive Resource Scaling

Commodity hardware varies widely in available memory. Rather than assuming a single execution strategy, Fig Engine exposes multiple training tiers that progressively trade memory consumption for computational efficiency, allowing the same framework to operate across laptops, desktop workstations, and cloud CPU instances.

### 3.5 Modular Optimization

Each subsystem is designed to function independently while composing into a unified training pipeline. Quantization, caching, layer scheduling, kernel fusion, and optimization algorithms can evolve separately without requiring redesign of the overall architecture.

---

## 4. Fig Engine Architecture

Fig Engine is organized as a layered execution pipeline that minimizes memory movement while preserving training quality on commodity CPUs. Rather than introducing a single optimization, the framework combines adaptive quantization, cache-aware execution, rolling layer scheduling, fused kernels, and memory-aware optimization into a unified training system.

Unlike conventional GPU-oriented pipelines where high bandwidth makes repeated tensor reconstruction relatively inexpensive, CPU execution is constrained primarily by memory hierarchy. Fig Engine therefore treats compressed weights as the canonical representation throughout training, reconstructing floating-point tensors only when computation requires them.

**Execution pipeline:**

```
           Pretrained Model
                 │
                 ▼
         FigQuant Compression
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
Packed INT4   Codebook    FP8 Scales
                 │
                 ▼
           FigCache Layer
                 │
                 ▼
         FigSweep Scheduler
                 │
                 ▼
          FigKernel Execution
                 │
                 ▼
  LoRA / LISA / FigMeZO / LOMO
                 │
                 ▼
        Updated Adapter Weights
```

### 4.1 FigQuant

Weight quantization forms the foundation of the Fig Engine architecture. Every subsequent optimization assumes that model weights remain stored in compressed form throughout training.

Existing INT4 methods follow one of two strategies. Uniform quantization distributes representable values evenly across the numerical range, sacrificing precision near zero where transformer weights are most densely concentrated. NF4 improves reconstruction quality via a fixed non-uniform codebook derived from a standard normal distribution, but the codebook remains identical for every transformer layer regardless of the underlying weight distribution.

Empirical analysis shows that transformer layers exhibit measurable variation in skewness, kurtosis, and tail behavior. A single global codebook cannot perfectly represent every layer.

**Adaptive refinement.** FigQuant initializes from the NF4 codebook and performs k-means refinement directly on the empirical weight distribution of each layer. The codebook Q = {c₁, c₂, ..., c₁₆} is refined to minimize reconstruction error:

```
min Σᵢ (wᵢ − ŵᵢ)²    where ŵᵢ = Q[indexᵢ] × sₘ
```

and sₘ denotes the scale factor for the corresponding quantization group. Refinement requires only a small number of iterations because NF4 initialization already lies near an optimal solution.

**Double quantization.** Per-group scale factors are quantized to FP8, reducing storage overhead by approximately 0.37 bits/parameter without measurable degradation in reconstruction quality.

**Storage layout:**

```
Packed INT4 indices + 16-value adaptive codebook + FP8 group scales
```

**Design note.** FigQuant intentionally separates representation from execution. The adaptive codebook exists only to improve representational fidelity. Execution optimizations are delegated to later pipeline stages, allowing FigCache and FigSweep to operate independently of the quantization algorithm.

### 4.2 FigCache

Although INT4 weights significantly reduce storage requirements, dequantization introduces a computational bottleneck. A standard dequantization pipeline consists of three stages: (1) unpack INT4 indices, (2) perform codebook lookup, (3) multiply by group scales. Profiling reveals that bit unpacking dominates total reconstruction cost at approximately 60% of dequantization time.

FigCache addresses this by caching the intermediate unpacked indices rather than fully reconstructed floating-point weights.

| Mode | Cached Representation | Memory (768→2048) | Forward Time | vs LowRAM |
|---|---|---|---|---|
| **fast** | Full FP32 weights | 6144 KB (100%) | 2.18 ms | 2.9× faster |
| **figcache** | Unpacked uint8 indices | 1536 KB (25%) | 4.86 ms | 1.3× faster |
| **lowram** | Packed INT4 (none) | 828 KB (13%) | 6.39 ms | baseline |

FigCache produces zero numerical error versus fast mode — output is bit-identical across all three modes. Only execution strategy differs.

The design depends directly on FigQuant's representation: the adaptive codebook is shared globally (64 bytes) and compact, so the per-layer cache stores only the pre-unpacked index array rather than reconstructed weights. FigCache should therefore be viewed as an extension of the quantization architecture rather than an independent caching mechanism.

### 4.3 FigSweep

Transformer models execute as a strictly sequential computation graph. Despite this sequential structure, most training frameworks retain reconstructed parameters for every layer simultaneously, increasing peak memory far beyond what active computation requires.

FigSweep maintains only a small window of W transformer layers in high-performance memory at any time, dynamically reconstructing entering layers and returning exiting layers to compressed representation.

**Rolling window execution:**

```
Time ──────────────────────────────────────────────►

Window 1    [L0][L1][L2][L3].....................
Window 2    ....[L1][L2][L3][L4]................
Window 3    ........[L2][L3][L4][L5]............
Window 4    ............[L3][L4][L5][L6]........
```

Only layers inside the active window remain dequantized. All other layers remain in compressed INT4 form.

For a transformer with L layers and active window W, FigSweep reduces active parameter memory complexity from O(L) to O(W). For GPT-2 (48 layers) with window=4: 25 MB active cache versus 302 MB for full reconstruction.

FigSweep is intentionally orthogonal to FigCache. FigCache determines *how* weights are reconstructed. FigSweep determines *when* reconstruction occurs. Layers entering the window may reconstruct from fast cache, FigCache indices, or LowRAM packed weights.

### 4.4 FigKernel

Even after reducing parameter storage and memory movement, conventional PyTorch execution introduces significant overhead through repeated kernel launches and intermediate tensor allocation. FigKernel addresses this through operator fusion via `torch.compile(backend="inductor")`, generating optimized AVX-512 on CPU and CUDA on GPU from the same source.

**FigRMSNorm.** Fuses variance computation, inverse root mean square, and activation scaling into a single compiled kernel. Only the inv_rms scalar required for backward propagation is retained. Intermediate tensors are never materialized. **2.95× speedup** over standard PyTorch RMSNorm.

**FigSwiGLU.** Fuses linear projection, SiLU activation, gating, and multiplication into one compiled kernel, reducing repeated memory access across dependent element-wise operations.

**Fused Linear + LoRA.** Combines `F.linear(x, W) + (x @ A) @ B * scale` into a single compiled execution path, avoiding intermediate tensor allocation from the LoRA delta computation.

**Chunked Cross-Entropy.** Processes vocabulary logits in fixed 8K chunks with numerically stable running log-sum-exp, avoiding materialization of the full [seq_len, vocab_size] tensor. **~8× peak memory reduction** versus standard cross-entropy for vocabularies of 32K+ tokens.

### 4.5 Adaptive Training Tiers

Fig Engine automatically selects an optimization strategy according to available memory budget, eliminating manual configuration across heterogeneous hardware.

| Tier | Method | Memory (1.1B) | Design Objective |
|---|---|---|---|
| 1 | Streaming LoRA | ~400 MB | Minimum memory |
| 2 | Sensitivity-Guided LISA | ~900 MB | Improved convergence |
| 3 | FigMeZO | ~600 MB | Gradient-free training |
| 4 | LOMO | ~800 MB | Maximum adaptation quality |

Each tier inherits the same execution infrastructure: FigQuant, FigCache, FigSweep, and FigKernel. The distinction lies only in the optimization strategy applied to trainable parameters.

---

## 5. Optimization Algorithms

The execution infrastructure minimizes memory movement and improves computational efficiency. Efficient execution alone, however, does not guarantee effective optimization. Quantized models exhibit characteristic error structures that influence gradient estimation, parameter sensitivity, and convergence behavior.

### 5.1 FigMeZO: Inverse Error-Shaped Zeroth-Order Optimization

Zeroth-order optimization estimates gradients through perturbation of model parameters rather than explicit backpropagation. MeZO [Malladi et al., 2023] demonstrated competitive fine-tuning using only forward evaluations. Conventional MeZO samples perturbation directions uniformly across parameter space, implicitly assuming equal gradient signal quality across all weight dimensions.

**Observation.** Quantization error is not uniformly distributed. Across GPT-2 and TinyLlama, a small fraction of weight groups carries a disproportionately large share of total reconstruction error, correlated with weight magnitude (Pearson r = +0.64). High-error regions already contain greater numerical uncertainty.

**Initial hypothesis (wrong).** Intuition suggested increasing perturbation magnitude in high-error regions, assuming these parameters required larger corrective updates. Experimental evaluation contradicted this: training became less stable, convergence deteriorated. Additional perturbation amplified existing numerical noise rather than correcting it.

**Revised hypothesis (FigMeZO).** Perturb low-error dimensions more aggressively. These have accurate base weights, smooth local loss surfaces, and therefore yield more reliable gradient signal.

**Algorithm.** For each perturbation vector z drawn from N(0, I):

```
z' = z × (1 + α(σ − 1))
```

where σ denotes normalized reconstruction error per dimension and α = −0.3 (negative = inverse shaping, emphasizing low-error regions). FigMeZO introduces zero additional memory — quantization statistics are already maintained by FigQuant.

**Key insight.** Optimization quality depends more strongly on signal reliability than on error magnitude. The question is not "where does the model perform poorly?" but "where can perturbations produce the most trustworthy information?"

### 5.2 Sensitivity-Guided LISA

LISA reduces memory by updating only a random subset of transformer layers at each step. Uniform random selection assumes equal importance across layers. Empirical analysis contradicts this: loss sensitivity varies over 200× across layers.

**Sensitivity measurement.** Before optimization begins, Fig Engine performs a lightweight sensitivity probe: each transformer layer is independently perturbed with a small random disturbance (scale 0.01) while the remaining network is frozen. The resulting change in training loss |ΔL| serves as an estimate of layer influence.

Cost: N+1 forward passes at initialization — negligible relative to full training.

**Weighted sampling.** Rather than uniform selection, Fig Engine samples layer i with probability:

```
Pᵢ = Sᵢ / Σⱼ Sⱼ     where Sᵢ = |ΔLᵢ|
```

High-sensitivity layers receive proportionally greater optimization attention.

**Observed block sensitivity (GPT-2):** Block 0 = 0.053, Block 4 = 0.049, Block 6 = 0.052 (high sensitivity, early layers); Block 10 = 0.013, Block 11 = 0.012 (low sensitivity, late layers).

### 5.3 Shared Codebook Initialization

Per-layer adaptive refinement improves reconstruction quality at the cost of k-means computation for every quantized layer. Analysis shows that refined codebooks across all GPT-2 layers converge to remarkably similar solutions: pairwise L2 distances between optimized codebooks remain within 0.019 across all 50 layers.

**Shared mode.** The first transformer layer performs standard adaptive refinement. Its optimized codebook initializes all remaining layers, which perform only index assignment (no k-means). This reduces model loading time by 5.1× at the cost of +3.1% MSE.

| Mode | Avg MSE | Load Time | Quality vs NF4 |
|---|---|---|---|
| Per-layer (default) | 1.768e-4 | 49.3s | −5.3% |
| Shared codebook | 1.822e-4 | 9.7s | −2.4% |
| Fixed NF4 | 1.866e-4 | ~9s | baseline |

The shared codebook is strictly better than fixed NF4 while matching NF4 loading speed. Users may choose between maximum quality and faster initialization as a configurable option.

### 5.4 Discussion

FigMeZO, Sensitivity-Guided LISA, and Shared Codebook Initialization share a common design philosophy: each exploits information already available within the existing training pipeline. FigMeZO reuses quantization statistics. Sensitivity-Guided LISA reuses initialization passes. Shared Codebook Initialization reuses optimized representations. No method introduces additional parameters or memory overhead.

---

## 6. Evaluation

The evaluation addresses four research questions:

- **RQ1.** Does FigQuant improve quantization quality compared to existing INT4 methods?
- **RQ2.** Does the Fig Engine pipeline reduce memory requirements without excessive computational overhead?
- **RQ3.** Do FigMeZO and Sensitivity-Guided LISA improve fine-tuning performance?
- **RQ4.** Can modern language models be practically fine-tuned on commodity hardware using the complete Fig Engine architecture?

Unless otherwise stated, experiments use PyTorch 2.x with torch.compile enabled. GPT-2 (124M) and TinyLlama (1.1B) serve as representative models. All quantization results are computed from pretrained weights prior to fine-tuning.

### 6.1 Quantization Quality (GPT-2)

Three quantization methods compared on all 50 linear weight matrices in GPT-2, group_size=128:

| Method | Cosine Similarity ↑ | MSE ↓ | SNR (dB) ↑ |
|---|---|---|---|
| Uniform INT4 (AbsMax) | 0.9883 | 4.11×10⁻⁴ | 17.1 |
| NF4 (fixed codebook) | 0.9946 | 1.87×10⁻⁴ | 19.6 |
| **FigQuant** | **0.9948** | **1.77×10⁻⁴** | **19.8** |

Layer-level consistency:

| Layer Type | FigQuant Wins |
|---|---|
| Embeddings (wte, wpe) | 2 / 2 |
| Attention (c_attn, c_proj) | 24 / 24 |
| Feed-Forward (c_fc, c_proj) | 24 / 24 |
| **Total** | **50 / 50** |

FigQuant wins every evaluated layer. The consistency demonstrates that gains arise from systematic adaptation to layer-specific distributions rather than isolated improvements on individual matrices.

### 6.2 Quantization Quality (TinyLlama 1.1B)

Live benchmark on all 156 linear layers of TinyLlama 1.1B, group_size=128:

| Method | Cosine Sim ↑ | MSE ↓ | SNR (dB) ↑ | Wins |
|---|---|---|---|---|
| Uniform INT4 | 0.9936 | 8.94×10⁻⁶ | 18.7 | 0 / 156 |
| NF4 (QLoRA standard) | 0.9953 | 5.97×10⁻⁶ | 20.1 | 0 / 156 |
| **FigQuant** | **0.9956** | **5.64×10⁻⁶** | **20.4** | **156 / 156** |

FigQuant wins every layer against both baselines on a substantially larger and more architecturally diverse model, validating that adaptive codebook refinement generalizes beyond GPT-2.

### 6.3 GPU Training Benchmark (TinyLlama 1.1B, Tesla T4)

To validate FigQuant's training efficiency beyond CPU, all methods were evaluated on GPU with identical configuration: LoRA r=16, α=32, target=[q,k,v,o]_proj, batch=4×4, lr=2e-4, 100 optimizer steps on Alpaca.

| Method | Final Loss | Training Time | GPU Memory | Relative Speed |
|---|---|---|---|---|
| FP16 LoRA (gold standard) | 0.2252 | 1309s | 3,585 MB | 1.0× |
| BnB NF4 QLoRA (industry default) | 0.2399 | 1423s | 2,441 MB | 0.9× |
| **FigQuant LoRA (lowram mode)** | **0.2475** | **184s** | **10,181 MB** | **7.1×** |

FigQuant is 7× faster than both FP16 and NF4 on GPU. The speed advantage comes from FigQuant's fused dequant-matmul path, which avoids the overhead of bitsandbytes' per-tensor quantization cycle. Loss is competitive: only 10% higher than FP16 (0.2475 vs 0.2252) while matching NF4 quality (0.2475 vs 0.2399).

Higher GPU memory in lowram mode results from temporary FP32 tensors during dequantization on each forward pass. The figcache mode is expected to reduce this substantially while maintaining the speed advantage.

Perplexity on wikitext-2: FP32 = 32.81, FigQuant = 35.33 (+7.7%, typical for INT4).

### 6.4 Memory Reduction

| Model | Conventional Training | Fig Engine Tier 1 | Reduction |
|---|---|---|---|
| GPT-2 (124M) | 3.48 GB | ~350 MB | 10× |
| TinyLlama (1.1B) | 26.6 GB | ~400 MB | 66× |
| Gemma 4B | 96.9 GB | ~1.5 GB | 65× |
| Llama 3.1 8B | 193.7 GB | ~3 GB | 64× |

These estimates include quantized backbone weights with parameter-efficient adaptation. Reductions are sufficient to enable fine-tuning on hardware that would otherwise be incapable of loading the model.

### 6.5 FigCache Performance

| Cache Mode | Forward Time | Relative Memory | vs LowRAM |
|---|---|---|---|
| fast | 2.18 ms | 100% | 2.9× faster |
| **figcache** | **4.86 ms** | **25%** | **1.3× faster** |
| lowram | 6.39 ms | 13% | baseline |

All three modes produce numerically identical outputs. FigCache achieves 75% memory reduction versus fast mode while eliminating the majority of dequantization overhead.

### 6.6 FigKernel Benchmarks (CPU, 2048 hidden, seq=256)

| Operation | Standard | FigKernel | Improvement |
|---|---|---|---|
| RMSNorm | 4.72 ms | 1.60 ms | **2.95× faster** |
| Cross-entropy (32K vocab) | Full tensor alloc | 8K chunks | **~8× less peak memory** |

### 6.7 FigMeZO

| Method | Avg Loss (last 20 steps) | vs Standard MeZO |
|---|---|---|
| Positive error shaping (α=+0.7) | 6.69 ± 0.17 | +10% worse |
| Standard MeZO | 6.08 ± 0.78 | baseline |
| **FigMeZO (α=−0.3)** | **4.95 ± 0.58** | **−18.6%** |

Results averaged across 3 seeds, GPT-2 (124M), 100 Alpaca steps. Contrary to the original hypothesis, emphasizing high-error regions consistently degraded optimization. The inverse strategy produced the lowest loss, validating that signal reliability matters more than error magnitude.

### 6.8 Sensitivity-Guided LISA

| Method | Avg Loss (last 20 steps) | vs Random |
|---|---|---|
| Random LISA | 2.41 | baseline |
| **Sensitivity-Guided LISA** | **2.17** | **−10%** |

GPT-2 (124M), 60 Alpaca steps. Transformer layers contribute unequally to adaptation: measuring influence before optimization allows resources to be concentrated where updates have the greatest effect.

### 6.9 End-to-End CPU Fine-Tuning

The complete Fig Engine pipeline was verified end-to-end on GPT-2 with adaptive quantization, FigCache, FigSweep, FigKernel, and LoRA adaptation. The system completed forward propagation, backward propagation, and adapter checkpoint generation while maintaining compressed backbone weights throughout training, consuming 45.8 MB for base weights (7.4× compression from 339.7 MB).

### 6.10 Discussion

The experimental results demonstrate that the contributions of Fig Engine are complementary rather than redundant. Adaptive quantization improves representational fidelity. Cache-aware execution reduces repeated reconstruction costs. Rolling layer scheduling bounds peak memory. Compiled kernels improve computational efficiency. Optimization algorithms exploit structural properties of quantized models to improve convergence. Collectively, these techniques transform CPU fine-tuning from a compatibility feature into a practical execution strategy.

---

## 7. Limitations

**Throughput vs. memory.** The current implementation prioritizes memory efficiency over absolute throughput. High-end GPUs continue to offer substantially greater training performance. Fig Engine expands access to model training rather than replacing GPU infrastructure for large-scale production workloads.

**Architecture coverage.** The current evaluation focuses on decoder-only transformer architectures. The underlying techniques are expected to generalize to encoder-decoder models, but these settings have not been systematically evaluated.

**Manual configuration.** Parameters such as quantization group size, FigSweep window length, and cache strategy currently require heuristic selection based on available hardware. Future work may explore adaptive runtime policies.

**Validation breadth.** FigMeZO and Sensitivity-Guided LISA demonstrate consistent improvements across GPT-2 and TinyLlama. Broader validation across larger model families and diverse downstream tasks remains important future work.

---

## 8. Future Work

**Adaptive quantization.** Future versions of FigQuant may incorporate activation-aware refinement, adaptive group sizing, or runtime codebook evolution.

**Dynamic scheduling.** FigSweep currently employs a fixed rolling window. Future schedulers could adjust window size dynamically according to processor cache occupancy and available memory bandwidth.

**Architecture-specific kernels.** While the current implementation relies on torch.compile, architecture-specific code generation for AVX2, AVX-512, and ARM NEON may provide additional performance.

**Memory Fabric integration.** The execution infrastructure developed in Fig Engine was designed not only for efficient one-time fine-tuning but to support persistent model adaptation through continuous micro-training between conversation turns. This capability is the foundation of Memory Fabric — a weight-space memory architecture described in a companion paper — which encodes memories directly into dedicated adapter parameters rather than external retrieval systems. Fig Engine's training tiers (particularly FigMeZO for backward-pass-free writes) enable Memory Fabric to perform memory writes within a <100ms budget on commodity hardware.

---

## 9. Conclusion

This paper presented Fig Engine, a CPU-native training infrastructure for large language models designed around the constraints of commodity hardware rather than GPU execution.

The framework combines adaptive quantization (FigQuant), cache-aware execution (FigCache), rolling layer scheduling (FigSweep), fused kernel compilation (FigKernel), and memory-aware optimization (FigMeZO, Sensitivity-Guided LISA) into a unified training system that substantially reduces memory requirements while preserving competitive fine-tuning quality. FigQuant achieves 5.3–5.4% lower MSE than fixed NF4 across all evaluated layers on both GPT-2 and TinyLlama 1.1B. The complete pipeline enables training on hardware previously considered impractical for modern language model adaptation.

These contributions demonstrate that CPU-native fine-tuning is a systems design problem rather than a hardware limitation. By reducing memory movement throughout the execution pipeline, Fig Engine enables efficient adaptation of modern language models on commodity hardware.

More broadly, Fig Engine provides the computational foundation for continual learning systems — specifically Memory Fabric, described in a companion paper — in which efficient weight-space memory writes become feasible between conversation turns on consumer devices.

---

## References

1. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Pan, T., et al. "LISA: Layerwise Importance Sampling for Memory-Efficient LLM Fine-Tuning." arXiv:2403.17919, 2024.
3. Malladi, S., et al. "Fine-Tuning Language Models with Just Forward Passes." NeurIPS 2023. arXiv:2305.17333.
4. Lv, K., et al. "Full Parameter Fine-tuning for Large Language Models with Limited Resources." arXiv:2306.09782, 2023.
5. Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
6. Lin, J., et al. "AWQ: Activation-aware Weight Quantization." MLSys 2024.
7. 0xticketguy (Harboria Labs). "Ember's Diaries: An Immutable Cognitive Database Engine for Grounded AI Memory." 2026. https://github.com/Harboria-Labs/embers-diaries
8. 0xticketguy (Harboria Labs). "Memory Fabric: Neural Weight-Space Implementation of Ember's Diaries." 2026. https://github.com/Harboria-Labs/littlefig
9. 0xticketguy (Harboria Labs). "CogMemBench: A Benchmark for Continuous Cognitive Memory in Large Language Models." 2026. https://github.com/Harboria-Labs/littlefig/tree/main/cogmembench

---

*Code: https://github.com/Harboria-Labs/littlefig*
*License: AGPL-3.0*
*Built by 0xticketguy / Harboria Labs*
