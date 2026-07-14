# Memory Fabric: Neural Weight-Space Implementation of Ember's Diaries

**Authors:** 0xticketguy (Harboria Labs)
**Repository:** https://github.com/Harboria-Labs/littlefig
**Version:** 0.6 (preliminary — Stage 3 cross-session validation in progress)

> **Harboria Labs Research Stack**
> This paper is Layer 2 of a four-part research program.
> Layer 1 — Ember's Diaries: cognitive memory specification
> Layer 2 — **Memory Fabric** *(this paper)*: neural weight-space implementation
> Layer 3 — Fig Engine: training infrastructure
> Layer 4 — CogMem Benchmark: evaluation

---

## Abstract

Existing approaches to LLM memory fall into two failing categories. External retrieval systems (RAG, vector databases) require explicit lookup at inference time, introducing latency, context pressure, and retrieval boundaries. Destructive weight update systems (MemoryLLM, MEGa) modify base model parameters directly, causing catastrophic forgetting and instability after hundreds of edits. We propose a third approach: Memory Fabric, a neural architecture that encodes adaptive memory directly into dedicated trainable adapter parameters, organized according to the cognitive principles specified in Ember's Diaries, without modifying the pretrained base weights.

Memory Fabric introduces a dual-architecture within a single model: a frozen Cognitive Core (the pretrained base model) and a dynamic Memory Fabric (multiple LoRA adapter namespaces per layer with learned gating, confidence-as-magnitude, structural decay, and conflict routing). The Cognitive Core handles reasoning; Memory Fabric handles knowledge. No external retrieval store is required. Memory lives in dedicated adapter parameters that activate during inference via a learned per-namespace gate.

We describe the architecture, its relationship to the Ember's Diaries behavioral specification, two identified theoretical gaps (gate generalization and adapter delta generalization), and synthetic sandbox experiments validating that both gaps are mathematically bridgeable. The cross-session empirical validation (Stage 3) is ongoing and will be reported in a future revision. This paper is honest about what is proven and what is not.

---

## 1. Introduction

The memory problem in LLMs has two well-known failure modes.

**External retrieval systems** treat the model as stateless and memory as a separate database. At inference time, relevant documents are retrieved and injected into the context window. This approach has three structural costs: retrieval latency, context window pressure (memories consume tokens that could otherwise support active reasoning), and retrieval boundaries (facts that span chunk boundaries are lost). For a system intended to accumulate thousands of personal facts over months of interaction, these costs compound.

**Destructive weight update systems** write memories into the model's base parameters via gradient descent. MemoryLLM randomly drops old memory tokens. MEGa overwrites LoRA adapters with new information. Model editing approaches (ROME, MEMIT) modify the dense weight matrices of the base model directly. The consequence is catastrophic: base model capabilities degrade under continuous editing, and the system becomes unstable after hundreds of updates.

**SCR** [2024] and similar work identify these problems and conclude that contextual retrieval is more effective than parameter editing. They are correct — for a monolithic network where editing the base weights destroys reasoning. They have not evaluated an architecture where the base weights are frozen and memory occupies an isolated, dedicated adapter partition.

Memory Fabric introduces that architecture. The base model (Cognitive Core) is never modified. A separate Memory Fabric partition — multiple LoRA adapters per layer, organized by namespace — receives continuous micro-training updates. The Cognitive Core handles general reasoning. Memory Fabric handles personal knowledge, episodic history, and verified facts. They share the same forward pass but are structurally independent.

This is analogous to ROM/RAM separation in hardware: the instruction ROM (Cognitive Core) is read-only; the data RAM (Memory Fabric) is writeable and organized. Memory lives in dedicated parameters, not in an external database, and not in the base model weights.

### 1.1 Research Context

Memory Fabric is Layer 2 of the Harboria Labs research stack. It implements the behavioral specification defined in Ember's Diaries (Layer 1) within a neural architecture. The eight principles of Ember's Diaries — append-only history, supersession, confidence decay, episodic organization, conflict preservation, consolidation, reflection, and provenance — are approximated in neural form through adapter weight structure, magnitude-based confidence, structural decay, and namespace routing.

Fig Engine (Layer 3) provides the training infrastructure that makes continuous micro-training between conversation turns feasible on commodity hardware. CogMem Benchmark (Layer 4) evaluates whether the implementation actually improves memory, independently of the specific architecture.

### 1.2 Contributions

1. **Dual-architecture design** — A novel separation of Cognitive Core (frozen pretrained weights) and Memory Fabric (dynamic adapter partition) within a single model, enabling continuous learning without catastrophic forgetting and without external retrieval.

2. **Multi-adapter FigLinear** — N parallel LoRA adapters per linear layer, one per memory namespace, with learned per-namespace gating conditioned on input content.

3. **Confidence as adapter magnitude** — Structural implementation of Ebbinghaus confidence decay: adapter weight magnitude represents confidence; weight decay represents forgetting; re-training represents reinforcement. No confidence metadata required.

4. **Structural conflict detection** — Opposing cosine-distance adapter activations on the same input trigger routing to a contested namespace rather than silent averaging, preventing hallucination from blended conflicting facts.

5. **Consolidation pipeline** — Knowledge promotion from episodic to personal to wiki namespace following the Atkinson-Shiffrin memory model, via a structural adapter merging procedure.

6. **Theoretical gap analysis** — Formal identification of two unproven gaps (gate generalization and adapter delta generalization) and synthetic sandbox evidence that both are mathematically bridgeable.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG [Lewis et al., 2020] and its derivatives treat the LLM as stateless and memory as an external key-value store retrieved by semantic similarity. The fundamental limitation is architectural: memory and reasoning are decoupled, requiring explicit retrieval at every turn. Token costs, retrieval latency, and chunk boundary losses scale with the size of the memory store.

SCR [2024] argues that contextual reasoning over retrieved text is more reliable than parameter-level editing. This conclusion is correct within its scope — but its scope excludes isolated adapter partitions where the base model is never modified.

### 2.2 Memory in Model Weights

**MemoryLLM** [Wang et al., 2024] introduces memory pools within transformer layers, updated by replacing old tokens with new ones. Destructive: old memories are lost without explicit supersession.

**M+** [Wang et al., 2025] extends MemoryLLM with a retrieval bank. The pool still overwrites.

**Memory³** [Yang et al., 2024] externalizes specific knowledge as sparse attention KV pairs. Near-zero hallucination on those facts, but static — no continuous learning.

**MEGa** [2025] stores each memory as a gated LoRA adapter activated by query-matching. Near-perfect recall. The adapter weights are overwritten during fine-tuning on new memories — the same catastrophic forgetting problem in adapter form.

Memory Fabric differs from all of these: adapters are organized by *namespace* (type of knowledge) rather than by individual fact, gates are learned from input content rather than query-matched to fact identity, and the base model weights are never touched.

### 2.3 Model Editing

ROME [Meng et al., 2022], MEMIT, and related work localize and edit factual associations in base model weights. These approaches achieve high precision on individual edits but degrade rapidly under continuous editing — the dense base weights are not designed for incremental modification. Memory Fabric avoids this by writing only to isolated adapter parameters.

### 2.4 Continual Learning

Catastrophic forgetting [McCloskey & Cohen, 1989] is the fundamental problem of continual learning: updating neural network weights to learn new information degrades performance on previously learned information. Memory Fabric sidesteps this for the base model by never writing to it. Within the adapter partition, namespace isolation reduces inter-namespace interference, and structural weight decay provides a forgetting mechanism that mirrors biological memory rather than fighting it.

---

## 3. Architecture

### 3.1 Design Philosophy

The key design constraint is: **no external retrieval.** Memory Fabric encodes adaptive memory directly into trainable parameters rather than relying on an external retrieval store. At inference time, the model does not look up information — it holds it in dedicated adapter parameters that activate via a learned gating mechanism.

The secondary constraint is: **base weights are never modified.** The Cognitive Core (pretrained model) handles language understanding, reasoning, and general knowledge. The Memory Fabric handles personal facts, episodic history, and time-sensitive information. Both operate in the same forward pass but modify disjoint parameter sets.

### 3.2 Dual-Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      SINGLE MODEL                        │
│                                                          │
│  ┌──────────────────┐        ┌──────────────────────┐  │
│  │  COGNITIVE CORE  │        │    MEMORY FABRIC     │  │
│  │                  │        │                      │  │
│  │  Base weights    │        │  Multi-adapter LoRA  │  │
│  │  (FigQuant INT4) │        │  per namespace:      │  │
│  │                  │        │  • personal/  (r=8)  │  │
│  │  Frozen during   │        │  • episodic/  (r=16) │  │
│  │  memory updates  │        │  • wiki/      (r=32) │  │
│  │                  │        │  • schedule/  (r=4)  │  │
│  │  General         │        │  • contested/ (r=4)  │  │
│  │  intelligence    │        │                      │  │
│  └────────┬─────────┘        └──────────┬───────────┘  │
│           │          GATING             │              │
│           └──────────────┬──────────────┘              │
│                          │                             │
│             Internal activation bus                     │
└─────────────────────────────────────────────────────────┘
```

The forward pass flows through both subsystems simultaneously. A learned gating mechanism at each layer decides how much each namespace adapter contributes to the Cognitive Core's processing. This is not attention over an external key-value store — it is weight-space routing within the model itself.

### 3.3 Memory Namespaces

Memory Fabric organizes knowledge by type, not by individual fact:

| Namespace | Purpose | Adapter Rank | Decay Rate |
|---|---|---|---|
| `personal/` | Facts about the user | 8 | Medium |
| `episodic/` | Conversation history, events | 16 | Fast |
| `wiki/` | Verified, permanent knowledge | 32 | Near-zero |
| `schedule/` | Time-sensitive information | 4 | Fast |
| `contested/` | Conflicting information | 4 | Medium |

Higher rank = more capacity = used for knowledge that is more stable or complex. Higher decay rate = knowledge naturally becomes uncertain faster.

### 3.4 Multi-Adapter FigLinear

Standard LoRA adds one adapter pair (A, B) per linear layer. Memory Fabric requires N parallel adapters per layer, one per namespace:

```
output = base_weight(x) + Σᵢ gateᵢ(x) × (x @ Aᵢ) @ Bᵢ × scaleᵢ
```

where `gateᵢ(x)` ∈ [0, 1] is a learned per-namespace activation conditioned on the input. Different inputs activate different memory namespaces. The gate is a small linear projection followed by sigmoid:

```python
class MemoryGate(nn.Module):
    def __init__(self, hidden_size, n_namespaces):
        super().__init__()
        self.proj = nn.Linear(hidden_size, n_namespaces, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -1.0)   # starts mostly closed

    def forward(self, x):
        pooled = x.mean(dim=1) if x.dim() == 3 else x
        return torch.sigmoid(self.proj(pooled))
```

The gate starts with bias=-1 (sigmoid(-1) ≈ 0.27 — mostly closed), so namespaces only activate when the model has learned to open them for relevant inputs.

**Memory cost.** For 5 namespaces at average rank 12 on a 2048-dim model: 5 × 2 × 2048 × 12 × 4 bytes = 960 KB per layer. For 32 layers: ~30 MB total Memory Fabric. This fits alongside the FigQuant INT4 Cognitive Core in 8 GB RAM.

### 3.5 Confidence as Adapter Magnitude

Ember's Diaries specifies that memories should carry confidence scores with Ebbinghaus decay. Memory Fabric implements this *structurally* without explicit metadata:

**High confidence** = high adapter norm. The adapter contributes strongly to the forward pass.

**Decaying confidence** = selective weight decay applied proportionally to time since last access. The adapter contribution decreases naturally.

**Reinforcement** = micro-training on the same fact increases adapter magnitude.

**Forgetting** = a faded adapter contributes weakly to the output. The model becomes *less certain* about that knowledge without tracking a number.

At inference, uncertainty is structural — not a metadata field that could be corrupted or ignored, but a weight magnitude that directly determines how much the memory influences generation.

### 3.6 Conflict Detection in Activation Space

When the same input activates two namespace adapters that produce opposing hidden state contributions (high cosine distance between their outputs), this signals a conflict:

```
conflict = cosine_similarity(adapter_ns_A(x), adapter_ns_B(x)) < −0.5
```

On conflict detection:

1. Gating mechanism routes to the `contested/` namespace
2. Contested adapter holds the uncertainty signal
3. Cognitive Core generates a response that surfaces the conflict
4. User resolution → winning version promoted, losing version decayed

This is a structural implementation of Ember's Diaries conflict preservation principle: both versions persist rather than being silently averaged into a hallucinated middle ground.

### 3.7 Knowledge Consolidation

Information is promoted through a three-stage pipeline inspired by the Atkinson-Shiffrin model:

```
Heard once → episodic/ adapter (high decay, rank 16)
     ↓ repeated exposure across sessions
Reinforced → personal/ adapter (medium decay, rank 8)
     ↓ user confirmation / multi-source agreement
Verified → wiki/ adapter (near-zero decay, rank 32)
```

Structurally, consolidation copies a fraction of the source namespace adapter into the target namespace adapter, then applies decay to the source:

```python
def promote(from_ns, to_ns, scale=0.3):
    # Project source adapter into target rank (truncate or pad)
    r = min(src_rank, dst_rank)
    dst_A.data[:, :r] += src_A.data[:, :r] * scale
    dst_B.data[:r, :] += src_B.data[:r, :] * scale
    # Source adapter decays but is not deleted
```

Once in the `wiki/` namespace, knowledge is effectively permanent — analogous to how a pretrained model "just knows" that Paris is the capital of France. The model built this knowledge from its own experience rather than from pretraining data.

### 3.8 Micro-Training Between Turns

Memory writes occur between conversation turns, not during generation:

1. **During conversation** — Cognitive Core generates. Hidden states at gating layers signal "store this" (learned from Memory Fabric token training: `<|mem_store|>`, `<|mem_recall|>`, etc.).

2. **At turn boundary** — Pending memories are buffered as micro-training examples.

3. **Between turns (target: <100ms)** — Fig Engine runs 1–5 LoRA steps on the relevant namespace adapter via MicroTrainer. FigMeZO enables this without a backward pass on memory-constrained hardware.

4. **Next turn** — Memory is in weights. No retrieval needed.

The user experiences zero latency — the model "just remembers."

### 3.9 Relationship to Ember's Diaries

Memory Fabric is one neural implementation of the Ember's Diaries specification. The mapping is approximate — constrained by what gradient descent can encode in low-rank matrices — but the structural correspondence is direct:

| Ember's Diaries Principle | Memory Fabric Neural Implementation |
|---|---|
| Append-only history | New adapters coexist with old (no overwrite) |
| Supersession | New adapter trained; old one attenuated via decay |
| Confidence decay | Selective weight decay proportional to time since access |
| Epistemic status | Adapter magnitude = confidence level |
| Namespace organization | Separate LoRA adapters per knowledge domain |
| Conflict preservation | Opposing adapter activations → contested/ namespace routing |
| Consolidation | promote(): weak adapters merge into higher-rank verified adapter |
| Episode segmentation | Training batch boundaries = episode boundaries |
| Provenance | Adapter training metadata (step, timestamp, source) |
| Reflection | Adapter meta-state annotations |

Ember's Diaries does not care how these behaviors are implemented. A key-value database, a graph store, or a different neural architecture could each approximate the same specification. CogMem Benchmark evaluates any implementation against these principles independently.

---

## 4. Theoretical Analysis

Before the Stage 3 empirical validation can be run, two theoretical questions must be addressed: (1) Will the gate activate correctly for query sequences given training only on storage sequences? (2) Will the adapter delta from training on a stored fact generalize to the query hidden states?

We call these Gap 1 and Gap 2.

### 4.1 Gap 1 — Gate Generalization

The gate is trained on the storage sequence: "My API key expires September 1st." Will it fire for the query sequence: "When does my API key expire?"

These are different token sequences with different hidden states. The gate is a linear projection (small, 64×5 for 5 namespaces). Will a linear projection learned on storage sequences generalize to semantically related query sequences?

**Synthetic sandbox evidence.** We constructed pairs of (storage, query) hidden state vectors at controlled cosine similarities (0.3 to 0.95), trained the gate on storage embeddings, and measured activation on query embeddings. Results across 10 store/query pairs at similarity=0.70:

```
store activation: 0.563 ± 0.009
query activation: 0.562 ± 0.012
gap:              0.001
```

The gate generalizes almost perfectly — near-identical activation for store and query embeddings at typical semantic similarity levels. The linear projection naturally learns the shared feature space between storage and retrieval of the same fact.

**The real Gate 1 issue.** The gap is not that the gate doesn't generalize — it does. The issue is that 3 gradient steps are insufficient to meaningfully open the gate. After 3 steps at lr=5e-4, gate activation moves from 0.272 (sigmoid(-1), initial closed state) to approximately 0.278 — a change of 0.006. The gate does not contribute meaningfully to the forward pass.

**Fix identified.** The gate requires a higher learning rate than the adapter parameters, applied through separate parameter groups:

```python
optimizer = torch.optim.SGD([
    {'params': adapter_params, 'lr': 5e-4},     # adapter: careful steps
    {'params': gate_params,    'lr': 1e-2},      # gate: fast opening (20×)
])
```

This fix is implemented in Memory Fabric v0.6 and will be validated in Stage 3.

### 4.2 Gap 2 — Adapter Delta Generalization

After 3-step micro-training on the storage hidden state, does the adapter delta generalize to the query hidden state?

The adapter is trained: "given hidden states from 'API key expires September 1st', push output toward the target representation." The query produces different hidden states. Low-rank matrix modifications trained on one distribution don't automatically generalize to another.

**Synthetic sandbox evidence.** We measured the adapter delta transfer ratio (query improvement / store improvement) across 200 pairs at each similarity level:

| Semantic Similarity | Store Δ | Query Δ | Transfer Ratio |
|---|---|---|---|
| 0.30 | 1.0 | 0.50 | 0.50 |
| 0.50 | 1.0 | 0.87 | 0.87 |
| 0.70 | 1.0 | 0.99 | 0.99 |
| 0.85 | 1.0 | 1.00 | 1.00 |
| 0.95 | 1.0 | 1.00 | 1.00 |

At typical real-world semantic similarity between store and query sequences (~0.70), the transfer ratio is 0.99 — near-perfect. At low similarity (0.30), transfer is still 0.50, suggesting meaningful even in difficult cases.

**The real Gap 2 issue.** The B=zeros initialization creates a gradient propagation problem. With B initialized to zeros, the adapter output is x @ A @ 0 = zero vector. Cosine similarity loss against a zero vector is undefined, and gradient flow through B is degenerate on the first step.

**Fix identified.** Initialize B with small random values:

```python
# Before (broken):
self.B = nn.Parameter(torch.zeros(rank, hidden_size))

# After (fixed):
self.B = nn.Parameter(torch.empty(rank, hidden_size))
nn.init.normal_(self.B, std=0.02)
```

This fix is implemented in Memory Fabric v0.6.

### 4.3 Summary: Status of Theoretical Gaps

| Gap | Status | Evidence | Fix Implemented |
|---|---|---|---|
| Gap 1 (gate generalization) | Mathematically bridgeable | Synthetic sandbox: gap = 0.001 | Yes — separate gate lr |
| Gap 2 (adapter delta transfer) | Mathematically bridgeable | Synthetic sandbox: ratio ≥ 0.87 at sim ≥ 0.5 | Yes — B noise init |

Both gaps are confirmed as implementation issues, not fundamental architectural blockers. Stage 3 validates these fixes empirically on the actual model.

---

## 5. Stage 3: The Cross-Session Experiment

The critical empirical test that determines whether Memory Fabric works. Stage 3 is ongoing and results will be reported in a future revision.

### 5.1 Protocol

**Session A (storage):**
```python
# Tokenize a personal fact
input_ids = tokenize("My API key expires September 1st")

# Write to personal/ namespace via MicroTrainer
trainer.write_memory(
    model=model,
    namespace="personal",
    input_ids=input_ids,
    labels=input_ids,
)
# Context is cleared. No memory stored in prompt.
```

**Session B (retrieval — no context injection):**
```python
# Query with semantically related but different tokens
output = model.generate(
    tokenize("When does my API key expire?"),
    # NO memory injected into context
)

# Measure: does output contain "September" or "September 1st"?
```

### 5.2 Success Criteria

The cross-session experiment is considered successful if:
- Top-5 token probability for "September" increases relative to baseline (no micro-training)
- The model produces the correct answer for at least 3 of 10 distinct personal facts
- Performance degrades gracefully with semantic distance (lower accuracy at higher paraphrase distance)

A score of 30% would constitute novel evidence that weight-space memory works. A score of 0% would identify the specific failure mode (gate activation, adapter delta, or training stability) for targeted debugging.

### 5.3 Measurement Plan

Three diagnostic measurements are taken at each step:

1. **Gate activation** — Does the personal/ namespace gate fire for the query? Measured as σ(gate.proj(query_hidden_mean)).

2. **Adapter contribution** — Does the adapter delta point toward the correct answer? Measured as cosine_similarity(adapter_delta(query_hidden), target_embedding).

3. **Generation accuracy** — Does the model produce the correct factual answer? Measured by token-level match and perplexity on the correct continuation.

Each measurement isolates a different component of the system, enabling targeted diagnosis if the end-to-end result is below threshold.

---

## 6. Implementation

Memory Fabric is implemented as part of the littlefig repository:

```
src/little_fig/engine/
├── memory_fabric.py   # MultiAdapterLayer, MemoryFabric, MemoryGate
├── micro_trainer.py   # MicroTrainer, MicroTrainConfig
└── ember_integration.py  # EmberTrainingDataGenerator, special tokens
```

**Special tokens** injected into the model vocabulary:

```
<|mem_store|>        <|mem_recall|>       <|mem_consolidate|>
<|mem_forget|>       <|mem_conflict|>     <|mem_episode|>
<|mem_reflect|>      <|memory_start|>     <|memory_end|>
```

**Installation:** `pip install git+https://github.com/Harboria-Labs/littlefig.git`

---

## 7. Limitations

**Stage 3 results pending.** This paper reports an architecture and preliminary synthetic evidence. The critical cross-session empirical validation is ongoing. The architecture is sound; whether the implementation achieves reliable cross-session retrieval in practice is the open question.

**Small model testing only.** Current experiments target TinyLlama (1.1B). Larger models may have stronger hidden state semantics that improve gate generalization and adapter transfer; they may also have more interference between namespaces.

**Namespace interference at scale.** The synthetic sandbox tests single-fact storage. Storing 100+ facts across namespaces introduces interference between adapter updates. Stage 5 of the research roadmap addresses this.

**Consolidation unvalidated.** The promote() consolidation procedure is architecturally sound but has not been tested empirically. Stage 6 of the roadmap addresses this.

---

## 8. Research Roadmap

| Stage | Description | Status |
|---|---|---|
| 1 | Literature & challenge mapping | Complete |
| 2 | Synthetic sandbox: gap isolation | Complete — both gaps bridgeable |
| 3 | Cross-session baseline (red box experiment) | **In progress** |
| 4 | Triage & fix (gate or adapter, based on Stage 3) | Pending |
| 5 | Scaling: 5 → 20 → 100 facts, namespace interference | Pending |
| 6 | Consolidation loop: promote episodic → wiki | Pending |

---

## 9. Conclusion

Memory Fabric introduces a third approach to LLM memory: isolated adapter partitions that implement the behavioral principles of Ember's Diaries without modifying the pretrained base model and without requiring external retrieval.

The architecture is theoretically coherent. Synthetic sandbox experiments confirm that the two identified theoretical gaps (gate generalization and adapter delta transfer) are mathematically bridgeable rather than fundamental blockers. Implementation fixes for both have been identified and applied.

The open empirical question is whether real transformer hidden states — produced by actual tokenized sequences through the full model — have sufficient semantic similarity between storage and retrieval contexts to achieve reliable cross-session recall. That question is Stage 3.

If Stage 3 succeeds, Memory Fabric demonstrates that a model can learn persistent personal facts in its weights between conversation turns, with no external database, no context overhead, and no base model degradation. If it fails at the expected rate, the failure mode will directly guide Stage 4 architectural improvements.

The architecture is right. The question is whether the implementation is ready.

---

## References

1. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
2. Wang, Y., et al. "MEMORYLLM: Towards Self-Updatable Large Language Models." arXiv:2402.04624, 2024.
3. Wang, Y., et al. "M+: Extending MemoryLLM with Scalable Long-Term Memory." arXiv:2502.00592, 2025.
4. Yang, H., et al. "Memory³: Language Modeling with Explicit Memory." arXiv:2407.01178, 2024.
5. Meng, K., et al. "Locating and Editing Factual Associations in GPT." NeurIPS 2022 (ROME).
6. MEGa. "Memorization and Knowledge Injection in Gated LLMs." arXiv:2504.21239, 2025.
7. McCloskey, M. & Cohen, N.J. "Catastrophic Interference in Connectionist Networks." Psychology of Learning and Motivation, 1989.
8. Atkinson, R.C. & Shiffrin, R.M. "Human Memory: A Proposed System." 1968.
9. Ebbinghaus, H. "Über das Gedächtnis." 1885.
10. Conway, M.A. "Memory and the Self." Journal of Memory and Language, 2005.
11. HaluMem. "Evaluating Hallucinations in Memory Systems of Agents." arXiv:2511.03506, 2024.
12. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
13. 0xticketguy (Harboria Labs). "Ember's Diaries: An Immutable Cognitive Database Engine for Grounded AI Memory." 2026.
14. 0xticketguy (Harboria Labs). "Fig Engine: CPU-Native Training Infrastructure for Large Language Models." 2026.
15. 0xticketguy (Harboria Labs). "CogMemBench: A Benchmark for Continuous Cognitive Memory in Large Language Models." 2026.

---

*Code: https://github.com/Harboria-Labs/littlefig*
*License: CC-BY-4.0*
*Built by 0xticketguy / Harboria Labs*
