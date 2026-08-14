# Little Fig Research Session Summary

_Last updated: 2026-08-14 (Africa/Lagos)_

This is the durable cross-session handoff. Read it with `RESEARCH_PROGRESS.md` when
resuming work, and append a dated entry after each substantive session.

## Current objective

Finish P2: verify FigQuant on all 156 TinyLlama matrices and save the complete JSON.
Do not call the claim reproduced until the final JSON has 156 records.

## State at handoff

- Branch: `research/p1-figmezo-verify`.
- P1 FigMeZO is complete; the claimed -18.6% held-out improvement was refuted.
- P2 GPT-2 is reproduced: 50/50 wins and 5.280921% lower mean reconstruction MSE
  versus NF4.
- The user's TinyLlama run visibly completed 154 layers, with FigQuant better than
  NF4 on every completed layer. It then printed `[155/156] START lm_head.weight`
  followed by `^C`; layers 155 (`lm_head.weight`) and 156
  (`model.embed_tokens.weight`) have no result.
- That run had no checkpoint, so the 154 metrics cannot be reconstructed from the
  partial console output. This is partial evidence only.

## 2026-08-14 recovery work

The old benchmark allocated a `[65,536,000, 16]` FP32 distance tensor for the final
embedding matrix, about 3.9 GiB before other tensors. NF4 and uniform INT4 had the
same problem. A local `/content` checkpoint would also disappear after a Colab reset.

Changes made:

- `src/little_fig/engine/figquant.py`: bounded-chunk k-means assignment/update.
- `benchmark/experiment_figquant_v2.py`: atomic per-layer checkpoints, `--resume`,
  peak RSS records, bounded-memory baselines, and `--results-path`.
- `benchmark/P2_FigQuant_Colab.ipynb`: Google Drive mount, Drive checkpoint/results
  directory, TinyLlama-only execution, and resume-by-default workflow.
- `RESEARCH_PROGRESS.md`: session pointer added.

## 2026-08-14 memory-evidence extension

The P2 run now evaluates the 8 GiB objective explicitly rather than treating memory
as incidental logging. It records process peak RSS, whole-system total/available
memory, the minimum available-memory reading, per-layer workspace growth, the layer
responsible for peak RSS, budget utilization, and headroom against the configurable
`--memory-budget-gib` target (8 GiB by default).

A separate `figquant_v2_memory_<model>.json` heartbeat is atomically updated in the
Drive checkpoint directory every two seconds. If Colab kills the process inside the
final layer, this file preserves the phase, last RSS, observed peak RSS, and system
available-memory floor even though that layer never reaches its normal checkpoint.

Interpretation boundary: P2 measures the memory needed to benchmark quantization
quality while holding FP32 model weights, not the complete Fig Engine fine-tuning
pipeline. Passing 8 GiB here supports the quantizer's bounded-memory behavior; it
does not by itself prove the broader "fine-tune on 8 GB RAM" claim. That requires a
separate end-to-end training memory benchmark.

Verification: a synthetic run measured 0.397 GiB peak process RSS, correctly named
the largest matrix as the peak layer, measured 0.120 GiB peak layer workspace growth,
persisted the memory heartbeat, and passed a deliberately tight 0.5 GiB budget with
0.103 GiB headroom. The temporary verification files were removed afterward.

Notebook cleanup: the obsolete `benchmark/P1_FigMeZO_Colab.ipynb` testing notebook
was deleted. Keep `Little_Fig_Colab.ipynb` as the public/main usage notebook and
`benchmark/P2_FigQuant_Colab.ipynb` as the sole research-testing notebook.

## Exact next steps

1. Commit and push the current branch so Colab can clone the updated script. The
   notebook intentionally clones `research/p1-figmezo-verify`.
2. Open `benchmark/P2_FigQuant_Colab.ipynb` in Colab and run all cells.
3. Leave `RESUME = True`. On first run it starts a new Drive checkpoint; after any
   interruption, rerun setup/benchmark cells and completed layers are skipped.
4. Confirm Drive contains `littlefig-p2/figquant_v2_checkpoint_TinyLlama_TinyLlama-1.1B-Chat-v1.0.json`.
5. When complete, retrieve `littlefig-p2/figquant_v2_results.json`, commit it under
   `benchmark/`, and update P2 with exact 156-layer numbers.

## Verification required

- Run synthetic stop/resume and unit tests locally.
- Re-run GPT-2 with chunked code and compare against the committed 50/50,
  5.280921% reference within tolerance.
- Keep the TinyLlama verdict partial until all 156 records are present.
