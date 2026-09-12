# Compression Results — What the Numbers Actually Mean

This decodes the `benchmark.py` output into plain language: what each column
measures, what each model variant is, and what to actually take away from it.

## The raw table

| Variant | Size | Params | Sparsity | MACs | Latency (ms) | Accuracy |
|---|---|---|---|---|---|---|
| baseline_best | 959.9 KB | 242,508 | 0.0% | 186,222,336 | 17.072 ± 6.757 | 91.88% |
| pruned_structured_30-30-30-30 | 101.6 KB | 22,077 | 0.0% | 17,019,456 | 2.526 ± 0.227 | 79.19% |
| pruned_structured_70-70-70-70 | 486.4 KB | 120,291 | 0.0% | 91,472,400 | 8.051 ± 2.590 | 92.82% |
| pruned_unstructured_50 | 960.6 KB | 242,508 | 49.8% | 186,222,336 | 12.585 ± 2.046 | 93.99% |
| quantized_ptq_fbgemm\_\_from_baseline_best | 287.0 KB | N/A | N/A | N/A | 6.102 ± 3.408 | 91.61% |
| quantized_ptq_fbgemm\_\_from_pruned_structured_70-70-70-70 | 168.4 KB | N/A | N/A | N/A | 4.585 ± 0.877 | 92.35% |
| quantized_qat_fbgemm\_\_from_baseline_best | 287.0 KB | N/A | N/A | N/A | 5.600 ± 1.905 | 93.49% |
| **quantized_qat_fbgemm\_\_from_pruned_structured_70-70-70-70** | **168.4 KB** | N/A | N/A | N/A | **2.903 ± 0.529** | **93.41%** |

## What each column means

- **Size** — the actual file size of the saved checkpoint on disk. This is
  the number that matters for "does it fit in flash/memory on the target
  device."
- **Params** — total number of weights in the model. `N/A` for quantized
  models because their weights are packed into 8-bit integer tensors
  internally, which our parameter-counting code (written for plain float
  models) doesn't see the same way. It's a benchmarking-script limitation,
  not a sign anything went wrong with those models.
- **Sparsity** — the percentage of weights that are exactly zero. Only
  meaningful for unstructured pruning; everything else is 0.0% because
  their weights are still (or newly) dense.
- **MACs** (multiply-accumulate operations) — a hardware-relevant proxy for
  "how much computation does one inference actually do." Also `N/A` for
  quantized models for the same reason as Params above.
- **Latency** — how long one single inference takes on this machine's CPU,
  averaged over 100 runs, single-threaded (mean ± standard deviation). This
  is a *software reference number*, not a promise about FPGA performance —
  it's here so we have an apples-to-apples comparison across variants.
- **Accuracy** — percentage correct on the held-out test set.

## What each variant actually is

- **baseline_best** — the original trained model, no compression.
- **pruned_structured_X-X-X-X** — channels were physically removed from each
  of the 4 conv blocks, keeping only X% of channels per block. This
  actually shrinks the tensors, so size/speed genuinely improve.
- **pruned_unstructured_50** — 50% of individual weights were zeroed out
  (magnitude-based), but the tensors stay full-size. This is why its size
  and latency barely differ from baseline — a normal CPU multiplies by a
  zero exactly as slowly as by anything else, so pruning like this only
  pays off with specialized sparse-compute hardware, which we don't have.
- **quantized_ptq / quantized_qat** — the same model converted from 32-bit
  floating point down to 8-bit integers. `ptq` (post-training quantization)
  just converts an already-trained model; `qat` (quantization-aware
  training) fine-tunes the model *with* the quantization noise simulated
  during training, so the weights adapt to it. The `__from_X` suffix says
  which model was quantized — e.g. `__from_pruned_structured_70-70-70-70`
  means we quantized the already-pruned model, stacking both techniques.

## The takeaways

1. **Unstructured pruning is a dead end for this hardware target.** 49.8%
   sparsity, but no size or speed benefit. It's included because it's the
   standard reference point in the literature, but it isn't a real
   candidate for deployment here.
2. **Structured pruning at a moderate ratio (70% keep) is a genuine win.**
   About 2x smaller, ~2x faster, and accuracy didn't drop — in fact it
   ticked up slightly (92.82% vs. baseline's 91.88%).
3. **Quantization is the single biggest lever.** Every quantized variant is
   3-4x smaller and 3-6x faster than baseline, largely independent of which
   model was quantized.
4. **Best overall result: structured pruning + QAT, stacked.**
   `quantized_qat_fbgemm__from_pruned_structured_70-70-70-70` is **~5.7x
   smaller and ~5.9x faster than baseline, with equal-or-better accuracy**
   (93.41% vs. 91.88%). This is the strongest deployment candidate and the
   headline number for the project.
5. **Aggressive structured pruning (30% keep) is too aggressive on its
   own** — a 13-point accuracy drop for extra speed you probably don't
   need on top of what quantization already gives you. Worth keeping in
   the results as a "how far is too far" data point, not as a deployment
   candidate.

## Caveats — read before quoting these numbers elsewhere

- **Latency numbers are noisy.** Baseline shows `17.072 ± 6.757 ms` — a
  spread of ~40% of the mean. This batch was measured right after several
  heavy training/quantization jobs ran back-to-back, so background system
  load likely inflated and destabilized the timings. The *relative*
  ordering (baseline is clearly slowest; quantized variants are clearly
  fastest) is trustworthy, since the gaps are several times larger than the
  noise. Small differences — e.g. PTQ vs. QAT latency for the same source
  model — are within the noise and shouldn't be over-interpreted; in an
  earlier benchmark run, PTQ was faster than QAT, and in this one QAT was
  faster than PTQ. Re-running `benchmark.py` on its own (not immediately
  after a big batch job) would give a cleaner reading before this goes in
  a final report.
- **Some accuracy gains over baseline are probably partly "more training,"
  not purely "compression helping."** Pruning and QAT both involve
  additional fine-tuning epochs on top of the original training run, which
  on its own can nudge accuracy up a little (extra gradient steps, mild
  regularization from a smaller/noisier model). It's a real result, but
  it's fair to describe it as "pruning/quantization didn't hurt, and the
  extra fine-tuning likely helped a bit," rather than "pruning made the
  model better than training longer would have."
- **`pruned_unstructured_50` was never quantized** — there's no
  `..._from_pruned_unstructured_50` row in this table. Worth adding for
  completeness, though given point 1 above it's unlikely to add much new
  information versus quantizing the baseline directly.

## For the FPGA implementation

The most relevant reference numbers to hand off are for
`pruned_structured_70-70-70-70` (the pre-quantization structured-pruned
model): its actual per-layer channel counts, MAC counts, and weight value
ranges are in `results/hw_report_pruned_structured_70-70-70-70.json` if
`benchmark.py` was run with `--hardware-report`. That gives an accurate
picture of the computation and dynamic range the FPGA datapath needs to
support, independent of which CPU quantization backend (`fbgemm`/`qnnpack`)
was used to produce the PyTorch-side INT8 numbers above.
