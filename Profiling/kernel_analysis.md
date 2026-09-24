# ML Kernel Analysis -- KeywordSpottingCNN

Written after re-profiling the **current** `src/model.py` (4 conv blocks).
All earlier profiling in this folder described the old 3-block model and was stale.

## What changed in the model

| | Old (profiled at commit 7788e6a) | Current |
| --- | --- | --- |
| Conv blocks | 3 (32, 64, 64) | 4 (32, 64, 128, 128) |
| MaxPool stages | 1 | 3 |
| Dropout | none | 0.3 before classifier |
| Classifier | Linear(64, 12) | Linear(128, 12) |
| Parameters | 56,844 | 242,508 |
| MACs / inference | 112.95 M | 186.77 M |

`src/train.py` was also pinned to CPU (`device = torch.device("cpu")`), with the
CUDA selection commented out. This does not affect the kernel shapes.

## The kernel

The whole network is one kernel repeated four times: **3x3 same-padded 2-D
convolution**, each followed by BatchNorm + ReLU, and (after blocks 2-4) a 2x2
max-pool. Everything else is negligible arithmetic: global average pool, one
Linear(128, 12) at 1,548 MACs, and the BatchNorms at 256 MACs each.

Convolution is **99.3% of all MACs**. Accelerate 3x3 conv and nothing else matters.

## Per-layer GEMM mapping (batch 1, im2col form)

`M` = output channels, `K` = in_channels x 9, `N` = output pixels (H x W).

| Layer | M | K | N | MACs | Share | 32x32 array utilisation |
| ----- | - | - | - | ---- | ----- | ----------------------- |
| conv1 `features[0]`  | 32  | 9    | 4040 | 1,292,800  | 0.7%  | rows 28%, cols 100% |
| conv2 `features[3]`  | 64  | 288  | 4040 | 74,723,840 | 40.0% | fully tiled |
| conv3 `features[7]`  | 128 | 576  | 1000 | 73,856,000 | 39.5% | fully tiled |
| conv4 `features[11]` | 128 | 1152 | 250  | 36,896,000 | 19.8% | fully tiled |

Two observations that matter for the accelerator:

1. **The MAC load is now flat, not peaked.** In the old model one layer held
   66% of the MACs. Now conv2 and conv3 are nearly tied at ~40% each and conv4
   adds 20%. A design tuned for a single hot layer will not pay off; the array
   must run all three well.
2. **conv1 is the only badly shaped layer.** With `K = 9` it fills 9 of 32 rows
   of a 32x32 array. It is only 0.7% of the work, so it is cheap to leave it on
   the host or pad it, but do not size the array around it.

## Measured CPU time breakdown (batch 1, 100 iterations, torch.profiler)

| Op | Self CPU | Share |
| -- | -------- | ----- |
| `aten::mkldnn_convolution` | 249,202 us | 55.2% |
| `aten::max_pool2d_with_indices` | 123,116 us | 27.3% |
| `aten::native_batch_norm` | 22,851 us | 5.1% |
| `aten::clamp_min_` (ReLU) | 9,233 us | 2.0% |
| everything else | -- | ~10% |

Convolution totals **58%** of host runtime (up from 47% on the old model), which
is the Amdahl fraction now used for the roofline.

**Max-pool at 27% of runtime is the surprise.** It is ~0% of the MACs, so it is
pure memory movement, and the new model has three max-pool stages instead of
one. On the host it costs almost half as much as all the convolution. If the
accelerator offloads only convolution, max-pool becomes the next wall and caps
the system speedup. Folding pooling into the conv output path (pool-on-write, or
strided conv) is the obvious co-design item.

## Roofline summary (current model)

Both configurations are **compute-bound**, not bandwidth-bound, under the
`fused` dataflow (weights resident, activations kept in on-chip SRAM).

| | FP32, 16x16 @ 500 MHz | INT8, 32x32 @ 500 MHz |
| --- | --- | --- |
| Arithmetic intensity | 378.75 FLOP/byte | 1,515.01 FLOP/byte |
| Ridge point | 160.0 | 640.0 |
| Chiplet peak | 256.0 GFLOP/s | 1,024.0 GFLOP/s |
| Latency / inference | 1459.1 us | 364.8 us |
| Required interface BW | 0.68 GB/s (2.37x margin) | 0.17 GB/s |
| Kernel speedup vs host | 3.0x | ~12x |
| System speedup (Amdahl, 58%) | 1.63x | ~2.14x |

On-chip SRAM needed for the FP32 fused dataflow: 947.3 KiB weights +
1515.0 KiB for the largest live activation pair = **2462.3 KiB**. The activation
buffer, not the weights, dominates -- driven by conv2's 64 x 40 x 101 output.

## Caveats on these numbers

- CPU peak (268.8 GFLOP/s) and interface bandwidth are **spec-sheet hypotheses**
  for the i5-10210U host, not measurements. Only the achieved GFLOP/s is timed.
- `torchaudio` is not installed here, so input shape falls back to
  `src/dataset.py`'s constants (40 mels x 101 frames). Those are the real values.
- Weights are randomly initialised; MACs and shapes do not depend on training.
- Unstructured pruning will not reduce these MACs -- the zeros still occupy the
  tensors. `results/benchmark_table.md` confirms this: `pruned_unstructured_50`
  keeps all 186 M MACs at 49.8% sparsity, while structured 70% pruning cuts them
  to 91.5 M.
