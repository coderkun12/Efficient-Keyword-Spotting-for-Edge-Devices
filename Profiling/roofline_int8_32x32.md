# Roofline Analysis -- Keyword Spotting CNN

**Model:** `KeywordSpottingCNN` (56,844 params, 12 classes)  
**Input:** `[1, 1, 40, 101]` (batch, channel, n_mels, time frames)  
**Precision:** INT8 (1 byte/element)  
**Dataflow:** `fused` -- weights resident in the array; activations stay in on-chip SRAM between layers (batch 1)

![Roofline](roofline.png)

## Platforms

> Peak numbers are **hypotheses** except where marked measured. Override with `--cpu-gflops`, `--cpu-bw`, `--array`, `--freq-mhz`, `--chiplet-bw`.

| Platform | Peak | Bandwidth | Ridge point |
| -------- | ---- | --------- | ----------- |
| i5-10210U (host) | 268.8 GFLOP/s | 45.8 GB/s | 5.9 FLOP/byte |
| Chiplet 32x32 @ 500 MHz | 1,024.0 GFLOP/s (1.024 TFLOP/s) | 1.6 GB/s | 640.0 FLOP/byte |

## Workload

| Quantity | Value |
| -------- | ----- |
| MACs per inference | 112,945,740 |
| FLOPs per inference | 225,891,480 (225.9 MFLOP) |
| DRAM bytes per batch | 60,896 |
| **Arithmetic intensity** | **3,709.46 FLOP/byte** |
| Bound on chiplet | **compute-bound** (AI 3,709.5 vs ridge 640.0) |
| Attainable on chiplet | 1,024.0 GFLOP/s |
| Latency per inference | 220.6 us |

## Interface Bandwidth

```
Required BW = chiplet peak / arithmetic intensity
            = 1024.0 GFLOP/s / 3,709.46 FLOP/byte
            = 0.28 GB/s
```

Provided: 1.6 GB/s (5.80x margin).

## On-Chip SRAM Required

| Buffer | Size |
| ------ | ---- |
| Weights (all layers, resident) | 55.5 KiB |
| Activations (largest live pair) | 378.8 KiB |
| **Total** | **434.3 KiB** |

## Speedup

| Quantity | Value |
| -------- | ----- |
| Host CPU achieved (measured, batch 1) | 83.3 GFLOP/s (2.71 ms/inference) |
| Chiplet attainable | 1,024.0 GFLOP/s |
| Kernel speedup | 12.3x |
| Accelerated fraction (Amdahl) | 47% |
| **System speedup** | **1.76x** |

```
Speedup_system = 1 / ((1 - 0.470) + 0.470 / 12.3)
               = 1.76x
```

## Dataflow Comparison

Same model, same precision -- only the DRAM traffic model changes:

| Dataflow | DRAM bytes | AI (FLOP/byte) |
| -------- | ---------- | -------------- |
| `none` | 1,609,760 | 140.33 |
| `ws` | 1,609,760 | 140.33 |
| `fused` | 60,896 | 3,709.46 |

Weight-stationary alone moves the needle very little at batch 1: the weights
are only 56 KiB of the traffic, while the activations are the bulk.
The large gain comes from keeping activations on-chip between layers (`fused`).

## Precision Note

Operation count is fixed by the architecture, so AI scales inversely with bytes
per element: FP32 -> INT8 divides DRAM traffic by 4 and multiplies AI by 4.
Re-run with `--bytes-per-elem 1` for the INT8 roofline.
