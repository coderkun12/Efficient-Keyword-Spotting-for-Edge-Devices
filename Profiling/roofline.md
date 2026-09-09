# Roofline Analysis -- Keyword Spotting CNN

**Model:** `KeywordSpottingCNN` (56,844 params, 12 classes)  
**Input:** `[1, 1, 40, 101]` (batch, channel, n_mels, time frames)  
**Precision:** FP32 (4 byte/element)  
**Dataflow:** `fused` -- weights resident in the array; activations stay in on-chip SRAM between layers (batch 1)

![Roofline](roofline.png)

## Platforms

> Peak numbers are **hypotheses** except where marked measured. Override with `--cpu-gflops`, `--cpu-bw`, `--array`, `--freq-mhz`, `--chiplet-bw`.

| Platform | Peak | Bandwidth | Ridge point |
| -------- | ---- | --------- | ----------- |
| i5-10210U (host) | 268.8 GFLOP/s | 45.8 GB/s | 5.9 FLOP/byte |
| Chiplet 16x16 @ 500 MHz | 256.0 GFLOP/s (0.256 TFLOP/s) | 1.6 GB/s | 160.0 FLOP/byte |

## Workload

| Quantity | Value |
| -------- | ----- |
| MACs per inference | 112,945,740 |
| FLOPs per inference | 225,891,480 (225.9 MFLOP) |
| DRAM bytes per batch | 243,584 |
| **Arithmetic intensity** | **927.37 FLOP/byte** |
| Bound on chiplet | **compute-bound** (AI 927.4 vs ridge 160.0) |
| Attainable on chiplet | 256.0 GFLOP/s |
| Latency per inference | 882.4 us |

## Interface Bandwidth

```
Required BW = chiplet peak / arithmetic intensity
            = 256.0 GFLOP/s / 927.37 FLOP/byte
            = 0.28 GB/s
```

Provided: 1.6 GB/s (5.80x margin).

## On-Chip SRAM Required

| Buffer | Size |
| ------ | ---- |
| Weights (all layers, resident) | 222.0 KiB |
| Activations (largest live pair) | 1515.0 KiB |
| **Total** | **1737.0 KiB** |

## Speedup

| Quantity | Value |
| -------- | ----- |
| Host CPU achieved (measured, batch 1) | 77.2 GFLOP/s (2.93 ms/inference) |
| Chiplet attainable | 256.0 GFLOP/s |
| Kernel speedup | 3.3x |
| Accelerated fraction (Amdahl) | 47% |
| **System speedup** | **1.49x** |

```
Speedup_system = 1 / ((1 - 0.470) + 0.470 / 3.3)
               = 1.49x
```

## Dataflow Comparison

Same model, same precision -- only the DRAM traffic model changes:

| Dataflow | DRAM bytes | AI (FLOP/byte) |
| -------- | ---------- | -------------- |
| `none` | 6,439,040 | 35.08 |
| `ws` | 6,439,040 | 35.08 |
| `fused` | 243,584 | 927.37 |

Weight-stationary alone moves the needle very little at batch 1: the weights
are only 222 KiB of the traffic, while the activations are the bulk.
The large gain comes from keeping activations on-chip between layers (`fused`).

## Precision Note

Operation count is fixed by the architecture, so AI scales inversely with bytes
per element: FP32 -> INT8 divides DRAM traffic by 4 and multiplies AI by 4.
Re-run with `--bytes-per-elem 1` for the INT8 roofline.
