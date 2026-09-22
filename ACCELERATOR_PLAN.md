# KWS Inference Accelerator — Design Plan

**Target:** INT8 systolic-array chiplet for `KeywordSpottingCNN`
**Scope:** inference only. Training stays on the host, offline.
**Nature:** HW/SW co-design research. The output is a design justified by
measured workload behaviour, taken through RTL, synthesis, and place-and-route.
**Reference process:** the milestone structure from the ECE 410/510 anemia
Conv2d accelerator (`D:\PSU\Q3\MY_SUBMISSION`), adapted to this model.

**P&R must close.** This is a hard constraint, not an aspiration. The reference
project's 32x32 BF16 array reached 5.59M cells and Innovus **could not close
timing at 500 MHz** — their `run_innovus.tcl` is committed with that outcome
recorded. Every sizing decision below is made against that documented failure.

---

## 0. Where we already are

Task #1 (profiling and roofline) is complete. Everything below builds on it.

| Quantity | Value | Source |
| --- | --- | --- |
| Model | 242,508 params, 4 conv blocks | `Profiling/eda_profiling.txt` |
| Work per inference | 186,770,892 MACs (373.5 MFLOP) | `Profiling/kws_analysis.md` |
| Convolution share of MACs | 99.3% | `Profiling/kernel_analysis.md` |
| Convolution share of host runtime | 55.38% | `Profiling/op_breakdown.txt` |
| Conv + pool + BN + ReLU share | 97.74% | `Profiling/op_breakdown.txt` |
| Host baseline | 8.32 ms, batch 1, single thread | `Profiling/host_vitals.txt` |
| Reference host | i5-12450H, 140.8 GFLOP/s per core, 51.2 GB/s | `Profiling/host_vitals.txt` |
| Chiplet design point | 16x16 INT8 @ 500 MHz = 256 GOP/s | `Profiling/roofline_chiplet.png` |
| Projected | 1,459 us/inference, 5.7x kernel, 5.2x system | same |

All host numbers come from `LAPTOP-FFRP12TK`, the machine that produced the
trained checkpoints. They were re-measured after the first baseline turned out
to be a mean of noisy samples (17.072 +/- 6.757 ms, 40% relative deviation)
taken after only 0.2 s of warmup. The 8.32 ms figure is a median over 800
iterations after a 5 s warmup, with a p90/best spread of 1.3x.

Runtime shares were confirmed independently on a second machine (i5-10210U,
Comet Lake): convolution 54.49%, max-pool 34.36%, feature total 97.22%. Two
different microarchitectures agree, so the bottleneck is a property of the
model, not of one laptop.

Two results from profiling drive every decision below:

1. **The MAC load is flat.** conv2 40.0%, conv3 39.5%, conv4 19.8%. There is no
   single hot layer to specialise for. The array must run all three well.
2. **Max-pool is 38.1% of host runtime at ~0% of the MACs.** It is pure data
   movement, and this model has three pooling stages. This is the single
   biggest differentiator between our design and the reference project's.

---

## 1. Four decisions to lock now and never revisit

The reference project lost real time to churn. Its precision changed three times
(Q16.16 → FP16 → BF16), and a synthesis run at CF07 discovered a 21.65 ns
critical path against a 2 ns target, forcing a written re-scope. Its own
"what would be done differently" section says to start on the commercial PDK.
We can skip all of that because we already have the evidence.

| Decision | Choice | Why it is already settled |
| --- | --- | --- |
| Precision | INT8, INT32 accumulate | QAT measures **93.49%**, above the 91.88% FP32 baseline. Not a projection. |
| PDK and tool | SAED14nm RVT, Cadence Genus | sky130A failed to close a 1024-tile design at 130 nm. Their documented advice. |
| Dataflow | Weight-stationary, tiled over output x input channel | Our convolutions need channel accumulation. See §4. |
| Partition | Conv + BN + ReLU + pool all on-chip | Conv alone caps the system at 2.24x. See §2. |

**Do not** copy the reference array's organisation. It broadcasts one 3x3 kernel
to 1024 spatial tiles, which only works for single-channel convolution. Ours are
32→64, 64→128 and 128→128 and need accumulation across input channels, which that
structure has no path for. Their own report flags this in section 8.5.

---

## 2. Task #2 — HW/SW partition rationale

**Deliverable:** `accelerator/m1/partition_rationale.md`

The argument writes itself from the profiler data, and it is a stronger argument
than the reference project's because of the pooling finding.

| Scope accelerated | Runtime share | Amdahl ceiling | Projected system speedup |
| --- | --- | --- | --- |
| Convolution only | 55.38% | 2.24x | 1.84x |
| Conv + pool + BN + ReLU | 97.74% | 44.3x | 5.15x |

**Goes to hardware:** all four conv layers, BatchNorm, ReLU, max-pool, global
average pool.

Two of those are nearly free and must not be separate hardware:

- **BatchNorm folds into the conv weights offline.** At inference, BN is an affine
  transform with fixed parameters; fold scale and shift into the weight and bias
  before quantisation. This removes 6.3% of runtime and all BN silicon. Standard,
  zero risk, do it in the host toolchain.
- **Max-pool folds into the output write path.** A 2x2 max over the accumulator
  output as it drains costs one comparator and a half-line register per output
  channel. This is what buys the jump from a 2.24x ceiling to a 44.3x ceiling.

**Stays in software:** mel-spectrogram feature extraction, the final
Linear(128, 12) classifier at 1,548 MACs, argmax, and all control. The classifier
is under one thousandth of a percent of the model; putting it in hardware would be
pure overhead, and keeping it on the host is also what makes speaker
personalisation possible later without a gradient datapath.

**conv1 is a judgement call.** With K = 9 it fills 9 of 16 rows of the array
(56% utilisation) but is only 0.7% of the MACs. Run it on the array with the
padding waste rather than adding a special case. Note the inefficiency in the
report and move on.

---

## 3. Task #3 — Interface selection

**Deliverable:** `accelerator/m1/interface_selection.md`

Here we should deliberately **not** copy the reference project. It selected
AXI4-Stream at 512-bit / 500 MHz for 32 GB/s against a 21.3 GB/s requirement.
Our requirement is three orders of magnitude smaller.

```
Required BW = chiplet peak / arithmetic intensity
            = 256 GOP/s / 1,515 FLOP/byte
            = 0.17 GB/s
```

| Interface | Rated | vs 0.17 GB/s | Verdict |
| --- | --- | --- | --- |
| SPI @ 50 MHz | 0.006 GB/s | 29x short | No |
| AXI4-Lite alone | 1-4 GB/s | sufficient on paper | No, no burst support for activation tiles |
| **AXI4-Stream 64-bit @ 500 MHz** | **4 GB/s** | **24x margin** | **Select this** |
| AXI4-Stream 512-bit @ 500 MHz | 32 GB/s | 188x margin | Over-specified, wastes pins and area |

Pair it with AXI4-Lite for control and result readback, as they did. The
justification to write up is that over-specifying the interface on an always-on
edge part costs pins, area and idle power for margin we provably do not need.
That contrast with the reference design is worth stating explicitly in the report.

**Deliverable also includes:** a bandwidth table showing the design is compute-bound
on the chiplet (AI 1,515 versus ridge 160), which `Profiling/roofline_chiplet.py`
already generates.

---

## 4. Task #4 — Single tile design

**Deliverable:** `accelerator/m2/` with RTL, cocotb testbenches, passing sim log.

**The PE.** One INT8 x INT8 multiply into an INT32 accumulator. Registered inputs,
registered output. That is the whole cell. The reference project's PE held nine
BF16 multipliers and a block-float adder tree, which is why 96.9% of its area and
92.6% of its power sat in the array. Ours should be small and boring.

**The mapping.** Convolution becomes a matrix multiply where M is output channels,
K is input channels times 9, and N is output pixels:

| Layer | M | K | N | MACs | Share | 16x16 tiles (M x K) |
| --- | --- | --- | --- | --- | --- | --- |
| conv1 | 32 | 9 | 4040 | 1,292,800 | 0.7% | 2 x 1 (56% row fill) |
| conv2 | 64 | 288 | 4040 | 74,723,840 | 40.0% | 4 x 18 |
| conv3 | 128 | 576 | 1000 | 73,856,000 | 39.5% | 8 x 36 |
| conv4 | 128 | 1152 | 250 | 36,896,000 | 19.8% | 8 x 72 |

Weights stay resident in the M x K grid. Activations stream through the N
dimension. Three of the four layers tile the array exactly, which is the payoff
for choosing channel tiling over the reference project's spatial tiling.

**Verification.** cocotb plus Icarus Verilog, same as theirs, with a pure-Python
INT8 reference model generating golden vectors. Their `ref_model.py` pattern is
worth copying directly. Test the single PE, then a 4x4 array, then a full tile
with real weights pulled from the trained checkpoint.

**Exit criterion for M2:** a single tile computes a correct INT8 dot product
against the Python reference, and the AXI handshake passes, both in cocotb.

---

## 5. Task #5 — Pipelining

**Deliverable:** `accelerator/m3/` RTL revision plus first synthesis run.

The reference design reached only **50% pipeline utilisation**: its PE spends two
of every four cycles on LOAD and DONE_ST states, doing no arithmetic. That is a
2x throughput loss baked into the architecture. Their section 8.5 lists it as a
known gap. We should not inherit it.

**Target:** one MAC per cell per cycle in steady state. A systolic array should
have no per-tile FSM at all in the inner loop. Weights load once per tile; after
that, data flows and results drain continuously.

**Expected critical path.** Their own INT8 analysis estimated 4.0 to 6.5 ns
unregistered on sky130 at 130 nm. On SAED14nm that should land near 1.0 to 1.5 ns,
so a 2-stage pipeline (multiply, then accumulate) should close 500 MHz with
margin. This is the single strongest reason INT8 was the right call: the entire
floating-point normalisation chain that blew their timing budget simply does not
exist.

**Also fix here:** their remaining-tasks note about a `MUX2` on the output select
landing 10 ps from the capture flop. Use an early-select pattern so the output mux
is never on the critical path.

**Exit criterion for M3:** Genus reports WNS >= 0 at 500 MHz on the single tile,
and the cocotb suite still passes.

---

## 6. Task #6 — On-chip memory

**Deliverable:** `accelerator/m3/` scratchpad RTL plus an SRAM sizing note.

This is where the real design work is, and where naively copying the reference
project would hurt most. Their design has a 34x34 scratchpad holding one tile.
Ours has to handle full feature maps, and the obvious approach does not fit.

**The trap.** Holding whole feature maps on-chip needs the largest live pair:

```
conv2 output 64 x 40 x 101 = 258,560 B
conv1 output 32 x 40 x 101 = 129,280 B
                             --------
                             387,840 B = 379 KiB at INT8
```

379 KiB of SRAM on a part meant to be small is not acceptable. SRAM would
dominate area and leakage, and leakage is what kills an always-on device.

**The fix: row-banded line buffers with double buffering.** A 3x3 convolution
needs only three input rows live at a time. Band the feature map by rows:

The input bands come out almost perfectly balanced across the three heavy layers,
which is a good sign the banding is the right shape for this model:

| Layer | Input band, 3 rows | Output band, 2 rows (pool needs 2) |
| --- | --- | --- |
| conv2 | 32 ch x 3 x 101 = 9,696 | 64 ch x 2 x 101 = 12,928 |
| conv3 | 64 ch x 3 x 50 = 9,600 | 128 ch x 2 x 50 = 12,800 |
| conv4 | 128 ch x 3 x 25 = 9,600 | 128 ch x 2 x 25 = 6,400 |

Sizing for the worst case and double buffering so load overlaps compute:

| Buffer | Bytes |
| --- | --- |
| Input band (worst case 9,696) x 2 | 19,392 |
| Output band (worst case 12,928) x 2 | 25,856 |
| Weight tile staging, 16 x 16 INT8 x 2 | 512 |
| **Total** | **45,760 B = 44.7 KiB** |

That is roughly **48 KiB** rather than 379 KiB, an 8x reduction, and it is what
makes the pooling fusion natural: a 2x2 max-pool consumes two output rows, which
is exactly what the output band already holds.

Weights stream per tile from the host. A 16x16 INT8 weight tile is 256 bytes, so
there is no reason to hold all 237 KiB of model weights on-chip.

**Exit criterion:** SRAM sizing note with the banding scheme, and synthesis
showing the memory does not dominate area.

---

## 7. Task #7 — Sparsity

**Deliverable:** `accelerator/m4/` sparsity support plus a measured accuracy table.

Our sparsity story is substantially stronger than the reference project's, and it
is worth being explicit about why. Their plan was zero-gating the multiplier to
save switching power, with the zeros still occupying bus beats and PE slots, so
throughput was unchanged. We already have measured pruning results:

| Variant | Accuracy | Params | MACs | CPU latency |
| --- | --- | --- | --- | --- |
| baseline | 91.88% | 242,508 | 186,222,336 | 8.32 ms |
| unstructured 50% | 93.99% | 242,508 | 186,222,336 | 12.585 ms |
| **structured 70% keep** | **92.82%** | **120,291** | **91,472,400** | **8.051 ms** |
| structured 70% + INT8 QAT | 93.41% | — | — | 2.903 ms* |

\* Latencies other than the baseline are still the noisy means from the first
benchmark run. Re-run `src/benchmark.py` on the reference host for medians.

**Structured channel pruning is the primary lever, and it needs no special
hardware at all.** It shrinks M and K directly, so the array simply runs fewer
tiles. It halves the MACs, accuracy goes *up* relative to baseline, and on the
chiplet it takes the projection from 1,459 us to roughly **715 us**.

Unstructured sparsity is the opposite. At 49.8% sparsity the MAC count is
unchanged, because the zeros still sit in the tensor. On a dense systolic array
that buys nothing in throughput. Implement zero-gating on the multiplier only as a
*power* optimisation, and label it as such rather than claiming a speedup.

**Exit criterion:** benchmark table comparing dense against structured-pruned on
the array, with accuracy from the existing checkpoints rather than projections.

---

## 8. Array sizing — driven by P&R feasibility

Array size is the dominant performance lever and it is linear in PEs. It is
also the thing that decides whether place-and-route closes. Both are settled
by the same table.

Cell counts scale from the reference project's actual SAED14nm synthesis
(0.512 um2/cell; 607 cells per BF16 MAC in-array). An INT8 PE is taken as
~600 cells: an 8x8 signed multiplier, an INT32 accumulate add and register,
and control. Overhead for buffers, interface and control adds 30%.

| Array | PEs | Cells | Area | vs. the P&R that failed | Kernel | System |
| --- | --- | --- | --- | --- | --- | --- |
| 8x8 | 64 | 50 K | 0.026 mm² | 112x smaller | 1.4x | 1.4x |
| **16x16** | **256** | **200 K** | **0.10 mm²** | **28x smaller** | **5.7x** | **5.2x** |
| 32x32 | 1024 | 799 K | 0.41 mm² | 7x smaller | 22.8x | 15.3x |
| 64x64 | 4096 | 3.2 M | 1.64 mm² | 2x smaller | 91.2x | 30.0x |

**Design point: 16x16.** It is 28x smaller than the design Innovus could not
close, which makes P&R a routine step rather than a research risk of its own.
**Stretch goal: 32x32**, attempted only after 16x16 has closed end to end.
64x64 sits within 2x of the documented failure and is out of scope.

Correcting an earlier estimate in this project: the array was previously put at
~0.010 mm², assuming an INT8 MAC roughly an order of magnitude smaller than a
BF16 one. That is wrong. A BF16 multiplier shares the same 8x8 mantissa
multiply core and adds exponent, normalise and round logic, so the real ratio
is nearer 2-4x. The corrected figure is ~0.10 mm², and SRAM sits on top of it.

### Where more speed comes from, in priority order

1. **Structured pruning — free, already validated.** The 70% keep variant runs
   91.47 M MACs instead of 186.77 M, a 2.04x work reduction, at 92.82%
   accuracy, *above* the 91.88% dense baseline. This doubles effective
   throughput and costs no silicon. It should be the default model.
2. **Array size.** Linear in PEs, bounded by P&R as above.
3. **Clock.** Linear, but dynamic power scales with it. Wrong direction for an
   always-on part, and 1 GHz is a timing risk the reference project hit at
   500 MHz with deeper logic.
4. **N:M structured sparsity.** A 2:4 pattern gives a guaranteed 2x and maps
   onto a systolic array. Unstructured sparsity does not: the 50% variant has
   49.8% zeros but identical MACs, and scattered zeros break array lockstep.

Pipelining is **not** on this list. Every throughput number here already assumes
one MAC per PE per cycle in steady state. Pipelining prevents losing that; the
reference PE spent 2 of every 4 cycles on load and drain, running at half rate.
Counting it as extra speedup would be double-counting.

---

## 9. Milestone mapping

| Milestone | Tasks | Deliverable |
| --- | --- | --- |
| M1 | #2, #3 | Partition rationale, interface selection, roofline (done) |
| M2 | #4 | Single-tile INT8 RTL, cocotb passing |
| M3 | #5, #6 | Pipelined array, banded SRAM, first Genus close at 500 MHz |
| M4 | #7 | Sparsity, full 16x16 synthesis **and P&R**, benchmark, report |

**Projected final numbers**, to be replaced by synthesis and P&R:

| Metric | Reference (BF16 32x32) | This design (INT8 16x16) |
| --- | --- | --- |
| Throughput | 2,304 GFLOP/s | 256 GOP/s |
| Die area | 2.86 mm² | ~0.10 mm² + SRAM (estimated) |
| Power | 1,359 mW | ~30-60 mW (estimated) |
| Latency | — | 1,459 us dense, 715 us pruned |
| Kernel speedup | 16.1x | 5.7x dense, 11.6x pruned |
| System speedup | 3.31x | 5.2x dense, 9.4x pruned |
| P&R at 500 MHz | **did not close** | must close; 28x smaller |

We give up an order of magnitude of raw throughput and gain roughly 28x in
area and 30x in power, while accelerating 97.74% of runtime instead of 74.4%.
For an always-on battery-powered keyword spotter that is the correct trade, and
it is the trade that makes P&R tractable.

---

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| Array utilisation below the assumed ~100% | Model it in M2 simulation before committing to the M4 array size. Budget 70-85%: conv1 fills only 9 of 32 rows, conv4's 250 output pixels tile unevenly, and every tile pays fill and drain |
| SRAM dominates area at 14 nm | Sizing note lands in M3, before the array is scaled |
| INT8 QAT accuracy drifts once BN is folded | Re-run QAT with BN folded during M1 and re-measure, do not assume |
| Chiplet must beat software INT8, not FP32 | INT8 QAT already reaches 5.60 ms on the host (mean; re-measure for a median). That, not 8.32 ms, is the honest number to beat |
| Amdahl ceiling | 44.3x at f = 97.74%. No array size exceeds it. Anything above that in a projection is an arithmetic error |

The fourth row deserves emphasis in the report. Against the FP32 baseline the
16x16 chiplet looks like a 5.7x win. Against what the host already does in INT8
software it is closer to 2x, and honest benchmarking should say so. The
defensible claim for this design is **energy per inference**, tens of
milliwatts against a 45 W laptop part, not latency against a modern CPU core.

## 11. The result worth publishing

The finding that justifies the whole project is not the speedup. It is that
**max-pool is 34-38% of host runtime while contributing ~0% of the MACs.**

A MAC-count analysis — the standard way accelerator targets get chosen, and what
the reference project did — misses this completely and would size a conv-only
accelerator. That design caps at 2.24x no matter how large the array. Fusing
pool, BatchNorm and ReLU into the convolution write-back path costs very little
hardware and moves the ceiling to 44.3x.

That gap between where the operations are and where the time goes, measured on
two independent microarchitectures, is the co-design contribution.
