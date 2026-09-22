# M2 + M3 — INT8 MAC Tile, Pipelining, Banded Scratchpad, Fused Write-Back

RTL and verification for the keyword-spotting accelerator's compute tile.
Design rationale lives in [`../ACCELERATOR_PLAN.md`](../ACCELERATOR_PLAN.md);
this file covers what the hardware is and how to run it.

**Status: 99/99 tests passing** (`sim/final_run.log`), structural lint clean.

## Layout

```
rtl/
├── rtl_design/          synthesizable RTL
│   ├── pe_int8.sv           one INT8 MAC cell
│   ├── mac_array.sv         K x M weight-stationary systolic array
│   ├── axis_result_fifo.sv  elastic buffer for the result stream
│   ├── band_sram.sv         M3: row-banded scratchpad + 3x3 window extractor
│   ├── writeback.sv         M3: requantise + BatchNorm + ReLU + 2x2 max-pool
│   └── tile_top.sv          AXI4-Lite + AXI4-Stream wrapper
├── tb/                  cocotb testbenches
│   ├── ref_model.py         pure-Python INT8 golden model
│   ├── test_pe.py           6 tests
│   ├── test_array.py        7 tests, run at PIPE=1 and PIPE=2
│   ├── test_band_sram.py    6 tests  (M3)
│   ├── test_writeback.py    8 tests  (M3)
│   └── test_top.py          10 tests, run at PIPE=1 and PIPE=2
└── sim/                 runners and logs
    ├── run_all.py           full regression -> final_run.log
    ├── run_pe.py  run_array.py  run_top.py  run_band.py
    ├── tile_schedule.py     array utilisation from the real tiling loop
    ├── speedup.py           end-to-end system speedup
    ├── lint_rtl.py          structural lint for synthesis and P&R
    ├── sram_sizing.py       reproduces SRAM_SIZING.md
    └── final_run.log
```

## What the tile computes

One streaming matrix-vector product per cycle:

```
Y[m] = sum over k of  W[m][k] * X[k]        m = 0 .. M-1
```

which is the im2col GEMM inner loop for convolution, with `M` = output
channels, `K` = input channels x 9, and `N` = output pixels streaming through.
Defaults are K = M = 16, the design point from the plan's sizing section.

Present an aligned K-vector on any cycle, get the aligned M-vector back
`PIPE*K + M - 1` cycles later, one result per cycle, no stalls.

### Why channel tiling, not spatial

The reference ECE 410/510 anemia accelerator broadcast one 3x3 kernel across
1024 spatial tiles. That only works for single-channel convolution. Every
convolution here is multi-channel (32→64, 64→128, 128→128) and needs
accumulation across input channels, which that structure has no path for — the
reference report flags this in its own section 8.5. Tiling over output channel
x input channel puts the reduction on the array's vertical partial-sum chain.
conv2, conv3 and conv4, which are 99.3% of all MACs, then tile 16x16 exactly.

### The systolic skew

A value hops one COLUMN per cycle but one ROW per `PIPE` cycles. Activation
`X[k]` reaches `PE(k,m)` at cycle `inject_k + m`, while a partial sum started at
row 0 reaches row k at `start + PIPE*k`. Those coincide for every k only if row
k is injected `PIPE*k` cycles late. `mac_array.sv` therefore carries:

- **input skew**, row k delayed by `PIPE*k` cycles
- **switch skew**, row k's weight commit delayed by `PIPE*k` cycles
- **output deskew**, column m delayed by `M-1-m` cycles

All three are internal, so the external interface is fully aligned. The switch
skew matters as much as the data skew: a commit must travel down the array at
exactly the speed of the data it must not overtake, or diagonals still in flight
finish with the wrong tile's weights.

At K = M = 16 the deskew costs 3,840 flops, the input skew 960 and the switch
skew 120. The deskew is 42% of the array's registers and is the first thing to
revisit if synthesis reports the array as register-bound — draining columns
sequentially trades those flops for M cycles per burst.

### M3: pipeline depth and overlapped weight loads

`PIPE` selects the psum pipeline depth. `PIPE=1` fuses multiply and accumulate
into one cycle; `PIPE=2` registers the product first, halving the combinational
path for timing closure at the cost of one cycle per row. The array's skew
follows it: row k is delayed by `PIPE*k`, and latency is `PIPE*K + M - 1`. Both
depths are in the regression, because getting that coupling wrong silently
computes the wrong dot product.

Each PE now holds a **shadow** weight register as well as an active one. The
shift chain writes the shadow, so the next tile loads while the current one
streams and the K-cycle load bubble disappears. The commit is skewed down the
array at exactly the speed of the data, so diagonals still in flight finish
with the old tile's weights. `sim/tile_schedule.py` measures the payoff:

| | Utilisation | Latency |
| --- | --- | --- |
| Blocking loads (M2) | 93.9% | 1.550 ms |
| Overlapped loads (M3) | 99.5% | 1.462 ms |

### No per-PE FSM

`pe_int8` is a multiply, an add, and registers. The reference design ran a
4-state FSM per tile and spent 2 of every 4 cycles in LOAD and DONE_ST, giving
50% pipeline utilisation. This array sustains one MAC per PE per cycle.

## Register map (AXI4-Lite)

| Address | Access | Contents |
| --- | --- | --- |
| `0x0000` | W | CTRL — `[0]` load + auto-commit (blocking), `[1]` load only (non-blocking), `[2]` commit shadow to active. All self-clearing. |
| `0x0004` | R | STATUS — `[0]` w_loaded `[1]` w_busy `[2]` fifo_empty `[3]` fifo_full `[4]` overflow |
| `0x0008` | R | ID — `0x4B575301` |
| `0x000C` | R | CFG — `[7:0]` K, `[15:8]` M, `[23:16]` ACC_W |
| `0x1000+` | R/W | weight staging, K*M bytes, index `m*K + k` |
| `0x2000+` | R | last result vector, M words (debug; the stream is the data path) |

Unmapped writes return SLVERR rather than silently succeeding.

## Operating sequence

1. Write `K*M` weight bytes to `0x1000+`.
2. **Simple path:** write `CTRL[0]`. A K-cycle FSM shifts the tile in, all M
   columns in parallel, then auto-commits. `s_axis_tready` stays low throughout,
   so the host cannot stream into a half-loaded array. Poll STATUS bit 0.

   **Overlapped path:** write `CTRL[1]` to shift into the shadow registers
   without stalling the stream, keep streaming the current tile, then poll
   STATUS bit 0 and write `CTRL[2]` to commit. A commit issued before the shift
   finishes is deferred rather than splicing two tiles together.
3. Stream activation vectors on `s_axis` (K bytes per beat).
4. Read result vectors from `m_axis` (M x 4 bytes per beat). `tlast` mirrors
   the input framing: send N vectors as one frame, get N results as one frame.

Results pass through a 16-deep FIFO because the array cannot stall and AXI
consumers can. If a consumer stalls long enough to fill it, data is genuinely
lost and STATUS bit 4 latches high — reported rather than hidden, since a
silent drop would look exactly like an arithmetic bug.

## Running

```bash
pip install cocotb cocotbext-axi        # Icarus Verilog must be on PATH
python rtl/sim/run_all.py               # full regression
python rtl/sim/run_top.py 16 16         # just the AXI suite at the design point
```

Tested with Icarus Verilog 11.0, cocotb 2.0.1, cocotbext-axi 0.1.28.

Note for cocotb 2.x: scalar signals read back as `Logic` and vectors as
`LogicArray`, so only vectors have `to_unsigned()`. `cocotb.result.TestSuccess`
is gone; a runtime skip is an early return. Both bit the first drafts here.

## Verification approach

Every test compares against `tb/ref_model.py`, an independently written pure
Python model. A bug has to appear in both to slip through. Beyond the happy
path the suite covers the INT8 `-128` asymmetry, which catches sign-extension
bugs nothing else does; gaps in the input stream, which catch a mistracked
valid pipeline; result-stream back-pressure; weight reload between bursts, the
inner loop of real tiling (conv3 alone reloads 288 times per layer); and real
conv2 weights from the trained checkpoint, quantised to INT8, which runs
automatically when `results/checkpoints/baseline_best.pt` is present and skips
with a logged warning otherwise.

## M3 banded scratchpad

`band_sram.sv` holds three rows of a feature map for many channels, double
buffered, and emits a 3x3 im2col window per output column as it sweeps. Holding
whole feature maps would need 378.8 KiB; banding by rows needs **44.4 KiB**, an
8.5x reduction. Full derivation in [`SRAM_SIZING.md`](SRAM_SIZING.md).

Consecutive output columns share two of their three taps, so the window is kept
in registers and shifted: 9 byte reads per cycle instead of 27. Columns outside
the row read as zero, which is what `padding=1` means for every convolution in
this model.

## The fused write-back path

`writeback.sv` is the module the project turns on. Profiling found max-pool at
38% of host runtime while contributing essentially none of the MACs, so an
accelerator that takes only convolution is capped by Amdahl at 2.24x no matter
how large the array. Folding pooling, BatchNorm and ReLU into the array's
output path raises the ceiling to 44.3x.

The pipeline is requantise, then ReLU, then 2x2 max-pool. BatchNorm costs no
hardware at all: at inference it is affine with constants known after training,
so folded into the convolution it becomes the per-channel scale and offset that
INT8 requantisation needs anyway.

Requantise, ReLU and max are all monotonic and therefore commute, so pooling
first would be numerically identical. Requantising first was chosen because the
pooling line buffer then holds INT8 rather than INT32 and is 4x smaller, and the
requantisers stay fully utilised instead of idle 75% of the time.

Pooling floors the way `torch.nn.MaxPool2d(2)` does: a trailing odd column or
row is dropped, turning conv2's 40x101 into 20x50. Getting that wrong shifts
every downstream feature by a pixel, which shows up as a quiet accuracy loss,
so the suite asserts it directly on odd geometries.

`python rtl/sim/speedup.py`:

| Configuration | Accel | Host | Total | Kernel | System |
| --- | --- | --- | --- | --- | --- |
| conv only, overlapped loads | 1.462 ms | 3.712 ms | 5.175 ms | 3.15x | **1.61x** |
| fused, overlapped loads | 1.462 ms | 0.188 ms | 1.650 ms | 5.56x | **5.04x** |
| fused + structured pruning | 0.718 ms | 0.188 ms | 0.906 ms | 11.32x | **9.18x** |

**Fusion alone is worth 3.14x on system speedup** and costs no extra cycles: the
write-back is a pipeline on the output path, adding latency but not reducing
throughput. As a side effect, 2x2 pooling cuts the result stream to a quarter of
its unpooled rate, reducing the interface bandwidth the chiplet needs.

## Synthesis and P&R readiness

`python rtl/sim/lint_rtl.py` checks the structural failure modes that survive
simulation and only bite at place-and-route:

| | Check |
| --- | --- |
| L1 | every instance port connected — an unconnected **input** floats |
| L2 | no combinational block can infer a latch |
| L3 | no declared net is left undriven |
| L4 | no net has multiple continuous drivers |
| L6 | every file guards with `` `default_nettype none `` |

Currently clean on all six modules. Two further guards:

- Every file sets `` `default_nettype none ``, so a mistyped signal name is a
  compile error instead of silently becoming an undeclared 1-bit wire. That is
  the single most common way nets end up floating.
- `test_top.test_no_floating_signals` sweeps the elaborated hierarchy after a
  full weight load and activation stream and fails on any signal reading X or Z.
  Undriven nets simulate as X, so this is direct evidence rather than a promise.

## Known limitations, carried forward

- **Weight staging is flops with M combinational read ports.** Fine at 256
  bytes, wrong at scale. M3 replaces it with per-column SRAM, alongside the
  activation scratchpad.
- **Which PIPE to ship is a synthesis question.** Both depths are verified;
  only Genus can say whether `PIPE=1` closes 500 MHz.
- **band_sram is not yet wired into tile_top.** It is verified standalone. The
  sequencer that walks the tiling schedule and connects the two is the next
  step, and it is also what will generate real `tlast`.
- **Register arrays stand in for SRAM macros.** Async reads today, synchronous
  on SAED14nm; the window absorbs the extra cycle by priming one earlier.
- **No pooling, BatchNorm or ReLU yet.** These are the M3 fused write-back
  path, and they are the whole point of the project: max-pool is 34–38% of host
  runtime at ~0% of the MACs, and fusing them moves the Amdahl ceiling from
  2.24x to 44.3x.
- **No tile sequencer.** The host currently drives every tile. M3 adds the
  sequencer that walks the M x K tiling schedule and generates real `tlast`.
