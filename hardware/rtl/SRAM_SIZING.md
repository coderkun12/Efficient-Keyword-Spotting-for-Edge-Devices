# On-Chip Memory Sizing — M3

Exit criterion for Task #6 in [`../ACCELERATOR_PLAN.md`](../ACCELERATOR_PLAN.md):
the banding scheme and the SRAM it implies. Numbers are reproducible with
`python hardware/rtl/sim/sram_sizing.py`.

## The trap we are avoiding

Holding whole feature maps on chip needs the largest live pair:

| Layer | Input map | Output map |
| --- | --- | --- |
| conv1 | 4,040 B | 129,280 B |
| conv2 | 129,280 B | 258,560 B |
| conv3 | 64,000 B | 128,000 B |
| conv4 | 32,000 B | 32,000 B |

The worst pair is conv1's output plus conv2's output: **387,840 B = 378.8 KiB**.
On an always-on part that is not acceptable. SRAM would dominate both area and
leakage, and leakage is what drains the battery between wake words.

Copying the reference ECE 410/510 accelerator would hurt most here. Its 34x34
scratchpad holds one tile of a single-channel image and does not generalise to
multi-channel feature maps at all.

## The banding scheme

A 3x3 convolution only ever needs **three input rows** live at once. Band the
feature map by rows and stream the bands, double-buffered so the next band
loads while the current one computes.

Per-row band sizes, which come out strikingly balanced across the three heavy
layers — a good sign the banding matches this model's shape:

| Layer | Input band per row | Output band per row |
| --- | --- | --- |
| conv1 | 101 B | 3,232 B |
| conv2 | **3,232 B** | **6,464 B** |
| conv3 | 3,200 B | 6,400 B |
| conv4 | 3,200 B | 3,200 B |

## Result

| Buffer | Sizing | Bytes |
| --- | --- | --- |
| Input band | 3,232 x 3 rows x 2 banks | 19,392 |
| Output band | 6,464 x 2 rows x 2 banks | 25,856 |
| Weight staging | 16 x 16 INT8 | 256 |
| **Total** | | **45,504 B = 44.4 KiB** |

**44.4 KiB against 378.8 KiB, an 8.5x reduction.**

The output band is two rows because a 2x2 max-pool consumes exactly two output
rows. That is not a coincidence to be grateful for, it is the reason the
pooling fusion is cheap: the rows pooling needs are already resident, so fusing
pool, BatchNorm and ReLU into the write-back path costs almost no extra memory.
That fusion is what moves the Amdahl ceiling from 2.24x to 44.3x.

Weight staging came out at 256 B rather than the 512 B the plan projected,
because M3 moved the double buffer into the PEs themselves as shadow registers
rather than keeping two staging copies in SRAM.

## Registers, which are not SRAM but are not free

| Structure | Flops | Equivalent |
| --- | --- | --- |
| Array input skew | 960 | 120 B |
| Array switch skew | 120 | 15 B |
| Array output deskew | 3,840 | 480 B |
| PE shadow + active weights | 4,096 | 512 B |
| Band window | 216 | 27 B |
| **Total** | **9,232** | **1,154 B** |

The **output deskew dominates** at 3,840 flops, roughly 2.5% of the array's
estimated 153,600 cells. It exists so all M columns present their results on
the same cycle. Draining columns sequentially instead would cost M extra cycles
per burst, which is negligible against bursts of 250 to 4,040 vectors. This is
the first thing to revisit if synthesis reports the array as register-bound.

## Carried into synthesis

The three row banks are modelled as register arrays with **asynchronous** reads,
which is right for simulation and for a small FPGA mapping. On SAED14nm they
become SRAM macros with **synchronous** reads, adding one cycle to the load
path. The sliding window absorbs that without changing external timing: prime
one cycle earlier. That swap is a synthesis task, not a functional change, and
the cocotb suite will not notice it.
