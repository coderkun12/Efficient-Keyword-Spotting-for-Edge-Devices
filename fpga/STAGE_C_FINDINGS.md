# Stage C findings — full layer on the DE2i-150

**Status:** C1–C4c complete. The harness found a **pre-existing RTL bug in
`mac_array`**, now diagnosed and fixed. Full layer passes in simulation:
60/60 outputs, zero mismatches. Ready for Quartus (C4d).

Stage A proved the systolic array on silicon. Stage B ran the trained conv2
weights through it. Stage C targets the **complete fused datapath** —
scratchpad, array, k-tile accumulation, fused BatchNorm + ReLU + max-pool —
as one layer on the FPGA.

---

## 1. Headline result: the layer fits

`layer_top` at the FPGA configuration (`MAXW=32`, `MAX_KTILES=72`,
`BAND_DEPTH=3232`) on the EP4CGX150DF31C7:

| resource | used | available | % |
|---|---|---|---|
| logic elements | 63,673 | 149,760 | **42.5%** |
| registers | 35,209 | 149,760 | 23.5% |
| memory bits | 851,968 | 6,635,520 | 12.8% |
| 9-bit multipliers | 322 | 720 | 44.7% |

All 25 memories map to real M9K block RAM: 9 for `band_sram`, 16 for the
weight lanes. This is a genuine feasibility result and stands independently
of whether the layer runs correctly today.

**Correction to an earlier figure.** The multiplier budget was previously
quoted as 256/360 = 71%. Cyclone IV counts multipliers in **9-bit elements**
and the device has **720**, not 360. The 16×16 array's 8×8 products each fit
one 9-bit element, so the honest figure is ~36% for the array alone. That
matters for array sizing: 24×24 (576 PEs) would fit; 32×32 (1024) would not.

---

## 2. Bugs that only synthesis could find

`writeback` had never been synthesised before stage C. Two real defects
surfaced on its first pass.

### 2.1 Inferred latch on a loop counter

```
Warning (10240): writeback.sv(105): inferring latch(es) for variable "ci"
```

`integer ci` was declared at module scope and used as a loop counter. Its only
loop is in the reset branch; the `cfg_we` branch never touches it, so it stays
live on that path. That is state, and synthesis built a register for a
variable meant to vanish at elaboration.

The instructive contrast is `m1`, two declarations away: also module-scope, but
it loops in **both** branches, so it is dead at the end of each and drew no
warning. **The rule is not "module-scope integers are bad" — it is "a variable
live across a branch is state."**

Fixed by declaring in the loop: `for (int ci = 0; ...)`.

### 2.2 Constant overflow in the saturation clamp

```
Warning (10259): writeback.sv(173): constant value overflow
```

`-8'sd128`. Eight-bit signed spans −128…127, so `8'sd128` overflows to −128,
and negating *that* overflows again back to −128. It reached the correct value
by two wrongs cancelling. Now `8'sh80`, the bit pattern stated directly.

### 2.3 Lint gap this exposed, and a third instance

Lint check L2 only inspects **combinational** blocks, so a latch in a clocked
one was invisible to it. Added **L9**, validated three ways:

- clean on the fixed code
- re-flags the original `ci` bug when reverted
- **found a third instance** — `axis_result_fifo.sv`, identical shape, not in
  the current build so Quartus had never seen it. Fixed before it could bite.

Also fixed a false positive in **L3**: the index class `[^;]` matched
newlines, so a match could run past its own line, swallow the next statement,
and credit the assignment to the wrong signal. A lint that cries wolf on
correct code teaches people to skim past it.

---

## 3. The memory-inference problem

The largest time sink of stage C, and worth recording because none of it is
reconstructable from the code.

### 3.1 `band_sram` — solved

Two changes were needed for block RAM:

- **synchronous read.** The window now comes from a registered read stage.
  This costs one cycle, absorbed by priming for three cycles instead of two.
  The window must also be **held still** on the first priming cycle, or stale
  bytes march into tap 0 — exactly the padding position `x=0` reads.
- **3× replication.** Each channel tap reads `rd_base + c*ch_stride + xc`:
  three addresses `ch_stride` apart, so not one wide word, and no block RAM
  has three read ports. Replication gives each copy one read and one write
  port. Cost ~590 Kb, ~9% of M9K.

External timing is unchanged — `win_vld`, `win` and `win_x` still arrive
together — which is why `layer_top` needed no modification.

### 3.2 `wmem` — four failed fixes before the answer

Four plausible changes each cost a ~20-minute synthesis run and changed
**nothing**:

| attempt | result |
|---|---|
| split the `M*8`-bit array into M byte-wide lanes | no change |
| replace the computed write address with a bit-slice | no change |
| add `(* ramstyle = "M9K" *)` | **silently ignored**, no error |
| round the depth 1152 → 2048 | no change; **+114,688 registers** |

Quartus emitted **no diagnostic at all** — no "uninferred" message. The only
symptom was 147 Kb quietly becoming registers, putting the design 357% over
the device.

**What found it:** a probe synthesising four structural variants side by side
(`fpga/memcheck/wmem_only/`). One run, four independent answers.

**The answer:** a reg array inside a **generate block** does not infer as
block RAM here. An array at **module scope** does. The fix is
[`byte_ram.sv`](../rtl/rtl_design/byte_ram.sv) — one lane per module,
instantiated M times. A generate containing module *instances* is not the same
thing as a generate containing array *declarations*.

Result: **8.4× fewer logic elements**, 535,025 → 63,673.

`byte_ram.sv` also carries a direct `altsyncram` instantiation behind
`` `define USE_ALTSYNCRAM ``, unused, as a guaranteed escape hatch — that path
cannot fail because it is not inference.

### 3.3 The method lesson

Three of the four `wmem` fixes were guesses. The probe that actually answered
it took 13 minutes. **When a tool fails silently and each iteration is
expensive, building the discriminating experiment is cheaper than another
plausible hypothesis** — a lesson already available from `band_sram`, where
the same technique resolved it in one minute.

---

## 4. The C4 harness

[`de2i150_layer_top.sv`](de2i150_layer_top.sv) drives `layer_top` through the
sequence lifted from `test_layer_top.py`, with vectors from
[`gen_layer_vectors.py`](gen_layer_vectors.py). Golden data comes from
`conv_layer()` and `fused_writeback()` in `rtl/tb/ref_model.py` — the same
functions the cocotb suite checks `layer_top` against, so there is no second
model to disagree with the first.

Target: **conv4 m-tile 0** — 128→16 channels, W=25, H=10, 72 k-tiles, fused
BN+ReLU+2×2 pool, 60 pooled output vectors. ROM cost 411,936 bits (6.2% of
M9K). Runs in 141,419 cycles.

Two harness bugs found in simulation:

- **Extra pipeline stage.** ROM addresses were registers fed by non-blocking
  assignment, inserting a second stage, so every byte landed one address late.
  Symptom: 60/60 mismatches with perfect sequencing, timing and result count
  around them. Fixed by making addresses and write ports combinational off
  one-cycle-delayed copies. → 15/60.
- **No settle gap** after `busy`. `busy` going low does not mean the row is
  finished; the write-back pipeline is still draining.

---

## 5. THE BUG — pre-existing, in `mac_array`, now FIXED

The remaining mismatches are **not** the harness. The verified cocotb driver
reproduces them at the same index, and — critically — **so does the committed
RTL from before any stage C change**.

### 5.1 It was already there

Extracted the pre-stage-C sources from git into a temp directory and ran the
probes against them, working tree untouched:

```
test_ktile_accumulation   PASS    (2 k-tiles, W=6)
test_col18_1ktile         PASS    (1 k-tile,  W=20)
test_col18_2ktile         FAIL    (2 k-tiles, W=20)
test_col18_4ktile         FAIL    (4 k-tiles, W=20)
```

The C1/C2 memory-inference work did not introduce it. It has been in the
design since it was written.

### 5.2 Characterisation

**Needs a k-tile handover.** One sweep is always correct:

| k-tiles | W | result |
|---|---|---|
| 1 | 20 | **PASS** |
| 2 | 20 | FAIL, output 19 |
| 4 | 20 | FAIL, output 18 |
| 18 | 20 | FAIL, output 18 |

**Width matters, but not as a threshold.** At 2 k-tiles:

| W | 6 | 8 | 12 | 16 | 17 | 18 | 19 | **20** | 25 |
|---|---|---|---|---|---|---|---|---|---|
| | pass | pass | pass | pass | pass | pass | pass | **FAIL** | FAIL |

W=19 reaches column 18 and passes; W=20 fails. So it is not "column 18".

**Deterministic, not data-dependent.** Five different seeds at 2 k-tiles /
W=20 all fail. Failing indices cluster at the **end of output rows**
(19, 19, 19, 79, 18 for H=4, W=20 — 19 and 79 are last-of-row).

**Fails with pooling disabled**, so the fault is in the convolution / k-tile
accumulation path, *before* the write-back. Pooling and the line buffer are
ruled out.

**The error is small.** One channel, one LSB-ish:

```
got  [0, 0, 2, 6, 0, 0, 0, 0, 14, 0, 0, 80, 11, 0, 2, 0]
want [0, 0, 2, 6, 0, 0, 0, 0, 14, 0, 0, 80, 11, 0, 4, 0]
                                                  ^^^
```

With 2 k-tiles the second tile carries only 2 real taps, so a missing or
mistimed final-tile contribution on the last columns would look exactly like
this.

### 5.3 Ruled out

- **`W >= 2K` handover stall** — W=20 fails, W=16 passes, both below 2K=32.
- **Band channel padding** — the last k-tile's window reads two channels past
  the layer's last, and `test_layer_top.py` filled two too few, leaving X in
  the band. Fixed (`+2` in `win_channels`); a genuine latent defect, but not
  this one.
- **`W > MAXW`** — W=40 with MAXW=32 fails because `rowacc` has only MAXW
  entries. Expected, and a different issue.
- **Accumulator overflow** — conv4's deepest reduction is 1152 × 127 × 127 =
  18.5M against ACC_W=28 (134M). Four times the headroom.
- **Data dependence** — five seeds, all fail.

### 5.4 Root cause: the weight commit was skewed in one dimension, not two

Probing `rowacc` — writeback's actual input, before requantisation — gave a
clean diagonal:

| flush_x | channels wrong |
|---|---|
| 18 | **15** |
| 19 | **14, 15** |

That is the array's **column** dimension, and it names the fault exactly.

In `mac_array`, an activation presented at time *t* reaches `PE(k,m)` at

```
t + PIPE*k + m
```

`PIPE*k` for the input skew, plus `m` because the value hops one **column**
per cycle. The commit was delayed by `PIPE*k` only, so every PE in row *k*
switched at `c + PIPE*k`. The `PIPE*k` terms cancel and `PE(k,m)` keeps the
old weights only while `m < c − t`.

Solving that against the observed diagonal gives **c − t = 14**, where
**M = 16** is required. **The commit landed exactly two columns early**, and
the last two columns switched weights underneath data still travelling
through them.

Everything else follows: one k-tile has no commit and always passed; W ≤ 16
finished before the effect was reachable; and every pre-existing test used
W ≤ 8, so none could see it.

### 5.5 The fix — two halves, both required

**`mac_array`:** the commit now travels the way the data does — down one row
per `PIPE` cycles **and right one column per cycle**, so `PE(k,m)` switches at
`c + PIPE*k + m`. The condition becomes `t >= c`, independent of both *k* and
*m*. Cost: `K*(M-1)` = 240 one-bit flops, no throughput loss.

**`layer_top`:** `hold`, the number of cycles the shadow registers must stay
untouched after a commit, was `K`. That was correct only while the commit was
row-skewed. The last PE to copy shadow into active is now `PE(K-1, M-1)` at
`commit + PIPE*(K-1) + (M-1)`, so `hold` becomes `PIPE*(K-1) + M` = 31.

Changing one without the other is worse than changing neither: with the
column skew but `hold = K`, the next tile's shift overwrites shadows the far
corner has not read yet, and **every** output is wrong rather than the last
two columns. That intermediate state was observed — 80/80 accumulator values
wrong — and is why both halves are documented together.

### 5.6 Verification

| | |
|---|---|
| small-config regression | **129/129 pass** |
| full FPGA config | **30/30 pass** |
| lint | clean |
| C4 layer harness | **60/60 outputs, 0 mismatches** |

The nine probes that found it stay in the suite, guarded by `build_fits()` so
they skip rather than fail on the small build.

### 5.7 Reproducer, for the record

Two seconds in cocotb:

```
test_col18_2ktile       # 2 channels, H=4, W=20, pooling off
test_probe_rowacc       # dumps rowacc vs conv_layer(), shows the diagonal
```

## 6. Where to look next

The failure is reproducible in cocotb in ~20 seconds, with full signal
visibility — seconds per iteration instead of 20-minute Quartus runs.

Suspects, in order:

1. **`S_DRAIN` timing.** It waits `LATENCY + 2` = 33 cycles from the last
   window of the last k-tile before flushing `rowacc`. If a late result lands
   after `flush_x` has already passed its column, that column keeps a partial
   sum — which is precisely "the tail of the row is short by the last tile's
   contribution". Instrument `r_vld` against `state` to see whether any result
   arrives during `S_FLUSH`.
2. **`oc` / `rkt` derivation.** `rkt` increments every `cfg_width` results,
   so the tile identity is inferred from the result *count*. Any extra or
   missing `r_vld` pulse across a handover shifts every subsequent result into
   the wrong tile and the wrong column. Count `r_vld` pulses per sweep and
   confirm it is exactly `cfg_width`.
3. **The `S_WWAIT` path.** With W < 2K the next tile's weights are not ready
   when the sweep ends, so the FSM detours through `S_WWAIT` instead of
   swapping in place. W=20 takes that path; whether W=17..19 also do is worth
   checking, since they pass.

**Do not build this in Quartus yet.** It would faithfully reproduce the same
15 mismatches and cost 20 minutes to learn nothing new.

---

## 7. What stage C has produced

- The full fused layer **fits** the EP4CGX150 at 42.5% logic, 12.8% memory
- `band_sram` and the weight memory both map to **real block RAM**
- Two RTL bugs fixed that only synthesis could find
- A third latent latch caught by a new lint check before synthesis saw it
- **A real datapath bug found by the layer-level self-test**, invisible to 107
  passing unit tests and to 14 nm synthesis
- A characterised, reproducible failure with the search space narrowed to the
  accumulate path and a 4-wide window on the triggering parameter

The last two are the ones worth writing up. A self-test that finds a defect
the unit tests could not is the argument for building it.
