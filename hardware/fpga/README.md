# FPGA bring-up — DE2i-150 (Cyclone IV GX EP4CGX150DF31C7)

Emulating the keyword-spotting ML kernel on the accelerator, on real hardware.

Per the framing set at the start of this milestone: **the FPGA run is emulation
of the ML kernel on the AI accelerator, and a couple of inferences — not a
performance result.** Performance numbers come from the 14 nm Genus synthesis
(1 GHz, 9.76 TOPS/W). Cyclone IV GX is a 60 nm-class part and will run roughly
an order of magnitude slower. What the board proves is *correctness after
place-and-route on silicon*, which simulation cannot prove.

---

## What you need

### 1. Software

| item | requirement | note |
|---|---|---|
| Quartus | must list **Cyclone IV GX** | see the version check below |
| Cyclone IV device support | installed with Quartus, or via Tools ▸ Install Devices | a Quartus install without the device family will not build |
| USB-Blaster driver | bundled with Quartus | Windows may need it pointed at `quartus/drivers/usb-blaster` manually |
| Python 3 + PyTorch | already in your environment | only for regenerating vectors |
| Icarus Verilog | already installed | for the pre-build harness check |

**Version check — do this first, it takes 30 seconds.** Open Quartus ▸ New
Project Wizard ▸ Family & Device. If **Cyclone IV GX** is not in the family
dropdown, or `EP4CGX150DF31C7` is not in the device list, this Quartus cannot
build the design and no amount of settings will change that.

Quartus **Prime Pro** does not support Cyclone IV at all. You need **Quartus
Prime Lite** (20.1 and earlier carry Cyclone IV; later Lite releases dropped
it) or the older **Quartus II 13.x**. If the wizard comes up empty, that is
the thing to fix before anything else.

### 2. Hardware

- The DE2i-150 board and its 12 V supply
- A USB cable to the **Blaster** port — the board has several USB connectors,
  and only the one marked for the on-board USB-Blaster programs the FPGA
- Nothing else. No PCIe host software, no SD card, no serial terminal.

### 3. The board pin assignments

**Already done — see [`de2i150_pins.qsf`](fpga/de2i150_pins.qsf).**

The 32 locations this design needs are transcribed from the *DE2i-150 FPGA
System User Manual v1.3*, Tables 3-3 (push-buttons), 3-4 (LEDs) and 3-6 (clock
inputs). Get them into the project with **Assignments ▸ Import Assignments** ▸
`de2i150_pins.qsf`.

You do not need the System CD or the golden top for this build. The manual is
a free direct download from the Terasic DE2i-150 page, Resources tab, no
registration.

Two things the manual settled that are easy to get wrong:

- **Mixed I/O banks.** `CLOCK_50` is a **3.3 V** input; every LED and
  push-button is **2.5 V**. One global setting cannot express that, so each
  pin carries its own `IO_STANDARD`.
- **Polarity.** LEDs light when driven **high**; `KEY` reads **high when
  released, low when pressed**, and is hardware-debounced by Schmitt triggers.
  Both match what the RTL already assumes, so no change was needed.

---

## Files here

| file | what it is |
|---|---|
| `de2i150_mac_top.sv` | the self-test: ROMs, sequencer FSM, checker, LED status |
| `tb_de2i150_mac_top.sv` | Icarus testbench for the harness itself |
| `gen_test_vectors.py` | writes the vector ROMs from `rtl/tb/ref_model.py` |
| `de2i150_mac_top.qsf` | Quartus settings — everything except pins |
| `de2i150_mac_top.sdc` | 50 MHz clock constraint |
| `de2i150_pins.qsf` | the 32 pin locations, from the user manual |
| `de2i150_mac_top.qpf` | project file — open this, do not run the wizard |
| `weights/acts/expected .mif` | ROM contents for Quartus |
| `weights/acts/expected .txt` | the same contents for `$readmemb` in simulation |

---

## What the design does

`mac_array` alone, with no memory subsystem and no host interface. That is the
right first target: `layer_top`'s four memories use asynchronous and
multi-ported reads, which Quartus cannot map onto M9K blocks and would infer
as LUT RAM — a large, slow build that tells you nothing about the array.

Weights, activations and golden results are compiled into on-chip ROMs. On
power-up the FSM runs:

```
POR  →  LOAD 16  →  COMMIT 1  →  SETTLE 16  →  STREAM 64  →  DRAIN 48  →  DONE
```

150 cycles of actual work, then it holds the verdict. `KEY[0]` restarts it.

The `SETTLE` state is not padding. The weight commit is skewed one row per
`PIPE` cycles, so row `K-1` does not hold the new tile until `commit + PIPE*(K-1)`.
Streaming activations before that mixes old and new weights into one result —
the bug that made k-tile 0 produce garbage during integration.

### Reading the LEDs

| LED | meaning |
|---|---|
| `LEDG[7]` | heartbeat, ~1.5 Hz — **check this first** |
| `LEDG[0]` | running |
| `LEDG[1]` | done |
| `LEDG[2]` | **PASS** — all 64 results matched *and* exactly 64 arrived |
| `LEDG[3]` | **FAIL** |
| `LEDR[7:0]` | mismatch count, saturating at 255 |
| `LEDR[15:8]` | results received |

If `LEDG[7]` is dark, the clock is the problem, not the array — wrong
`CLOCK_50` pin, or the board is not configured. Nothing else on the board is
worth looking at until that LED blinks.

The count check matters as much as the value check: an array that emits 63
correct results and drops one is broken, and comparing only the results that
*do* arrive would call that a pass.

---

## Running it

### Step 1 — regenerate the vectors (optional)

They are committed, so this is only needed if you change `K`, `M` or `NVEC`.

```bash
python hardware/fpga/gen_test_vectors.py
```

`NVEC` in `de2i150_mac_top.sv` must match `--vectors`. The generator uses the
real `conv2` weight tile from `results/checkpoints/baseline_best.pt` when that
checkpoint is present, and a seeded RNG otherwise.

### Step 2 — simulate the harness

Do this before opening Quartus. A broken harness and a broken array light the
same LED, and the board gives you no way to tell them apart.

```bash
iverilog -g2012 -s tb_de2i150_mac_top -o /tmp/h.vvp \
    fpga/tb_de2i150_mac_top.sv fpga/de2i150_mac_top.sv \
    rtl/rtl_design/mac_array.sv rtl/rtl_design/pe_int8.sv
vvp /tmp/h.vvp
```

Current result:

```
  cycles        : 150
  results recv  : 64 of 64
  mismatches    : 0
  LEDG[2] PASS  : 1

  HARNESS PASS -- safe to build in Quartus
```

### Step 3 — build

**File ▸ Open Project** ▸ `de2i150_mac_top.qpf`. Do not run the New Project
Wizard — it writes its own `.qsf` over the settings file and silently discards
the device choice, file list, `.mif` registrations and DSP flag.

Then Assignments ▸ Import Assignments ▸ `de2i150_pins.qsf`, and compile. Or
from the Quartus shell:

```bash
cd fpga
quartus_sh --flow compile de2i150_mac_top
```

The project must sit in `fpga/` — the `.mif` paths in the RTL resolve relative
to the project directory.

### Step 4 — program

Tools ▸ Programmer ▸ Hardware Setup ▸ USB-Blaster ▸ Auto Detect ▸ load
`de2i150_mac_top.sof` against the EP4CGX150 ▸ Start.

`.sof` is volatile and is lost at power-off, which is what you want while
iterating. Converting to `.pof` for the config flash only matters once the
design is final.

### Step 5 — read the result

`LEDG[2]` lit and `LEDG[3]` dark is a pass: the array computed all 64
matrix-vector products correctly, in hardware, after place-and-route.

---

## Measured results

These are read off the Fitter and Timing Analyzer reports, not predicted.

### Stage A — `mac_array` alone

| resource | used | available | % |
|---|---|---|---|
| logic elements | 17,384 | 149,760 | 12% |
| registers | 14,971 | 149,760 | 10% |
| **9-bit multipliers** | **256** | **720** | **36%** |
| M9K bits | 0 | 6,635,520 | 0% |
| Fmax, slow 85 C | +5.801 ns slack | — | ~69 MHz |
| verification | **64/64 outputs, 0 mismatches** | | |

**Two caveats on that logic-element figure.** This build has the real conv2
weights in ROM, and Quartus constant-propagated them -- trained weights cluster
near zero and do not span full INT8, so five of sixteen columns needed only
6-bit multipliers. The random-weight build, where operands use the full range,
came out at **~20,216 logic cells**, and that is the honest number for what a
general 16x16 array costs. A real accelerator loads weights at runtime and
cannot be specialised this way.

### Stage C — the full fused layer

| resource | used | available | % |
|---|---|---|---|
| logic elements | 49,961 | 149,760 | 33% |
| registers | 32,832 | 149,760 | 22% |
| memory bits | 968,192 | 6,635,520 | 15% |
| 9-bit multipliers | 320 | 720 | 44% |
| pins | 32 | 508 | 6% |
| Fmax, slow 85 C | +0.431 ns slack, TNS 0.0 | — | ~51 MHz |
| verification | **60/60 outputs, 0 mismatches** | | |

### On the multiplier count

An earlier draft of this file quoted **256 / 360 = 71%**, treating the device as
having 360 multipliers. Cyclone IV counts them in **9-bit elements** and this
part has **720**; a 16x16 array's 8x8 products each occupy one. The correct
figure is **36%**, and the difference changes a design conclusion:

| array | PEs | 9-bit elements | fits in 720? |
|---|---|---|---|
| 16x16 | 256 | 256 | yes, 36% |
| **24x24** | 576 | 576 | **yes, 80%** |
| 26x26 | 676 | 676 | tight, 94% |
| 32x32 | 1024 | 1024 | no |

So the multipliers are **not** the binding constraint they were described as,
and this board would support an array roughly 2.25x larger than the one built.

### On clock frequency

~51 MHz for the full layer against **1 GHz at 14 nm**. That gap is process and
tooling, not design: Cyclone IV is a 60 nm-class part whose DSP blocks have no
accumulator, so every partial-sum adder lands in LE carry chains. The board is
for **correctness**; the 14 nm synthesis is for **performance**. Neither number
substitutes for the other.

## One correction to the earlier comparison

I said previously that neither board has a hard CPU and so there is no fair
on-board baseline. That is **wrong for the DE2i-150**, which carries an
on-board **Intel Atom N2600** (dual-core, 1.6 GHz, Cedar Trail) connected to
the Cyclone IV over PCIe. That is a legitimate host-versus-accelerator
comparison on a single board, with one interconnect and one power domain.

It is worth having, and it is the natural phase 2:

1. Instantiate the Altera PCIe hard IP and a DMA path to `layer_top`
2. Restructure `layer_top`'s memories to synchronous single-port reads so they
   map to M9K (already on the "what remains" list in `MILESTONE_REPORT.md`)
3. Write a Linux kernel module on the Atom to move activations and weights
4. Run the same inference on the Atom in PyTorch and on the fabric, same board

That is a substantial piece of work — the PCIe stack is comparable in size to
everything in `rtl/` so far. The self-test here is the prerequisite either
way: there is no point debugging a DMA path into an array that has never been
shown to work on silicon.

---

## Known Quartus risks in the RTL

Flagged from a scan, none of them blocking, all worth knowing if the build
throws errors:

- `mac_array.sv:145-147` use **2D unpacked wire arrays**. Legal SystemVerilog
  and supported by Quartus, but not by Verilog-2001 — the files must be added
  as `SYSTEMVERILOG_FILE`, which `de2i150_mac_top.qsf` does.
- **Size casts** `X'(...)`: 8 in `layer_top.sv`, 7 in `tile_top.sv`. Neither
  file is in this build.
- **`$clog2` in port lists**: 16 in `layer_top.sv`, 5 in `band_sram.sv`, 2 in
  `writeback.sv`. Again, not in this build — but they are the first thing to
  check when `layer_top` goes to the board in phase 2.
