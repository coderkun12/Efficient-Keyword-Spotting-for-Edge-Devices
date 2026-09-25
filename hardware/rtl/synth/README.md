# Synthesis — scripts, results, and how to reproduce them

Cadence Genus 17.14-s037_1, SAED14nm RVT, typical corner, 0.8 V, 25 °C.

## Measured results

`mac_array`, the 16x16 INT8 systolic array:

| | 500 MHz | 1 GHz |
| --- | --- | --- |
| Worst negative slack | +993 ps | **+299 ps** |
| Violating paths | 0 | **0** |
| Leaf cells | 60,607 | 60,607 |
| Sequential cells | 17,071 | 17,071 |
| Combinational cells | 43,536 | 43,536 |
| Cell area | 40,474.9 µm² | 40,486.9 µm² |

**1 GHz closes for a 0.03% area cost and not one extra gate.** The real
combinational path through a PE is 347 ps against a 1,000 ps period, so the
design was never near the timing wall. `PIPE=2`, held in reserve for timing
closure, is unnecessary.

Power at 1 GHz: **52.452 mW** active, of which leakage is **17.5 µW** — 0.03% of
the total. Leakage is what an always-on part burns between wake words, so it is
the figure that matters most here.

Three estimates this run replaced:

| | Estimated | Measured | |
| --- | --- | --- | --- |
| Sequential cells | 17,239 | 17,071 | **1.0% error** |
| Leaf cells | 153,600 | 60,607 | 2.5x pessimistic |
| Cell area | 0.102 mm² | **0.0405 mm²** | 2.5x pessimistic |

The register model was right; the per-cell area figure, scaled from the
reference project's BF16 synthesis, was not. A PE is 237 cells and 158 µm².

## Running

```bash
cd rtl/synth
source setup_env.sh      # finds the library, exports SAED14_LIB
bash check_setup.sh      # optional pre-flight diagnostic
bash smoke_test.sh       # can Genus read and elaborate all 5 modules?
bash run_all_stages.sh   # the real synthesis
```

One stage on its own, at any period:

```bash
SYN_TOP=mac_array CLK_PERIOD_NS=1.0 genus -f run_genus.tcl
```

**Do not pass `SYN_PARAMS`.** Genus 17.14 rejects `elaborate -parameters`, and
every RTL default already is the design point. To try a different shape, edit
the default in the `.sv` file.

## Files

| File | Role |
| --- | --- |
| `setup_env.sh` | locates the SAED14 library and exports `SAED14_LIB` |
| `check_setup.sh` | pre-flight: directory, files, line endings, tool, library |
| `smoke_test.sh` / `.tcl` | elaborate each module, one Genus process each |
| `run_genus.tcl` | the synthesis flow |
| `run_all_stages.sh` | the three stages in order |
| `constraints.sdc` | clock, I/O budget, false path on reset |

## Why the stage order matters

**`mac_array` first.** It is 256 processing elements and nothing else, so its
timing and area are uncontaminated by memory. This is the run that decided the
array size: at 60,607 cells, scaling says a 32x32 array would be about 243,000
cells and 0.163 mm², which is 23x below the size that failed place-and-route in
the reference project. 32x32 is clearly feasible.

**`writeback` second.** It isolates the requantiser's 33x17 multiply, the other
plausible critical path. It runs at a quarter of the array's rate after pooling,
so if it is slow it can be time-multiplexed rather than pipelined.

**`layer_top` last.** Read the caveat below before quoting its area.

## The memory caveat — read before reporting `layer_top` area

`layer_top` contains about **376 kbit** of storage:

| Memory | Bits |
| --- | --- |
| weight memory, 1152 x 128 | 147,456 |
| activation band, 2 banks x 3 rows x 3232 B | 155,136 |
| row accumulator, 128 x 448 | 57,344 |
| write-back line buffer, 64 x 128 | 8,192 |

Without SRAM macros Genus maps all of it to **flip-flops**, roughly 1.9M cells,
which swamps the array's 60,607 and makes the area report meaningless. Three
options, in order of preference:

1. **Map to SRAM macros** if the SAED14nm memory compiler is available. This is
   the right answer and gives a real area number.
2. **Shrink the parameters** for a first pass, e.g. `MAX_KTILES 4 MAXW 32
   BAND_DEPTH 256` edited into the `.sv` defaults, and report memory separately
   by hand from the table above. Timing stays representative; area does not.
3. **Accept it and say so**, reporting array and memory area separately.

Do not quote a total area from option 2 or 3 as if it were the design's area.

## Reading the reports

| Report | What matters |
| --- | --- |
| `qor.rpt` | one page: slack, violating paths, cell counts, area. Start here. |
| `timing.rpt` | the 20 worst paths in full |
| `timing_lint.rpt` | explains any `TIM-11` warning, usually unconstrained paths |
| `area_hier.rpt` | split by submodule |
| `power.rpt` | leakage vs dynamic, and the clock-gating cells |
| `gates.rpt` | cell-type histogram |

At 1 GHz every worst path runs `a_vec` to `psum_q_reg` in PE(0,0), i.e.
input-port to first register. The register-to-register paths have more slack and
do not appear. That is an artifact of the 30% I/O budget in `constraints.sdc`,
not of the logic.

## Place and route

**Not possible at 14nm.** SAED14nm ships no Cadence technology LEF — only
Milkyway (`saed14nm_1p9m_mw.tf`) and OpenAccess (`saed14nm_1p9m_oa.tf`) tech
files, plus a cell-only LEF with no `LAYER`, `SITE` or `UNITS` section. The kit
is packaged for the Synopsys flow; Genus read the Liberty file only because
Liberty is vendor-neutral.

Two routes remain open. Place and route at 45nm with `gsclib045`, a complete
Cadence kit (tech LEF, macro LEF, QRC, Liberty) present on the same machine,
which would require re-synthesising against its Liberty first since the netlist
is mapped to `SAEDRVT14_*` cells. Or Synopsys IC Compiler at 14nm, using the
Milkyway files beside each cell library.

## Two things fixed for synthesis

**The weight memory** was restructured from byte-addressed to one `M*8`-bit word
per (k-tile, tap). Byte addressing needed sixteen independent 18,432:1 muxes,
about fifteen levels each, which would have been the critical path — Genus would
have characterised a mux tree instead of the array. It is now a single 1,152:1
read at one address, and mappable to an SRAM macro.

**A forward reference in `layer_top.sv`.** `w_top` was assigned from `wkt` and
`wcnt` eight lines before they were declared. Icarus elaborates a whole module
before resolving names and accepted it through 107 passing tests; Genus reads
top to bottom and rejected it outright. `rtl/sim/lint_rtl.py` now checks for
this (L7) and for comments that look like synthesis pragmas (L8).

## Low-power settings

Every `set_db` for low power is wrapped in `catch`, because attribute names
drift between Genus releases and an unknown one aborts the run. On 17.14:

| Setting | Result |
| --- | --- |
| `lp_insert_clock_gating` | accepted, gating inserted |
| `lp_clock_gating_prefix` | accepted |
| `lp_clock_gating_min_flops` | not a recognised attribute, skipped |
| `lp_insert_operand_isolation` | beta feature, no `logic_gating` license |

Operand isolation would have captured some of the 14–51% zero activations after
ReLU. Its absence costs dynamic power, nothing functional.
