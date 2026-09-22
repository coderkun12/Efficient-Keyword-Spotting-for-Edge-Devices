# Synthesis results — `mac_array`, 16x16 INT8, 1 GHz

Cadence Genus 17.14-s037_1 · SAED14nm RVT · `saed14rvt_tt0p8v25c.lib`
(typical, 0.80 V, 25 °C) · clock period 1.0 ns.

Reproduce with:

```bash
cd rtl/synth && source setup_env.sh
SYN_TOP=mac_array CLK_PERIOD_NS=1.0 genus -f run_genus.tcl
```

## Headline

| | |
| --- | --- |
| Worst negative slack | **+299.3 ps** |
| Total negative slack | 0.0 |
| Violating paths | **0** |
| Leaf cells | 60,607 (17,071 sequential, 43,536 combinational) |
| Cell area | **40,486.9 µm² = 0.0405 mm²** |
| Power, active | **52.452 mW** |
| Power, leakage | **17.5 µW** (0.03% of total) |

At 500 MHz the same design gave +993 ps and 40,474.9 µm², so doubling the clock
cost 0.03% of area and not one extra gate. The real combinational path through a
processing element is 347 ps.

## Files

| File | Contents |
| --- | --- |
| `qor.rpt` | one-page summary — start here |
| `timing.rpt` | the 20 worst paths in full |
| `timing_lint.rpt` | explains the `TIM-11` warning |
| `area.rpt`, `area_hier.rpt` | cell area, flat and by hierarchy |
| `power.rpt` | leakage vs dynamic, per clock-gating cell |
| `gates.rpt` | cell-type histogram |
| `clock_gating.rpt` | what Genus inserted |
| `mac_array.sdc` | constraints as written back by Genus |
| `mac_array_netlist.v.gz` | gate-level netlist, `gunzip` to use (19.5 MB raw) |

## Caveats

**Wireload model, not extracted parasitics.** The report says
`Wireload mode: enclosed` and `Net Area 0.000`, so no real interconnect was
modelled. Place and route would shrink the 299 ps of slack. It was not run:
SAED14nm ships no Cadence technology LEF, only Milkyway and OpenAccess files.

**No switching activity annotation.** Genus used default toggle rates, so the
dynamic figure is an estimate. Leakage is exact.

**I/O budget.** `constraints.sdc` allows 30% of the period at each boundary, so
every worst path runs input-port to first register rather than register to
register. The register-to-register paths have more slack.

**Operand isolation unavailable.** It is a beta feature on this release with no
`logic_gating` license. Clock gating was inserted successfully.
