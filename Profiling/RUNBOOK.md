# Host Measurement Runbook

Instructions for capturing the machine-dependent numbers the accelerator
roofline depends on. Forward this file to whoever owns the reference machine.

## Which machine

All measured numbers must come from **one** machine, and it should be the same
one that produced `results/checkpoints/` and `results/benchmark_results.json`.
Mixing measurements from two laptops silently breaks the roofline: the CPU peak,
the ridge point and every speedup are all derived from one host's specifications.

## Setup

You need the repository, not just the scripts. Every profiling script imports
`src/model.py` live, so a loose `.py` file will fail with
`ModuleNotFoundError: No module named 'model'`.

```bash
git clone <repo-url>           # or: git pull
cd Efficient-Keyword-Spotting-for-Edge-Devices
pip install -r Profiling/requirements-profiling.txt
pip install py-cpuinfo         # optional, adds vector ISA flags to the vitals report
```

Run everything **from the repository root**, not from inside `Profiling/`.

## Commands to run

```bash
python Profiling/host_vitals.py
python Profiling/op_breakdown.py
python Profiling/eda_profiling.py
python Profiling/kws_analysis.py
python Profiling/roofline.py --dataflow fused --array 16x16 --bytes-per-elem 1
python src/benchmark.py --latency-runs 200 --warmup-runs 40
```

The last one needs the trained checkpoints and the dataset. Skip it if they are
not on this machine and say so.

## Files to send back

| File | What it carries |
| --- | --- |
| `Profiling/host_vitals.txt` | CPU model, cores, clocks, RAM modules and speed, measured throughput at 1/2/4/8 threads |
| `Profiling/op_breakdown.txt` | Per-operator runtime shares and the Amdahl accelerated fractions |
| `Profiling/eda_profiling.txt` | Layer table with the machine header |
| `Profiling/kws_analysis.md` | MAC ranking and arithmetic intensity |
| `Profiling/roofline.md` | Roofline report with the measured host point |
| `results/benchmark_results.json` | Per-variant latency and accuracy (only if benchmark.py ran) |
| `results/benchmark_machine.json` | Which machine produced those latencies (written automatically) |

Zip the `Profiling/` folder and send it. That covers everything except the
benchmark JSON.

## Two things to check before sending

**1. Was the machine quiet?** `host_vitals.txt` has a `spread` column showing
p90 divided by best latency. Under about 1.3x the machine was quiet and the
median is trustworthy. Above about 2x it was throttling or loaded, and only the
best-case column is reproducible. If the spread is wide, close everything else,
plug in the charger, set the power plan to maximum performance, and re-run.

**2. Does `kws_analysis.md` match?** Those numbers are pure static analysis and
contain no timing, so they should be **identical** on every machine. If yours
differs, the two of us are on different versions of `src/model.py` and every
other number is suspect. Check that first.

## What we still have to look up by hand

`host_vitals.txt` Section 1 prints the raw hardware query output. Two values in
it cannot be read off directly and need a spec-sheet check:

- **All-core sustained boost clock.** Windows reports `MaxClockSpeed` from
  firmware, which is usually neither the base nor the real turbo ceiling. Look
  up the actual part on Intel ARK or AMD's spec page.
- **Memory channels.** Count the modules listed under `DeviceLocator`. Two
  modules on different channels means dual channel; one module means single
  channel and **half** the bandwidth. Peak bandwidth is
  `module speed (MT/s) x channels x 8 bytes`.

Both feed the CPU roofline directly, so getting them wrong shifts the ridge
point and the bound classification.
