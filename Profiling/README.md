# Profiling & Roofline Analysis

HW/SW co-design inputs for `KeywordSpottingCNN`. All three scripts read
`src/model.py` and `src/dataset.py` live, run on synthetic input, and need no
dataset download and no trained checkpoint.

## Setup

```bash
pip install -r Profiling/requirements-profiling.txt
```

`torchaudio` is optional. Without it the scripts fall back to `src/dataset.py`'s
default constants (and say so in the report).

## Scripts

| Script | Output | What it gives you |
| ------ | ------ | ----------------- |
| `eda_profiling.py` | `eda_profiling.txt` | torchinfo layer summary: output shapes, params, mult-adds, plus the machine it ran on |
| `kws_analysis.py` | `kws_analysis.md` | Top layers by MAC count, and arithmetic intensity of the most MAC-intensive layer |
| `roofline.py` | `roofline.png`, `roofline.md` | Roofline plot: host CPU vs. proposed chiplet, dataflow comparison, SRAM and bandwidth requirements, Amdahl speedup |

Run them from the repository root:

```bash
python Profiling/eda_profiling.py
python Profiling/kws_analysis.py
python Profiling/roofline.py
```

## Which numbers change on your machine

- **Machine-independent** (identical everywhere): MACs, parameters, layer shapes,
  arithmetic intensity. These are derived analytically from the architecture.
- **Machine-dependent**: the environment header in `eda_profiling.txt`, and the
  measured host GFLOP/s point in `roofline.py` (timed at run time), which feeds
  the kernel and system speedup numbers.

## Platform hypotheses -- override these

`roofline.py` ships with placeholder specs for the host CPU and the proposed
chiplet. They are **hypotheses, not measurements**. Substitute your own:

```bash
python Profiling/roofline.py \
    --cpu-name "Ryzen 7 5800H" --cpu-gflops 400 --cpu-bw 51.2 \
    --array 32x32 --freq-mhz 500 --chiplet-bw 1.6 \
    --dataflow fused --bytes-per-elem 1
```

CPU peak is `cores x FMA units x vector lanes x 2 x clock`; CPU bandwidth comes
from the memory configuration. Both come off the spec sheet -- the script does
not guess them.

Key flags:

- `--dataflow {none,ws,fused}` -- DRAM traffic model: no reuse / weight-stationary /
  weight-stationary with activations kept in on-chip SRAM.
- `--bytes-per-elem {4,2,1}` -- FP32 / FP16 / INT8. Operation count is fixed by the
  architecture, so arithmetic intensity scales inversely with this.
- `--array`, `--freq-mhz` -- chiplet peak is derived as `PEs x 2 FLOP x clock`.
- `--accel-fraction` -- share of host runtime the chiplet replaces, for Amdahl.

## Rerun after model changes

The scripts import the model rather than hardcoding it, so after editing
`src/model.py` (pruning, quantization, architecture changes) just run them again.
Note that **unstructured** pruning will not change the MAC counts -- the zeros
still occupy the tensor. Structured/channel pruning will.
