# Reference-host profiling data

Everything here was measured on **LAPTOP-FFRP12TK** (i5-12450H), the machine
the model was trained on. It is the baseline the whole accelerator argument
rests on:

- **8.32 ms** host latency, batch 1, single thread, median of 800 runs after a
  5-second warmup
- **55.38%** convolution share of runtime
- **97.74%** for convolution + max-pool + BatchNorm + ReLU together
- **38.1%** max-pool at ~0% of the MACs — the finding the fused write-back
  exists because of

Every Amdahl figure in `MILESTONE_REPORT.md` derives from these numbers.

## There is a second, non-equivalent copy in `Profiling/`

`Profiling/host_vitals.txt`, `Profiling/op_breakdown.txt` and the rest were
regenerated on **DESKTOP-K8PQA4Q**, a different machine. They are kept because
the scripts that produced them live there, and re-running is how you check the
tooling still works — but they are **not** the reference.

A different CPU gives different absolute latencies and different operator
shares. Quoting a number from `Profiling/` alongside one from here would move
every speedup figure in the report without anything looking wrong.

**When a figure in the report disagrees with a file, this directory wins.**
