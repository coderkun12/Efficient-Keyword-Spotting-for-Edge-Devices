"""
Operator-level runtime breakdown for the keyword-spotting CNN.

Produces the Amdahl accelerated-fraction figures the roofline and the
accelerator plan depend on: how much of host inference time is convolution,
how much is max-pool, BatchNorm and ReLU, and how much is everything else.

Writes Profiling/op_breakdown.txt.

WHY THIS EXISTS SEPARATELY:
kws_analysis.py counts MACs, which says what the work *is*. It does not say
where the *time* goes. On this model those disagree sharply -- max-pool is
about 0% of the MACs but a large share of the runtime -- and the accelerated
fraction has to come from measured time, not from MAC counts.

Requirements:
    pip install torch

Usage:
    python Profiling/op_breakdown.py
    python Profiling/op_breakdown.py --threads 1 --iters 200
"""

import argparse
import platform
import sys
import time
from pathlib import Path

try:
    import torch
    from torch.profiler import profile, ProfilerActivity
except ImportError:
    sys.exit("Missing package: torch\nInstall with:  pip install torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from software.src.model import KeywordSpottingCNN  # noqa: E402

try:
    import software.src.dataset as dataset
    N_MELS, NUM_CLASSES = dataset.N_MELS, dataset.NUM_CLASSES
except Exception:
    N_MELS, NUM_CLASSES = 40, 12

# Each profiler op is assigned to exactly ONE group, first match wins, so the
# shares sum to 100% with no double counting. Order matters here.
GROUPS = [
    ("convolution", ("convolution", "conv2d", "conv1d", "mkldnn_conv", "slow_conv", "thnn_conv")),
    ("max-pool",    ("pool",)),
    ("batchnorm",   ("batch_norm", "batchnorm")),
    ("relu",        ("clamp_min", "relu", "threshold")),
    ("linear",      ("addmm", "linear", "matmul", "mm")),
]


def parse_args():
    p = argparse.ArgumentParser(description="Per-operator runtime breakdown")
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads (1 matches src/benchmark.py's methodology)")
    p.add_argument("--iters", type=int, default=200, help="profiled iterations")
    p.add_argument("--warmup-s", type=float, default=4.0,
                   help="seconds of spinning before profiling, to settle boost clocks")
    p.add_argument("--time-frames", type=int, default=101)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "op_breakdown.txt"))
    return p.parse_args()


def classify(key):
    k = key.lower()
    for name, needles in GROUPS:
        if any(n in k for n in needles):
            return name
    return "other"


def main():
    args = parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)

    model = KeywordSpottingCNN(num_classes=NUM_CLASSES).eval()
    input_shape = (1, 1, N_MELS, args.time_frames)
    x = torch.randn(*input_shape)

    with torch.no_grad():
        t_end = time.perf_counter() + args.warmup_s
        while time.perf_counter() < t_end:
            model(x)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(args.iters):
                model(x)

    events = prof.key_averages()
    total = sum(e.self_cpu_time_total for e in events)
    if total <= 0:
        sys.exit("Profiler returned no CPU time; try more --iters.")

    by_group = {}
    for e in events:
        by_group.setdefault(classify(e.key), 0.0)
        by_group[classify(e.key)] += e.self_cpu_time_total

    conv = by_group.get("convolution", 0.0) / total
    feature = sum(by_group.get(g, 0.0)
                  for g in ("convolution", "max-pool", "batchnorm", "relu")) / total

    L = []
    w = L.append
    bar = "=" * 86
    w(bar)
    w("OPERATOR RUNTIME BREAKDOWN -- Keyword Spotting CNN (inference, batch 1)")
    w(bar)
    w(f"Generated       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Machine         : {platform.node()}")
    w(f"Platform        : {platform.platform()}")
    w(f"Processor       : {platform.processor()}")
    w(f"Python / torch  : {platform.python_version()} / {torch.__version__}")
    w(f"torch threads   : {args.threads}")
    w(f"Input shape     : {tuple(input_shape)}")
    w(f"Profiled iters  : {args.iters} (after {args.warmup_s:g} s warmup)")
    w(f"Total self CPU  : {total:,.0f} us")
    w("")

    w("--- Grouped shares (each op counted once; groups sum to 100%) " + "-" * 24)
    w(f"{'group':>14} | {'self CPU us':>14} | {'share':>8}")
    w("-" * 44)
    for name in ("convolution", "max-pool", "batchnorm", "relu", "linear", "other"):
        v = by_group.get(name, 0.0)
        w(f"{name:>14} | {v:>14,.0f} | {100 * v / total:>7.2f}%")
    w("")

    w("--- Top 15 individual operators " + "-" * 52)
    w(f"{'operator':<46} | {'self CPU us':>13} | {'share':>7}")
    w("-" * 72)
    for e in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]:
        w(f"{e.key[:46]:<46} | {e.self_cpu_time_total:>13,.0f} | "
          f"{100 * e.self_cpu_time_total / total:>6.2f}%")
    w("")

    w(bar)
    w("AMDAHL ACCELERATED FRACTIONS -- the numbers the roofline consumes")
    w(bar)
    w(f"  f(convolution only)              = {conv:.4f}   ({100 * conv:.2f}%)")
    w(f"  f(conv + pool + BN + ReLU)       = {feature:.4f}   ({100 * feature:.2f}%)")
    w("")
    w("  Ceiling on system speedup (infinitely fast accelerator, S = 1/(1-f)):")
    w(f"    convolution only        : {1 / (1 - conv):.2f}x")
    w(f"    conv + pool + BN + ReLU : {1 / (1 - feature):.2f}x")
    w("")
    w("  Feed these to the roofline as:")
    w(f"    CONV_FRACTION    = {conv:.2f}")
    w(f"    FEATURE_FRACTION = {feature:.2f}")
    w("  in Profiling/roofline_chiplet.py, and --accel-fraction for roofline.py.")
    w(bar)

    out_path = Path(args.out)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
