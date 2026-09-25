"""
Single roofline figure for the KWS HW/SW co-design proposal:
measured i5-10210U host vs. the hypothesised INT8 systolic-array chiplet.

Unlike roofline.py (which sweeps dataflows and writes a full markdown report),
this script produces ONE presentation figure with exactly two platforms and
two operating points.

Defaults: INT8, 16x16 PE array, 500 MHz.

Usage:
    python Profiling/roofline_chiplet.py
    python Profiling/roofline_chiplet.py --array 32x32 --freq-mhz 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torchinfo import summary
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from software.src.model import KeywordSpottingCNN  # noqa: E402

try:
    import software.src.dataset as dataset
    N_MELS, NUM_CLASSES = dataset.N_MELS, dataset.NUM_CLASSES
except Exception:
    N_MELS, NUM_CLASSES = 40, 12

# --- Host platform: spec-sheet hypotheses (NOT measured) -------------------
CPU_NAME = "i5-12450H"
# Reference host = LAPTOP-FFRP12TK, the machine that produced the trained
# checkpoints and results/benchmark_results.json. Specs confirmed by the raw
# hardware probe in Profiling/host_vitals.txt Section 1.
#   i5-12450H: 4 P-cores (Golden Cove) + 4 E-cores (Gracemont), 12 threads.
#   P-core FP32 peak = 8 AVX2 lanes x 2 flop (FMA) x 2 FMA units x 4.4 GHz
#                    = 140.8 GFLOP/s.
# Batch-1 keyword spotting is an always-on streaming workload classifying one
# 1 s window at a time, so it gets ONE core, not the all-core ~774 GFLOP/s.
CPU_PEAK_PER_CORE = 140.8
CPU_BW_GBS = 51.2         # 2 x DDR4-3200 modules populated => dual channel

# --- Measured on the reference host (Profiling/op_breakdown.txt, 1 thread) --
CONV_FRACTION = 0.5538     # aten::*convolution* share of host runtime
FEATURE_FRACTION = 0.9774  # conv + max-pool + BatchNorm + ReLU

NL = "\n"


def parse_args():
    p = argparse.ArgumentParser(description="Single roofline: host vs proposed chiplet")
    p.add_argument("--array", default="16x16", help="PE array dimensions")
    p.add_argument("--freq-mhz", type=float, default=500.0)
    p.add_argument("--chiplet-bw", type=float, default=1.6,
                   help="chiplet interface bandwidth, GB/s")
    p.add_argument("--bytes-per-elem", type=int, default=1, choices=[1, 2, 4],
                   help="chiplet precision: 1 = INT8 (default)")
    p.add_argument("--time-frames", type=int, default=101)
    p.add_argument("--measure-iters", type=int, default=800)
    p.add_argument("--host-ms", type=float, default=None,
                   help="pin host batch-1 latency in ms (e.g. the median from "
                        "Profiling/host_vitals.txt). Overrides both the published "
                        "benchmark value and live re-timing.")
    p.add_argument("--host-ms-note", default="host_vitals.txt, 1 thread, median",
                   help="provenance string shown on the plot for --host-ms")
    p.add_argument("--measure-live", action="store_true",
                   help="re-time the host here instead of using the published "
                        "results/benchmark_results.json latency (noisy on this machine)")
    p.add_argument("--threads", type=int, default=1,
                   help="CPU threads for the measured point; CPU peak scales to match")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "roofline_chiplet.png"))
    return p.parse_args()


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def analyse(args):
    model = KeywordSpottingCNN(num_classes=NUM_CLASSES).eval()
    input_shape = (1, 1, N_MELS, args.time_frames)
    res = summary(model, input_size=input_shape, verbose=0)

    layers = [{
        "macs": li.macs,
        "in_elems": numel(li.input_size),
        "out_elems": numel(li.output_size),
        "weight_elems": li.num_params,
    } for li in res.summary_list if li.is_leaf_layer and li.macs > 0]

    total_macs = sum(ly["macs"] for ly in layers)
    total_flops = 2 * total_macs
    weight_elems = sum(ly["weight_elems"] for ly in layers)

    # Host: FP32, every operand re-fetched per layer (no layer fusion in PyTorch).
    host_bytes = sum(ly["in_elems"] + ly["out_elems"] + ly["weight_elems"]
                     for ly in layers) * 4

    # Chiplet: weights resident, activations stay in on-chip SRAM between layers.
    # Only the mel input, the logits and the weights cross the interface.
    chip_bytes = (numel(input_shape) + NUM_CLASSES + weight_elems) * args.bytes_per_elem

    return model, res, input_shape, total_macs, total_flops, host_bytes, chip_bytes


def published_cpu_latency(variant="baseline_best"):
    """Canonical host latency from results/benchmark_results.json.

    Preferred over re-timing inside this script. src/benchmark.py measures
    batch-1, single-threaded latency -- the right model for an always-on
    keyword spotter -- and its number is the one already published in
    results/benchmark_table.md, so the roofline stays consistent with it.

    Re-timing here is unreliable: this 15 W laptop part boosts to 4.2 GHz and
    throttles toward 1.6 GHz base, so repeated runs of the same measurement
    have ranged 10.5-24.8 ms. The published run reports 17.072 +/- 6.757 ms,
    and that standard deviation is the same effect.
    """
    path = REPO_ROOT / "results" / "benchmark_results.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("results", data) if isinstance(data, dict) else data
        for r in rows:
            if r.get("variant") == variant and (r.get("latency_median_ms")
                                                or r.get("latency_mean_ms")):
                # Median first: the mean is dragged up by throttling outliers.
                if r.get("latency_median_ms"):
                    return r["latency_median_ms"], None, str(path) + " (median)"
                return r["latency_mean_ms"], r.get("latency_std_ms"), str(path) + " (mean)"
    except Exception:
        pass
    return None, None, None


def measure_cpu(model, input_shape, iters, threads, warmup_s=4.0):
    """Live fallback: best of `iters` batch-1 inferences after a thermal warmup."""
    torch.set_num_threads(threads)
    x = torch.randn(*input_shape)
    with torch.no_grad():
        t_end = time.perf_counter() + warmup_s
        while time.perf_counter() < t_end:
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[0] * 1e3


def fmt(v, _pos):
    if v >= 1e3:
        return f"{v / 1e3:g} TOP/s"
    if v >= 1:
        return f"{v:g} GOP/s"
    return f"{v * 1e3:g} MOP/s"


def main():
    args = parse_args()
    torch.manual_seed(42)

    (model, res, input_shape, total_macs, total_flops,
     host_bytes, chip_bytes) = analyse(args)

    rows, cols = (int(v) for v in args.array.lower().split("x"))
    pes = rows * cols
    chip_peak = pes * 2 * args.freq_mhz * 1e6 / 1e9        # GOP/s
    chip_ridge = chip_peak / args.chiplet_bw

    host_ai = total_flops / host_bytes
    chip_ai = total_flops / chip_bytes

    cpu_peak = CPU_PEAK_PER_CORE * args.threads
    cpu_ridge = cpu_peak / CPU_BW_GBS
    if args.host_ms is not None:
        cpu_ms, cpu_std, cpu_src = args.host_ms, None, args.host_ms_note
    elif args.measure_live:
        cpu_ms = measure_cpu(model, input_shape, args.measure_iters, args.threads)
        cpu_std, cpu_src = None, f"re-timed here, best of {args.measure_iters}"
    else:
        cpu_ms, cpu_std, path = published_cpu_latency()
        if cpu_ms is None:
            cpu_ms = measure_cpu(model, input_shape, args.measure_iters, args.threads)
            cpu_std, cpu_src = None, f"re-timed here, best of {args.measure_iters}"
        else:
            cpu_src = "results/benchmark_results.json (src/benchmark.py)"
    cpu_ach = total_flops / (cpu_ms / 1e3) / 1e9
    chip_ach = min(chip_peak, args.chiplet_bw * chip_ai)
    chip_us = total_flops / (chip_ach * 1e9) * 1e6

    kernel_speedup = chip_ach / cpu_ach
    sys_conv = 1.0 / ((1 - CONV_FRACTION) + CONV_FRACTION / kernel_speedup)
    sys_feat = 1.0 / ((1 - FEATURE_FRACTION) + FEATURE_FRACTION / kernel_speedup)

    dtype = {4: "FP32", 2: "FP16", 1: "INT8"}[args.bytes_per_elem]

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    fig.patch.set_facecolor("#F7F7FA")

    lo, hi = 1e-1, 1e4
    xs = [lo * (hi / lo) ** (i / 700.0) for i in range(701)]

    ax.plot(xs, [min(cpu_peak, CPU_BW_GBS * x) for x in xs],
            color="#1f3fd8", lw=3.0, zorder=3,
            label=f"CPU roofline ({CPU_NAME}, FP32, {args.threads} thread)")
    ax.plot(xs, [min(chip_peak, args.chiplet_bw * x) for x in xs],
            color="#0f7a32", lw=3.0, zorder=3,
            label=f"Chiplet roofline ({args.array} {dtype} @ {args.freq_mhz:g} MHz)")

    for y, c in ((cpu_peak, "#1f3fd8"), (chip_peak, "#0f7a32")):
        ax.axhline(y, color=c, ls="--", lw=1.0, alpha=0.55, zorder=1)
    for x, c in ((cpu_ridge, "#1f3fd8"), (chip_ridge, "#0f7a32")):
        ax.axvline(x, color=c, ls=":", lw=1.2, alpha=0.6, zorder=1)

    ax.plot([host_ai], [cpu_ach], "D", ms=15, color="#1f3fd8",
            mec="white", mew=1.6, zorder=6,
            label=f"KWS CNN on CPU: {cpu_ach:.1f} GFLOP/s (measured, batch 1, 1 thread)")
    ax.plot([chip_ai], [chip_ach], "*", ms=26, color="#0f7a32",
            mec="white", mew=1.4, zorder=6,
            label=f"KWS CNN on chiplet: {chip_ach:.0f} GOP/s (projected)")

    ax.annotate("", xy=(chip_ai, chip_ach * 0.93), xytext=(host_ai, cpu_ach * 1.07),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.8,
                                ls="--", shrinkA=8, shrinkB=12), zorder=5)

    # CPU and chiplet peaks are numerically close, so separate the two peak
    # labels horizontally rather than stacking them on top of each other.
    ax.text(lo * 1.35, cpu_peak * 1.18,
            f"CPU peak: {cpu_peak:.1f} GFLOP/s "
            f"({args.threads} core x 8 AVX2 lanes x 2 flop x 4.2 GHz, spec sheet)",
            color="#1f3fd8", fontsize=11, fontweight="bold")
    ax.text(hi * 0.96, chip_peak * 1.30,
            f"Chiplet peak: {chip_peak:.0f} GOP/s "
            f"({pes} INT8 MACs x 2 OP x {args.freq_mhz:g} MHz)",
            color="#0f7a32", fontsize=11, fontweight="bold", ha="right", va="bottom")

    ax.annotate("CPU ridge" + NL + f"{cpu_ridge:.1f} FLOP/B",
                xy=(cpu_ridge, cpu_peak), xytext=(cpu_ridge * 0.30, cpu_peak * 3.2),
                color="#1f3fd8", fontsize=10.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="#1f3fd8", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1f3fd8"))
    ax.annotate("Chiplet ridge" + NL + f"{chip_ridge:.0f} FLOP/B",
                xy=(chip_ridge, chip_peak), xytext=(chip_ridge * 0.42, 1.5),
                color="#0f7a32", fontsize=10.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="#0f7a32", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#0f7a32"))

    ax.annotate(
        "KWS CNN on CPU (measured)" + NL + "FP32, no layer fusion" + NL
        + f"AI = {host_ai:.1f} FLOP/B" + NL
        + f"{cpu_ach:.1f} GFLOP/s  ({cpu_ms:.1f} ms/inf)",
        xy=(host_ai, cpu_ach), xytext=(host_ai * 0.013, cpu_ach * 0.15),
        fontsize=11, color="#1f3fd8", va="center",
        arrowprops=dict(arrowstyle="->", color="#1f3fd8", lw=1.6),
        bbox=dict(boxstyle="round,pad=0.5", fc="#eef2ff", ec="#1f3fd8", lw=1.5))

    ax.annotate(
        "KWS CNN on chiplet (projected)" + NL
        + f"{dtype}, weight-stationary + fused activations" + NL
        + f"AI = {chip_ai:,.0f} FLOP/B" + NL
        + f"{chip_ach:.0f} GOP/s  ({chip_us:.0f} us/inf)" + NL
        + f"COMPUTE-BOUND (AI >> ridge {chip_ridge:.0f})",
        xy=(chip_ai, chip_ach), xytext=(chip_ai * 0.030, chip_peak * 5.5),
        fontsize=11, color="#0f7a32", va="center",
        arrowprops=dict(arrowstyle="->", color="#0f7a32", lw=1.6),
        bbox=dict(boxstyle="round,pad=0.5", fc="#eaf7ee", ec="#0f7a32", lw=1.5))

    ax.text(hi * 0.90, cpu_ach * 0.045,
            f"Kernel speedup: {kernel_speedup:.1f}x" + NL
            + f"System (conv only, f={CONV_FRACTION:.0%}): {sys_conv:.2f}x" + NL
            + f"System (conv+pool+BN+ReLU, f={FEATURE_FRACTION:.0%}): {sys_feat:.2f}x",
            fontsize=11.5, fontweight="bold", color="#0f7a32", ha="right",
            bbox=dict(boxstyle="round,pad=0.55", fc="white", ec="#0f7a32", lw=1.8))

    # The headline insight: the two peaks are nearly equal. The win comes from
    # sustained utilisation, not from a higher peak.
    ax.text(lo * 1.35, 0.155,
            f"Where the {kernel_speedup:.0f}x comes from: {chip_peak / cpu_peak:.1f}x more peak "
            f"({chip_peak:.0f} vs {cpu_peak:.0f}) x {cpu_peak / cpu_ach:.1f}x better utilisation." + NL
            + f"The CPU sustains only {100 * cpu_ach / cpu_peak:.0f}% of its peak on this model; "
            "a systolic array is built to hold ~100%.",
            fontsize=10, color="#333", va="bottom",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff8e5", ec="#c9a227", lw=1.3))

    ax.text(hi * 0.55, 0.155, "Compute-" + NL + "bound", fontsize=13, color="#bbb",
            style="italic", fontweight="bold", ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.1, chip_peak * 30)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Performance", fontsize=14, fontweight="bold")
    ax.set_title(
        "Roofline -- Keyword Spotting CNN" + NL
        + f"{CPU_NAME} host vs. proposed {args.array} {dtype} systolic-array chiplet "
        f"@ {args.freq_mhz:g} MHz",
        fontsize=15, fontweight="bold")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.96)

    fig.text(0.012, 0.042,
             f"Model: KeywordSpottingCNN, {res.total_params:,} params, "
             f"{total_macs / 1e6:.1f} M MACs/inference ({total_flops / 1e6:.1f} MFLOP), "
             f"input {list(input_shape)}.   "
             f"Host DRAM traffic {host_bytes:,} B (FP32, no reuse); "
             f"chiplet {chip_bytes:,} B ({dtype}, weights resident + activations on-chip).",
             fontsize=8.8, color="#555")
    fig.text(0.012, 0.024,
             f"Required interface bandwidth = {chip_peak / chip_ai:.2f} GB/s of "
             f"{args.chiplet_bw:g} GB/s provided "
             f"({args.chiplet_bw / (chip_peak / chip_ai):.1f}x margin).   "
             f"CPU point: {cpu_ms:.2f}"
             + (f" +/- {cpu_std:.2f}" if cpu_std else "")
             + f" ms, batch 1, single-threaded, from {cpu_src}.",
             fontsize=8.8, color="#555")
    fig.text(0.012, 0.006,
             "CPU peak and bandwidth are spec-sheet hypotheses, not measurements; "
             "the chiplet is a hypothesis throughout (no RTL, no synthesis yet).   "
             "For reference, INT8 QAT already reaches 5.60 ms in software on the same host.",
             fontsize=8.8, color="#555")

    fig.tight_layout(rect=(0, 0.062, 1, 1))
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())

    print(f"Wrote {args.out}\n")
    print(f"Model   : {total_macs:,} MACs ({total_flops:,} FLOP)")
    print(f"Host    : AI {host_ai:8.2f} FLOP/B | {cpu_ach:7.1f} GFLOP/s | {cpu_ms:.2f} ms "
          f"[{cpu_src}]")
    print(f"Chiplet : AI {chip_ai:8.2f} FLOP/B | {chip_ach:7.1f} GOP/s peak     | {chip_us:.1f} us")
    print(f"Ridges  : CPU {cpu_ridge:.2f} | chiplet {chip_ridge:.1f} -> "
          f"{'compute' if chip_ai > chip_ridge else 'memory'}-bound")
    print(f"Speedup : kernel {kernel_speedup:.2f}x | "
          f"system {sys_conv:.2f}x (f={CONV_FRACTION:.0%}) | "
          f"{sys_feat:.2f}x (f={FEATURE_FRACTION:.0%})")


if __name__ == "__main__":
    main()
