"""
Roofline analysis for the keyword-spotting CNN: host CPU vs. a proposed
weight-stationary AI chiplet.

Writes Profiling/roofline.png and Profiling/roofline.md.

Three classes of number live here -- keep them straight:
  * Derived from the model (exact, machine-independent): per-layer MACs and
    operand sizes, from torchinfo.
  * Measured on this machine: host CPU achieved GFLOP/s (timed here, batch 1).
  * Platform hypotheses (EDIT THESE): CPU peak FLOP/s and bandwidth, chiplet
    array size / frequency / interface bandwidth, accelerated fraction.

DATAFLOW MODELS (--dataflow):
  none    Every operand fetched from DRAM per layer, no reuse. Worst case.
  ws      Weight-stationary: each weight loaded into the PE array once and
          reused across the whole feature map and the whole batch. Activations
          still stream to/from DRAM between layers.
  fused   Weight-stationary AND activations kept in on-chip SRAM between
          layers (layer fusion). Only the mel input, the logits, and the
          weights ever cross the DRAM interface. Requires enough SRAM to hold
          the largest live activation pair -- the script reports how much.

Requirements:
    pip install torch torchinfo matplotlib

Usage:
    python Profiling/roofline.py
    python Profiling/roofline.py --dataflow fused --freq-mhz 500 --array 32x32
    python Profiling/roofline.py --dataflow fused --bytes-per-elem 1   # INT8
"""

import argparse
import sys
import time
from pathlib import Path

_MISSING = []
try:
    import torch
except ImportError:
    _MISSING.append("torch")
try:
    from torchinfo import summary
except ImportError:
    _MISSING.append("torchinfo")
try:
    import matplotlib
    matplotlib.use("Agg")  # headless: never needs a display
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError:
    _MISSING.append("matplotlib")

if _MISSING:
    sys.exit(
        "Missing required package(s): " + ", ".join(_MISSING) + "\n"
        "Install with:  pip install " + " ".join(_MISSING) + "\n"
        "(for torch, see https://pytorch.org/get-started/locally/ to get the "
        "right CUDA build for your machine)"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from software.src.model import KeywordSpottingCNN  # noqa: E402


def load_dataset_config():
    try:
        import software.src.dataset as dataset
        return dataset.N_MELS, dataset.NUM_CLASSES
    except Exception:
        return 40, 12


N_MELS, NUM_CLASSES = load_dataset_config()

# ---------------------------------------------------------------------------
# PLATFORM HYPOTHESES -- edit or override on the command line
# ---------------------------------------------------------------------------
# Host CPU: Intel Core i5-10210U (4C/8T, AVX2, boost ~4.2 GHz, DDR4-2667 dual ch.)
#   peak FP32 = 4 cores x 2 FMA x 8 lanes x 2 flop x 4.2 GHz ~= 268.8 GFLOP/s
DEFAULT_CPU_NAME = "i5-10210U"
DEFAULT_CPU_GFLOPS = 268.8
DEFAULT_CPU_BW = 45.8

# Chiplet: weight-stationary systolic array. Peak is derived from array size
# and clock, not hardcoded: rows x cols MACs x 2 FLOP/MAC x frequency.
DEFAULT_ARRAY = "16x16"
DEFAULT_FREQ_MHZ = 500.0
DEFAULT_CHIPLET_BW = 1.6  # GB/s sustained over the host/DRAM interface

# Share of host runtime spent in Conv2d, i.e. the part the chiplet takes over.
# Source: torch.profiler on this model, batch 1 -- aten::convolution accounted
# for ~47% of forward CPU time. Override with --accel-fraction.
DEFAULT_ACCEL_FRACTION = 0.47

DTYPE_NAME = {4: "FP32", 2: "FP16", 1: "INT8"}


def parse_args():
    p = argparse.ArgumentParser(description="Roofline plot for the KWS CNN")
    p.add_argument("--cpu-name", default=DEFAULT_CPU_NAME)
    p.add_argument("--cpu-gflops", type=float, default=DEFAULT_CPU_GFLOPS,
                   help="host CPU peak throughput, GFLOP/s")
    p.add_argument("--cpu-bw", type=float, default=DEFAULT_CPU_BW,
                   help="host CPU peak memory bandwidth, GB/s")
    p.add_argument("--array", default=DEFAULT_ARRAY,
                   help="PE array dimensions, e.g. 16x16 or 32x32")
    p.add_argument("--freq-mhz", type=float, default=DEFAULT_FREQ_MHZ,
                   help="chiplet clock frequency in MHz")
    p.add_argument("--chiplet-bw", type=float, default=DEFAULT_CHIPLET_BW,
                   help="chiplet interface bandwidth, GB/s")
    p.add_argument("--dataflow", default="ws", choices=["none", "ws", "fused"],
                   help="DRAM traffic model (see module docstring)")
    p.add_argument("--bytes-per-elem", type=int, default=4, choices=[1, 2, 4],
                   help="4 = FP32 (default), 2 = FP16/BF16, 1 = INT8")
    p.add_argument("--batch-size", type=int, default=1,
                   help="inferences per weight load (weight-stationary amortisation)")
    p.add_argument("--accel-fraction", type=float, default=DEFAULT_ACCEL_FRACTION,
                   help="fraction of host runtime the chiplet replaces (Amdahl)")
    p.add_argument("--measure-iters", type=int, default=100,
                   help="timed CPU iterations for the measured point (0 to skip)")
    p.add_argument("--out-png", default=str(Path(__file__).resolve().parent / "roofline.png"))
    p.add_argument("--out-md", default=str(Path(__file__).resolve().parent / "roofline.md"))
    return p.parse_args()


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Model-derived quantities
# ---------------------------------------------------------------------------

def analyse_model(args):
    model = KeywordSpottingCNN(num_classes=NUM_CLASSES).to("cpu")
    model.eval()
    input_shape = (1, 1, N_MELS, args.time_frames) if hasattr(args, "time_frames") \
        else (1, 1, N_MELS, 101)
    result = summary(model, input_size=input_shape, device="cpu", verbose=0)

    layers = []
    for li in result.summary_list:
        if not (li.is_leaf_layer and li.macs > 0):
            continue
        layers.append({
            "name": li.get_layer_name(show_var_name=True, show_depth=True),
            "macs": li.macs,
            "in_elems": numel(li.input_size),
            "out_elems": numel(li.output_size),
            "weight_elems": li.num_params,
        })
    layers.sort(key=lambda d: d["macs"], reverse=True)

    # Activation SRAM needed for the fused dataflow: the largest pair of
    # consecutive live tensors (produce into one buffer while reading another).
    acts = sorted({ly["out_elems"] for ly in layers}, reverse=True)
    peak_pair_elems = sum(acts[:2]) if len(acts) >= 2 else (acts[0] if acts else 0)

    return model, result, layers, input_shape, peak_pair_elems


def dram_bytes(args, layers, input_shape, total_weight_elems):
    """DRAM traffic per batch under the selected dataflow."""
    b = args.bytes_per_elem
    n = args.batch_size

    if args.dataflow == "none":
        # Every operand re-fetched, every layer, every sample.
        elems = sum(ly["in_elems"] + ly["out_elems"] + ly["weight_elems"]
                    for ly in layers) * n
        note = "all operands from DRAM every layer, no reuse"
    elif args.dataflow == "ws":
        # Weights loaded once for the whole batch; activations still stream.
        elems = sum(ly["in_elems"] + ly["out_elems"] for ly in layers) * n \
            + total_weight_elems
        note = (f"weights loaded once and held in the array; activations stream "
                f"to/from DRAM (batch {n})")
    else:  # fused
        # Only the mel input, the logits and the weights cross the interface.
        elems = (numel(input_shape) + NUM_CLASSES) * n + total_weight_elems
        note = (f"weights resident in the array; activations stay in on-chip "
                f"SRAM between layers (batch {n})")

    return elems * b, note


# ---------------------------------------------------------------------------
# Measured host point
# ---------------------------------------------------------------------------

def measure_cpu_gflops(model, input_shape, total_flops, iters):
    if iters <= 0:
        return None, None
    x = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(max(iters // 5, 5)):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    times.sort()
    median_s = times[len(times) // 2]
    return total_flops / median_s / 1e9, median_s * 1000.0


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def flops_fmt(v, _pos):
    if v >= 1e3:
        return f"{v / 1e3:g} TFLOP/s"
    if v >= 1:
        return f"{v:g} GFLOP/s"
    return f"{v * 1e3:g} MFLOP/s"


def make_plot(args, ctx):
    cpu_peak = args.cpu_gflops
    chip_peak = ctx["chip_peak_gflops"]
    cpu_ridge = cpu_peak / args.cpu_bw
    chip_ridge = chip_peak / args.chiplet_bw
    ai = ctx["ai"]

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    fig.patch.set_facecolor("#F7F7FA")

    ai_min, ai_max = 1e-2, 3e3
    xs = [ai_min * (ai_max / ai_min) ** (i / 600.0) for i in range(601)]

    ax.plot(xs, [min(cpu_peak, args.cpu_bw * x) for x in xs],
            color="#1F3FCC", lw=2.6, zorder=3,
            label=f"CPU Roofline ({args.cpu_name})")
    ax.plot(xs, [min(chip_peak, args.chiplet_bw * x) for x in xs],
            color="#1B6B2E", lw=2.6, zorder=3,
            label=f"Co-processor Roofline ({chip_peak/1e3:.3f} TFLOP/s, "
                  f"{ctx['array_r']}x{ctx['array_c']} @ {args.freq_mhz:g} MHz)")

    ax.axhline(cpu_peak, color="#1F3FCC", ls="--", lw=1.0, alpha=0.55, zorder=2)
    ax.annotate(f"CPU Peak: {cpu_peak:.1f} GFLOP/s", (ai_min * 2.2, cpu_peak * 1.12),
                fontsize=9, color="#1F3FCC")
    ax.axhline(chip_peak, color="#1B6B2E", ls="--", lw=1.0, alpha=0.55, zorder=2)

    ax.axvline(cpu_ridge, color="#7A86E8", ls=":", lw=1.3, zorder=2)
    ax.axvline(chip_ridge, color="#4E9E63", ls=":", lw=1.3, zorder=2)
    ax.axvline(ai, color="#666666", ls=":", lw=1.3, zorder=2)

    box = dict(boxstyle="round,pad=0.45", fc="white", ec="#7A86E8", alpha=0.95)
    ax.annotate(f"CPU Ridge\n{cpu_ridge:.1f} FLOP/B", xy=(cpu_ridge, cpu_peak * 0.30),
                xytext=(cpu_ridge * 3.0, cpu_peak * 0.10), fontsize=9, color="#1F3FCC",
                bbox=box, arrowprops=dict(arrowstyle="->", color="#1F3FCC", lw=1.1))
    ax.annotate(f"Co-proc Ridge\n{chip_ridge:.1f} FLOP/B", xy=(chip_ridge, chip_peak * 0.55),
                xytext=(chip_ridge * 0.10, chip_peak * 0.22), fontsize=9, color="#1B6B2E",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#4E9E63", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#1B6B2E", lw=1.1))

    # --- plotted operating points ---
    if ctx["cpu_measured_gflops"]:
        ax.plot([ai], [ctx["cpu_measured_gflops"]], "D", color="#1F3FCC", ms=13, zorder=6,
                label=f"Host CPU: {ctx['cpu_measured_gflops']:.1f} GFLOP/s (measured)")
        ax.annotate(
            f"Host CPU (measured)\n{ctx['cpu_measured_gflops']:.1f} GFLOP/s\n"
            f"{ctx['cpu_ms']:.2f} ms / inference",
            xy=(ai, ctx["cpu_measured_gflops"]),
            xytext=(ai * 0.055, ctx["cpu_measured_gflops"] * 0.10), fontsize=9,
            color="#1F3FCC",
            bbox=dict(boxstyle="round,pad=0.5", fc="#EEF1FF", ec="#1F3FCC", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#1F3FCC", lw=1.2))

    pe_gflops = ctx["pe_gflops"]
    ax.plot([ai], [pe_gflops], "s", color="#E8641E", ms=12, zorder=6,
            label=f"Single PE: {pe_gflops:.2f} GFLOP/s @ {args.freq_mhz:g} MHz")
    ax.annotate(
        f"Single PE (1 MAC)\n{pe_gflops:.2f} GFLOP/s\n{args.freq_mhz:g} MHz, "
        f"weight-stationary",
        xy=(ai, pe_gflops), xytext=(ai * 2.6, pe_gflops * 0.16), fontsize=9,
        color="#B8480F",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3EA", ec="#E8641E", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#E8641E", lw=1.2))

    ax.plot([ai], [chip_peak], "*", color="#1B6B2E", ms=22, zorder=7,
            label=f"{ctx['array_r']}x{ctx['array_c']} array: "
                  f"{chip_peak/1e3:.3f} TFLOP/s")
    ax.annotate(
        f"{ctx['array_r']}x{ctx['array_c']} array ({ctx['n_pe']:,} PEs)\n"
        f"{chip_peak/1e3:.3f} TFLOP/s @ {args.freq_mhz:g} MHz\n"
        f"{ctx['latency_us']:.1f} us / inference\n"
        f"needs {ctx['required_bw']:.2f} GB/s of {args.chiplet_bw:.1f} GB/s",
        xy=(ai, chip_peak), xytext=(ai * 0.020, chip_peak * 0.42), fontsize=9,
        color="#14521F",
        bbox=dict(boxstyle="round,pad=0.5", fc="#E9F5EC", ec="#1B6B2E", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#1B6B2E", lw=1.2))

    if ctx["cpu_measured_gflops"]:
        ax.annotate("", xy=(ai, chip_peak), xytext=(ai, ctx["cpu_measured_gflops"]),
                    arrowprops=dict(arrowstyle="-|>", color="#1B6B2E", lw=2.6))
        ax.annotate(
            f"{ctx['kernel_speedup']:.1f}x kernel\n{ctx['system_speedup']:.2f}x system\n"
            f"(Amdahl, {args.accel_fraction:.0%})",
            xy=(ai * 1.35, chip_peak * 0.42), fontsize=10.5, color="#14521F",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#E9F5EC", ec="#1B6B2E", alpha=0.95))

    ax.annotate(f"AI = {ai:,.0f} FLOP/B", xy=(ai * 1.1, ai_min * 1.5),
                fontsize=9, color="#444444", rotation=90)

    ax.text(0.035, 0.055, "Memory-\nbound", transform=ax.transAxes, fontsize=11,
            color="#AAAAAA", style="italic", ha="center")
    ax.text(0.93, 0.055, "Compute-\nbound", transform=ax.transAxes, fontsize=11,
            color="#AAAAAA", style="italic", ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.05, chip_peak * 9)
    ax.yaxis.set_major_formatter(FuncFormatter(flops_fmt))
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance (FLOP/s)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Roofline -- KeywordSpottingCNN Conv2d Accelerator\n"
        f"{args.cpu_name} host + {chip_peak/1e3:.3f} TFLOP/s "
        f"{ctx['array_r']}x{ctx['array_c']} weight-stationary array | "
        f"{DTYPE_NAME[args.bytes_per_elem]} | dataflow: {args.dataflow}",
        fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.55)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Wrap the footnote: a single long line gets clipped at the figure edges.
    import textwrap
    footnote = "\n".join(textwrap.wrap(ctx["footnote"], width=155))
    fig.text(0.5, 0.010, footnote, ha="center", va="bottom",
             fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(args.out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.time_frames = 101
    torch.manual_seed(42)

    try:
        rows, cols = (int(v) for v in args.array.lower().split("x"))
    except ValueError:
        sys.exit(f"--array must look like 16x16, got {args.array!r}")

    model, result, layers, input_shape, peak_pair_elems = analyse_model(args)

    total_macs = sum(ly["macs"] for ly in layers)
    total_weight_elems = sum(ly["weight_elems"] for ly in layers)
    flops_per_inf = total_macs * 2
    total_flops = flops_per_inf * args.batch_size

    byts, dataflow_note = dram_bytes(args, layers, input_shape, total_weight_elems)
    ai = total_flops / byts

    n_pe = rows * cols
    pe_gflops = 2 * args.freq_mhz * 1e6 / 1e9          # 1 MAC = 2 FLOP per cycle
    chip_peak_gflops = pe_gflops * n_pe
    chip_ridge = chip_peak_gflops / args.chiplet_bw
    cpu_ridge = args.cpu_gflops / args.cpu_bw

    # Attainable performance is the roofline minimum, not the peak.
    chip_attainable = min(chip_peak_gflops, args.chiplet_bw * ai)
    latency_us = flops_per_inf / (chip_attainable * 1e9) * 1e6
    required_bw = chip_peak_gflops / ai

    cpu_gflops_meas, cpu_ms = measure_cpu_gflops(
        model, input_shape, flops_per_inf, args.measure_iters)

    kernel_speedup = (chip_attainable / cpu_gflops_meas) if cpu_gflops_meas else None
    if kernel_speedup:
        f = args.accel_fraction
        system_speedup = 1.0 / ((1 - f) + f / kernel_speedup)
    else:
        system_speedup = None

    sram_kib = peak_pair_elems * args.bytes_per_elem / 1024
    weight_kib = total_weight_elems * args.bytes_per_elem / 1024
    dtype = DTYPE_NAME[args.bytes_per_elem]

    footnote = (
        f"AI = {ai:,.1f} FLOP/byte ({dtype}, dataflow '{args.dataflow}': {dataflow_note}). "
        f"Model: {flops_per_inf/1e6:.1f} MFLOP/inference, {result.total_params:,} params. "
        f"Array: {n_pe:,} PEs @ {args.freq_mhz:g} MHz = {chip_peak_gflops/1e3:.3f} TFLOP/s peak. "
        f"On-chip SRAM required: {sram_kib:.0f} KiB activations + {weight_kib:.0f} KiB weights."
    )

    ctx = {
        "ai": ai, "array_r": rows, "array_c": cols, "n_pe": n_pe,
        "pe_gflops": pe_gflops, "chip_peak_gflops": chip_peak_gflops,
        "cpu_measured_gflops": cpu_gflops_meas, "cpu_ms": cpu_ms,
        "kernel_speedup": kernel_speedup, "system_speedup": system_speedup,
        "latency_us": latency_us, "required_bw": required_bw, "footnote": footnote,
    }
    make_plot(args, ctx)

    bound = "compute-bound" if ai >= chip_ridge else "memory-bound"
    md = [
        "# Roofline Analysis -- Keyword Spotting CNN",
        "",
        f"**Model:** `KeywordSpottingCNN` ({result.total_params:,} params, "
        f"{NUM_CLASSES} classes)  ",
        f"**Input:** `{list(input_shape)}` (batch, channel, n_mels, time frames)  ",
        f"**Precision:** {dtype} ({args.bytes_per_elem} byte/element)  ",
        f"**Dataflow:** `{args.dataflow}` -- {dataflow_note}",
        "",
        f"![Roofline]({Path(args.out_png).name})",
        "",
        "## Platforms",
        "",
        "> Peak numbers are **hypotheses** except where marked measured. Override with "
        "`--cpu-gflops`, `--cpu-bw`, `--array`, `--freq-mhz`, `--chiplet-bw`.",
        "",
        "| Platform | Peak | Bandwidth | Ridge point |",
        "| -------- | ---- | --------- | ----------- |",
        f"| {args.cpu_name} (host) | {args.cpu_gflops:,.1f} GFLOP/s | "
        f"{args.cpu_bw:,.1f} GB/s | {cpu_ridge:.1f} FLOP/byte |",
        f"| Chiplet {rows}x{cols} @ {args.freq_mhz:g} MHz | "
        f"{chip_peak_gflops:,.1f} GFLOP/s ({chip_peak_gflops/1e3:.3f} TFLOP/s) | "
        f"{args.chiplet_bw:,.1f} GB/s | {chip_ridge:,.1f} FLOP/byte |",
        "",
        "## Workload",
        "",
        "| Quantity | Value |",
        "| -------- | ----- |",
        f"| MACs per inference | {total_macs:,} |",
        f"| FLOPs per inference | {flops_per_inf:,} ({flops_per_inf/1e6:.1f} MFLOP) |",
        f"| DRAM bytes per batch | {byts:,} |",
        f"| **Arithmetic intensity** | **{ai:,.2f} FLOP/byte** |",
        f"| Bound on chiplet | **{bound}** (AI {ai:,.1f} vs ridge {chip_ridge:,.1f}) |",
        f"| Attainable on chiplet | {chip_attainable:,.1f} GFLOP/s |",
        f"| Latency per inference | {latency_us:.1f} us |",
        "",
        "## Interface Bandwidth",
        "",
        "```",
        "Required BW = chiplet peak / arithmetic intensity",
        f"            = {chip_peak_gflops:.1f} GFLOP/s / {ai:,.2f} FLOP/byte",
        f"            = {required_bw:.2f} GB/s",
        "```",
        "",
        f"Provided: {args.chiplet_bw:.1f} GB/s "
        f"({args.chiplet_bw / required_bw:.2f}x margin).",
        "",
        "## On-Chip SRAM Required",
        "",
        "| Buffer | Size |",
        "| ------ | ---- |",
        f"| Weights (all layers, resident) | {weight_kib:.1f} KiB |",
        f"| Activations (largest live pair) | {sram_kib:.1f} KiB |",
        f"| **Total** | **{weight_kib + sram_kib:.1f} KiB** |",
        "",
    ]

    if cpu_gflops_meas:
        md += [
            "## Speedup",
            "",
            "| Quantity | Value |",
            "| -------- | ----- |",
            f"| Host CPU achieved (measured, batch 1) | {cpu_gflops_meas:.1f} GFLOP/s "
            f"({cpu_ms:.2f} ms/inference) |",
            f"| Chiplet attainable | {chip_attainable:,.1f} GFLOP/s |",
            f"| Kernel speedup | {kernel_speedup:.1f}x |",
            f"| Accelerated fraction (Amdahl) | {args.accel_fraction:.0%} |",
            f"| **System speedup** | **{system_speedup:.2f}x** |",
            "",
            "```",
            f"Speedup_system = 1 / ((1 - {args.accel_fraction:.3f}) + "
            f"{args.accel_fraction:.3f} / {kernel_speedup:.1f})",
            f"               = {system_speedup:.2f}x",
            "```",
            "",
        ]

    md += [
        "## Dataflow Comparison",
        "",
        "Same model, same precision -- only the DRAM traffic model changes:",
        "",
        "| Dataflow | DRAM bytes | AI (FLOP/byte) |",
        "| -------- | ---------- | -------------- |",
    ]
    for df in ("none", "ws", "fused"):
        saved = args.dataflow
        args.dataflow = df
        b_df, _ = dram_bytes(args, layers, input_shape, total_weight_elems)
        args.dataflow = saved
        md.append(f"| `{df}` | {b_df:,} | {total_flops / b_df:,.2f} |")

    md += [
        "",
        "Weight-stationary alone moves the needle very little at batch 1: the weights",
        f"are only {weight_kib:.0f} KiB of the traffic, while the activations are the bulk.",
        "The large gain comes from keeping activations on-chip between layers (`fused`).",
        "",
        "## Precision Note",
        "",
        "Operation count is fixed by the architecture, so AI scales inversely with bytes",
        "per element: FP32 -> INT8 divides DRAM traffic by 4 and multiplies AI by 4.",
        "Re-run with `--bytes-per-elem 1` for the INT8 roofline.",
    ]

    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_md}")
    print(f"\nDataflow '{args.dataflow}' | {dtype} | AI = {ai:,.2f} FLOP/byte")
    print(f"Chiplet {rows}x{cols} @ {args.freq_mhz:g} MHz = "
          f"{chip_peak_gflops:,.1f} GFLOP/s peak, ridge {chip_ridge:,.1f} -> {bound}")
    print(f"Attainable {chip_attainable:,.1f} GFLOP/s, {latency_us:.1f} us/inference")
    if cpu_gflops_meas:
        print(f"CPU measured {cpu_gflops_meas:.1f} GFLOP/s -> "
              f"{kernel_speedup:.1f}x kernel, {system_speedup:.2f}x system")


if __name__ == "__main__":
    main()
