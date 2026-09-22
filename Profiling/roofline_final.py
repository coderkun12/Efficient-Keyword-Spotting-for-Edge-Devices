"""
Final roofline: measured CPU host vs the SYNTHESIZED INT8 accelerator.

Every number here is measured, not projected. That is what separates this plot
from the earlier ones in this folder:

  CPU   peak          140.8 GFLOP/s   i5-12450H, one P-core: 8 AVX2 lanes
                                      x 2 FMA units x 2 flop x 4.4 GHz
  CPU   bandwidth      51.2 GB/s      2 x DDR4-3200, dual channel (confirmed
                                      from Profiling/host_vitals.txt)
  CPU   achieved        44.9 GFLOP/s  373.5 MFLOP / 8.32 ms, median of 800
  ACCEL peak           512 GOP/s      256 INT8 MACs x 2 x 1 GHz; 1 GHz CLOSES
                                      in Genus with +299 ps slack, 0 violations
  ACCEL achieved       426 GOP/s      83.2% array utilisation from the
                                      cycle-accurate sequencer model
  ACCEL bandwidth       1.6 GB/s      host interface assumption (only hypothesis
                                      left in the plot)

Writes Profiling/roofline_final.png.

    python Profiling/roofline_final.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- measured -------------------------------------------------------------
CPU_PEAK, CPU_BW, CPU_ACHIEVED, CPU_AI = 140.8, 51.2, 44.9, 44.14
ACC_PEAK, ACC_BW, ACC_ACHIEVED, ACC_AI = 512.0, 1.6, 426.0, 1515.0

CPU_RIDGE = CPU_PEAK / CPU_BW
ACC_RIDGE = ACC_PEAK / ACC_BW

XS = [10 ** (i / 40) for i in range(-80, 148)]


def fmt(v, _):
    if v >= 1000:
        return f"{v/1000:g} TOP/s"
    if v >= 1:
        return f"{v:g} GOP/s"
    return f"{v*1000:g} MOP/s"


fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set_xscale("log")
ax.set_yscale("log")

ax.plot(XS, [min(CPU_PEAK, CPU_BW * x) for x in XS], color="#1f4e9c", lw=2.6,
        label=f"CPU roofline — i5-12450H, 1 core ({CPU_PEAK:.1f} GFLOP/s, "
              f"{CPU_BW:.1f} GB/s)")
ax.plot(XS, [min(ACC_PEAK, ACC_BW * x) for x in XS], color="#1a7d3c", lw=2.6,
        label=f"Accelerator roofline — 16x16 INT8 @ 1 GHz "
              f"({ACC_PEAK:.0f} GOP/s, {ACC_BW:.1f} GB/s)")

ax.axhline(CPU_PEAK, color="#1f4e9c", ls="--", lw=1.0, alpha=0.55)
ax.axhline(ACC_PEAK, color="#1a7d3c", ls="--", lw=1.0, alpha=0.55)
ax.axvline(CPU_RIDGE, color="#1f4e9c", ls=":", lw=1.1, alpha=0.6)
ax.axvline(ACC_RIDGE, color="#1a7d3c", ls=":", lw=1.1, alpha=0.6)

ax.plot([CPU_AI], [CPU_ACHIEVED], "D", ms=13, color="#1f4e9c", zorder=6,
        label=f"KWS on CPU — {CPU_ACHIEVED:.1f} GFLOP/s measured (8.32 ms)")
ax.plot([ACC_AI], [ACC_ACHIEVED], "*", ms=24, color="#1a7d3c", zorder=6,
        label=f"KWS on accelerator — {ACC_ACHIEVED:.0f} GOP/s (0.874 ms)")

box = dict(boxstyle="round,pad=0.45", fc="#eaf0fa", ec="#1f4e9c", lw=1.2)
ax.annotate(f"CPU\nAI = {CPU_AI:.1f} FLOP/B\n{CPU_ACHIEVED:.1f} GFLOP/s\n"
            f"32% of its own peak\ncompute-bound",
            xy=(CPU_AI, CPU_ACHIEVED), xytext=(1.6, 0.55), fontsize=9.5,
            bbox=box, arrowprops=dict(arrowstyle="->", color="#1f4e9c"))

box = dict(boxstyle="round,pad=0.45", fc="#e8f6ec", ec="#1a7d3c", lw=1.2)
ax.annotate(f"ACCELERATOR\nAI = {ACC_AI:.0f} FLOP/B\n{ACC_ACHIEVED:.0f} GOP/s\n"
            f"83% array utilisation\ncompute-bound",
            xy=(ACC_AI, ACC_ACHIEVED), xytext=(230, 8), fontsize=9.5,
            bbox=box, arrowprops=dict(arrowstyle="->", color="#1a7d3c"))

ax.annotate("", xy=(ACC_AI, ACC_ACHIEVED), xytext=(ACC_AI, CPU_ACHIEVED),
            arrowprops=dict(arrowstyle="<->", color="#b03030", lw=2.4))
ax.text(ACC_AI * 1.25, (ACC_ACHIEVED * CPU_ACHIEVED) ** 0.5,
        f"9.5x\nthroughput\n\n7.8x latency\n893x energy",
        fontsize=10.5, color="#b03030", fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fdeaea", ec="#b03030", lw=1.2))

ax.text(CPU_RIDGE * 1.12, 0.16, f"CPU ridge\n{CPU_RIDGE:.2f} FLOP/B",
        fontsize=8.5, color="#1f4e9c",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f4e9c", alpha=0.9))
ax.text(ACC_RIDGE * 1.12, 0.16, f"accelerator ridge\n{ACC_RIDGE:.0f} FLOP/B",
        fontsize=8.5, color="#1a7d3c",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1a7d3c", alpha=0.9))

ax.text(0.02, 0.06, "memory-bound", transform=ax.transAxes, fontsize=11,
        color="#999999", style="italic")
ax.text(0.86, 0.06, "compute-bound", transform=ax.transAxes, fontsize=11,
        color="#999999", style="italic")

ax.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=12.5, fontweight="bold")
ax.set_ylabel("Performance", fontsize=12.5, fontweight="bold")
ax.set_title("Roofline — KeywordSpottingCNN\n"
             "measured CPU host vs synthesized 16x16 INT8 accelerator "
             "(SAED14nm, 1 GHz)",
             fontsize=13.5, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(fmt))
ax.grid(True, which="major", ls="-", alpha=0.22)
ax.grid(True, which="minor", ls=":", alpha=0.10)
ax.set_xlim(1e-2, 3e4)
ax.set_ylim(0.1, 3e3)
ax.legend(loc="upper left", fontsize=9.2, framealpha=0.95)

fig.text(0.5, 0.012,
         "Both platforms are COMPUTE-bound: each workload's arithmetic "
         "intensity sits far right of its ridge point, so extra bandwidth "
         "cannot help — only arithmetic can.\n"
         "Accelerator: 60,607 cells, 0.0405 mm2, 52.45 mW active, 17.5 uW "
         "leakage, +299 ps slack at 1 GHz, 0 violating paths. "
         "Bandwidth (1.6 GB/s) is the only remaining assumption.",
         ha="center", fontsize=8.6, color="#444444")

fig.tight_layout(rect=[0, 0.045, 1, 1])
out = "Profiling/roofline_final.png"
fig.savefig(out, dpi=155)
print(f"Wrote {out}")
print(f"  CPU         AI {CPU_AI:8.2f}  {CPU_ACHIEVED:7.1f} GFLOP/s  "
      f"ridge {CPU_RIDGE:6.2f}  -> compute-bound")
print(f"  accelerator AI {ACC_AI:8.0f}  {ACC_ACHIEVED:7.0f} GOP/s    "
      f"ridge {ACC_RIDGE:6.0f}  -> compute-bound")
print(f"  throughput ratio {ACC_ACHIEVED/CPU_ACHIEVED:.1f}x")
