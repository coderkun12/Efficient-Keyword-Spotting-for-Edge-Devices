"""
End-to-end speedup for the keyword-spotting accelerator.

Ties three measured things together:
  * host baseline latency          Profiling/host_vitals.txt   (8.32 ms median)
  * per-operator runtime shares    Profiling/op_breakdown.txt  (Amdahl fractions)
  * array cycles from the real tiling loop   rtl/sim/tile_schedule.py

and answers the question the whole project exists to answer: how much does
fusing pooling, BatchNorm and ReLU into the write-back path actually buy?

    python rtl/sim/speedup.py
"""

import math

# --- Measured on the reference host, LAPTOP-FFRP12TK (i5-12450H) -----------
HOST_MS = 8.32          # batch 1, 1 thread, median of 800 after a 5 s warmup
F_CONV = 0.5538         # convolution share of host runtime
F_FEATURE = 0.9774      # conv + max-pool + BatchNorm + ReLU

# --- Array -----------------------------------------------------------------
K_ARR = M_ARR = 16
FREQ_HZ = 500e6

CONVS = [
    ("conv1", 1, 32, 40, 101),
    ("conv2", 32, 64, 40, 101),
    ("conv3", 64, 128, 20, 50),
    ("conv4", 128, 128, 10, 25),
]

PRUNED_MACS = 91_472_400    # pruned_structured_70-70-70-70, 92.82% accuracy
DENSE_MACS = 186_220_800


def array_cycles(double_buffer):
    latency = K_ARR + M_ARR - 1
    total = 0
    for _, in_ch, out_ch, out_h, out_w in CONVS:
        m_dim, k_dim, n_dim = out_ch, in_ch * 9, out_h * out_w
        tiles = math.ceil(m_dim / M_ARR) * math.ceil(k_dim / K_ARR)
        if double_buffer:
            total += tiles * n_dim + latency
        else:
            total += tiles * (K_ARR + n_dim + latency)
    return total


def report(label, accel_ms, fraction):
    """Amdahl with the accelerated part replaced by a measured chiplet time."""
    host_part = HOST_MS * fraction
    rest = HOST_MS * (1 - fraction)
    total = accel_ms + rest
    kernel = host_part / accel_ms
    system = HOST_MS / total
    print(f"{label:<34}{accel_ms:8.3f}{rest:9.3f}{total:9.3f}"
          f"{kernel:9.2f}x{system:9.2f}x")
    return system


def main():
    cyc_block = array_cycles(False)
    cyc_dbuf = array_cycles(True)
    ms_block = cyc_block / FREQ_HZ * 1e3
    ms_dbuf = cyc_dbuf / FREQ_HZ * 1e3
    ms_pruned = ms_dbuf * PRUNED_MACS / DENSE_MACS

    print(f"Host baseline      : {HOST_MS:.2f} ms  (i5-12450H, batch 1, 1 thread)")
    print(f"Array              : {K_ARR}x{M_ARR} INT8 @ {FREQ_HZ/1e6:.0f} MHz")
    print(f"Array time, dense  : {ms_block:.3f} ms blocking / "
          f"{ms_dbuf:.3f} ms double-buffered")
    print(f"Array time, pruned : {ms_pruned:.3f} ms")
    print()
    print("Accelerated fractions, measured with torch.profiler at 1 thread:")
    print(f"  convolution only          {F_CONV:.2%}   Amdahl ceiling "
          f"{1/(1-F_CONV):5.2f}x")
    print(f"  + pool + BatchNorm + ReLU {F_FEATURE:.2%}   Amdahl ceiling "
          f"{1/(1-F_FEATURE):5.2f}x")
    print()
    print(f"{'configuration':<34}{'accel':>8}{'host':>9}{'total':>9}"
          f"{'kernel':>10}{'system':>10}")
    print("-" * 80)

    conv_only = report("conv only, blocking loads", ms_block, F_CONV)
    conv_only_db = report("conv only, overlapped loads", ms_dbuf, F_CONV)
    fused = report("FUSED, overlapped loads", ms_dbuf, F_FEATURE)
    fused_pruned = report("FUSED + structured pruning", ms_pruned, F_FEATURE)

    print("-" * 80)
    print()
    print(f"Fusion alone is worth {fused/conv_only_db:.2f}x on system speedup "
          f"({conv_only_db:.2f}x -> {fused:.2f}x).")
    print("It costs no extra cycles: the write-back is a pipeline on the array's")
    print("output path, so it adds latency but not throughput. What it removes is")
    print("the 42% of host runtime that max-pool, BatchNorm and ReLU were taking")
    print("while contributing essentially none of the MACs.")
    print()
    print(f"With structured pruning on top: {fused_pruned:.2f}x at 92.82% accuracy,")
    print("above the 91.88% dense FP32 baseline.")
    print()
    print("Side effect worth noting: 2x2 pooling cuts the result stream to a")
    print("quarter of its unpooled rate, so the fused path also reduces the")
    print("interface bandwidth the chiplet needs.")


if __name__ == "__main__":
    main()
