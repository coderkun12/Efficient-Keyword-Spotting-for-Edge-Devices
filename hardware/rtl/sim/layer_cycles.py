"""
Cycle cost of the INTEGRATED accelerator (layer_top), calibrated against the
RTL rather than assumed.

rtl/sim/tile_schedule.py models an idealised array. This models what the real
sequencer does, including the costs that only appear once band_sram, the row
accumulator and the write-back are wired together behind one FSM.

CALIBRATION
The per-tile cost was measured in simulation by timing row_go to busy-low:

    W= 16 ktiles=2 -> 122 cycles        W= 40 ktiles=3 -> 221 cycles
    W= 24 ktiles=2 -> 138 cycles        W= 48 ktiles=4 -> 304 cycles
    W= 40 ktiles=2 -> 178 cycles

Differencing the two W=40 points gives 43 cycles per additional k-tile at
W=40, i.e. W+3. Below W = 2K the sweep is too short to hide the weight load and
the cost floors at about 2K+3.

WHY 2K AND NOT K
The commit that swaps weight tiles is skewed down the array, so row k copies
shadow to active at commit+k. The shadows therefore have to stay untouched for
K cycles after a commit before the next tile's K-cycle shift may start. That is
2K cycles of weight machinery per tile, which hides completely inside a sweep
only when W >= 2K. conv2 (W=101) and conv3 (W=50) hide it; conv4 (W=25) does
not, and stalls a few cycles per tile.

    python rtl/sim/layer_cycles.py
"""

import math

K = M = 16
PIPE = 1
LAT = PIPE * K + M - 1
FREQ_HZ = 500e6

HOST_MS = 8.32          # i5-12450H, batch 1, 1 thread, median
F_FEATURE = 0.9774      # conv + pool + BatchNorm + ReLU share of host runtime

CONVS = [
    ("conv1", 1, 32, 40, 101),
    ("conv2", 32, 64, 40, 101),
    ("conv3", 64, 128, 20, 50),
    ("conv4", 128, 128, 10, 25),
]

IDEAL_CYCLES = 186_220_800 // (K * M)   # every PE busy every cycle


def per_tile(width):
    """Steady-state cycles per k-tile sweep, from the calibration above."""
    return max(width + 3, 2 * K + 3)


def per_row(width, ktiles):
    """row_go to busy-low: startup, the tile sweeps, the drain, the flush."""
    startup = K + 2
    drain = LAT + 3
    return startup + ktiles * per_tile(width) + drain + width


def main():
    print(f"Integrated sequencer, {K}x{M} INT8 @ {FREQ_HZ/1e6:.0f} MHz, "
          f"PIPE={PIPE}\n")
    print(f"{'layer':8}{'W':>5}{'ktiles':>8}{'mtiles':>8}{'rows':>6}"
          f"{'cyc/tile':>10}{'cyc/row':>10}{'cycles':>12}")
    print("-" * 67)

    total = 0
    for name, in_ch, out_ch, out_h, out_w in CONVS:
        ktiles = math.ceil(in_ch * 9 / K)
        mtiles = math.ceil(out_ch / M)
        cyc_row = per_row(out_w, ktiles)
        cycles = mtiles * out_h * cyc_row
        total += cycles
        print(f"{name:8}{out_w:5d}{ktiles:8d}{mtiles:8d}{out_h:6d}"
              f"{per_tile(out_w):10d}{cyc_row:10d}{cycles:12,}")

    print("-" * 67)
    ms = total / FREQ_HZ * 1e3
    print(f"{'TOTAL':8}{'':37}{total:12,}  = {ms:.3f} ms")
    print(f"\narray utilisation : {IDEAL_CYCLES/total:.1%} "
          f"({IDEAL_CYCLES:,} ideal cycles)")

    rest = HOST_MS * (1 - F_FEATURE)
    system = HOST_MS / (ms + rest)
    print(f"host remainder    : {rest:.3f} ms  (the 2.26% not accelerated)")
    print(f"total latency     : {ms + rest:.3f} ms")
    print(f"SYSTEM SPEEDUP    : {system:.2f}x  against {HOST_MS:.2f} ms on the host")

    print("\nWhere the remaining overhead sits:")
    for name, in_ch, out_ch, out_h, out_w in CONVS:
        ktiles = math.ceil(in_ch * 9 / K)
        useful = ktiles * out_w
        print(f"  {name}: sweep {out_w:3d} of {per_tile(out_w):3d} cycles per "
              f"tile = {useful/(ktiles*per_tile(out_w)):5.1%} of the tile loop")
    print("\nconv4 is the weak layer: W=25 is below 2K=32, so its weight load "
          "cannot\nfully hide inside the sweep. Widening the sweep by "
          "processing two output\nrows per pass, or halving the commit skew, "
          "is where the next gain is.")


if __name__ == "__main__":
    main()
