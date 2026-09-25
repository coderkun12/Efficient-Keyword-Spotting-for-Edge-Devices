"""
Tile schedule and array utilisation model for the M2 accelerator.

ACCELERATOR_PLAN.md lists this as an M2 deliverable: "Model it in M2 simulation
before committing to the M4 array size." The roofline projection assumes 100%
array utilisation, which no real tiling schedule achieves. This works out the
real number from the RTL's actual timing behaviour.

Per weight tile the array spends:
    K cycles    shifting the weight tile in (all M columns in parallel)
    N cycles    streaming activation vectors, one result per cycle
    K+M-1       draining the systolic pipeline before the next tile's weights
                can be shifted in without corrupting in-flight diagonals

Those K and K+M-1 terms are pure overhead, and they hurt most where N is small.

    python rtl/sim/tile_schedule.py
    python rtl/sim/tile_schedule.py --array 32x32 --freq-mhz 500
    python rtl/sim/tile_schedule.py --double-buffer     # model the M3 optimisation
"""

import argparse
import math

# (name, in_channels, out_channels, out_h, out_w) for KeywordSpottingCNN.
# Matches Profiling/kernel_analysis.md; conv1 is the only badly shaped layer.
CONVS = [
    ("conv1", 1, 32, 40, 101),
    ("conv2", 32, 64, 40, 101),
    ("conv3", 64, 128, 20, 50),
    ("conv4", 128, 128, 10, 25),
]


def parse_args():
    p = argparse.ArgumentParser(description="Array utilisation from the tile schedule")
    p.add_argument("--array", default="16x16", help="K x M, e.g. 16x16 or 32x32")
    p.add_argument("--freq-mhz", type=float, default=500.0)
    p.add_argument("--double-buffer", action="store_true",
                   help="model shadow weight registers with a skewed switch: "
                        "removes both the load bubble and the inter-tile drain")
    return p.parse_args()


def model(k_arr, m_arr, freq_hz, double_buffer=False):
    latency = k_arr + m_arr - 1
    rows = []
    total = ideal_total = macs_total = 0

    for name, in_ch, out_ch, out_h, out_w in CONVS:
        m_dim = out_ch
        k_dim = in_ch * 9
        n_dim = out_h * out_w

        tiles = math.ceil(m_dim / m_arr) * math.ceil(k_dim / k_arr)
        if double_buffer:
            # Weights for tile i+1 shift into shadow registers while tile i
            # streams, and the switch is skewed row by row to match the data
            # wavefront, so consecutive tiles run back to back.
            per_tile = n_dim
            cycles = tiles * per_tile + latency      # one drain for the whole layer
        else:
            per_tile = k_arr + n_dim + latency
            cycles = tiles * per_tile

        macs = m_dim * k_dim * n_dim
        ideal = macs / (k_arr * m_arr)

        rows.append((name, m_dim, k_dim, n_dim, tiles, per_tile, cycles, ideal))
        total += cycles
        ideal_total += ideal
        macs_total += macs

    return rows, total, ideal_total, macs_total, latency


def main():
    args = parse_args()
    k_arr, m_arr = (int(v) for v in args.array.lower().split("x"))
    freq_hz = args.freq_mhz * 1e6

    rows, total, ideal_total, macs_total, latency = model(
        k_arr, m_arr, freq_hz, args.double_buffer
    )

    mode = "double-buffered weights" if args.double_buffer else "M2 RTL as built"
    print(f"Array {k_arr}x{m_arr} @ {args.freq_mhz:g} MHz  |  {mode}")
    print(f"Pipeline drain latency K+M-1 = {latency} cycles, "
          f"weight load = {k_arr} cycles per tile\n")
    print(f"{'layer':8}{'M':>5}{'K':>6}{'N':>6}{'tiles':>7}"
          f"{'cyc/tile':>10}{'cycles':>11}{'ideal':>11}{'util':>8}")
    print("-" * 72)
    for name, m_dim, k_dim, n_dim, tiles, per_tile, cycles, ideal in rows:
        print(f"{name:8}{m_dim:5d}{k_dim:6d}{n_dim:6d}{tiles:7d}"
              f"{per_tile:10d}{cycles:11,d}{ideal:11,.0f}{ideal / cycles:7.1%}")
    print("-" * 72)
    print(f"{'TOTAL':8}{'':24}{total:11,d}{ideal_total:11,.0f}"
          f"{ideal_total / total:7.1%}")

    peak_gmacs = k_arr * m_arr * freq_hz / 1e9
    print(f"\nMACs per inference   : {macs_total:,}")
    print(f"Array peak           : {peak_gmacs:.0f} GMAC/s "
          f"({2 * peak_gmacs:.0f} GOP/s)")
    print(f"Latency              : {total / freq_hz * 1e3:.3f} ms")
    print(f"Effective throughput : {macs_total / (total / freq_hz) / 1e9:.1f} GMAC/s")
    print(f"Overhead             : {(total - ideal_total) / total:.1%}")

    if not args.double_buffer:
        bubbles = sum(
            math.ceil(oc / m_arr) * math.ceil(ic * 9 / k_arr)
            for _, ic, oc, _, _ in CONVS
        ) * k_arr
        print(f"  of which weight-load bubbles : {bubbles / total:5.1%}")
        print(f"  of which pipeline drains     : {(total - ideal_total - bubbles) / total:5.1%}")
        print("\nconv4 is the worst layer: 576 tiles of only N=250 vectors each, so "
              "the\nfixed per-tile overhead is amortised over very little work. "
              "Re-run with\n--double-buffer to see what the M3 shadow-register "
              "optimisation recovers.")


if __name__ == "__main__":
    main()
