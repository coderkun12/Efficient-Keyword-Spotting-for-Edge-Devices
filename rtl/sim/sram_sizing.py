"""On-chip memory sizing for the M3 banded scratchpad.

Reproduces every number in rtl/SRAM_SIZING.md. Run it after any change to the
model geometry or the array shape.

    python rtl/sim/sram_sizing.py
"""

# (name, in_channels, out_channels, out_h, out_w) for KeywordSpottingCNN
CONVS = [
    ("conv1", 1, 32, 40, 101),
    ("conv2", 32, 64, 40, 101),
    ("conv3", 64, 128, 20, 50),
    ("conv4", 128, 128, 10, 25),
]

ROWS = 3        # rows live for a 3x3 kernel
OUT_ROWS = 2    # rows a 2x2 max-pool consumes
BANKS = 2       # double buffering
K = M = 16      # array shape
ACC_W = 32


def main():
    print(f"{'layer':8}{'in band/row':>14}{'out band/row':>14}"
          f"{'full in map':>14}{'full out map':>14}")
    print("-" * 64)
    max_in = max_out = 0
    maps = []
    for name, ci, co, h, w in CONVS:
        in_band, out_band = ci * w, co * w
        max_in, max_out = max(max_in, in_band), max(max_out, out_band)
        maps.append((ci * h * w, co * h * w))
        print(f"{name:8}{in_band:14,}{out_band:14,}{ci*h*w:14,}{co*h*w:14,}")

    inp = max_in * ROWS * BANKS
    out = max_out * OUT_ROWS * BANKS
    wgt = K * M
    total = inp + out + wgt

    print(f"\n{'Buffer':28}{'bytes':>10}")
    print("-" * 38)
    print(f"{'input band  (3 rows x 2)':28}{inp:10,}")
    print(f"{'output band (2 rows x 2)':28}{out:10,}")
    print(f"{'weight staging':28}{wgt:10,}")
    print("-" * 38)
    print(f"{'TOTAL':28}{total:10,}  = {total/1024:.1f} KiB")

    # Naive alternative: hold whole feature maps, largest live pair.
    pairs = [maps[i][1] + maps[i + 1][1] for i in range(len(maps) - 1)]
    naive = max(max(pairs), maps[0][0] + maps[0][1])
    print(f"\nnaive largest live pair : {naive:,} B = {naive/1024:.1f} KiB")
    print(f"reduction               : {naive/total:.1f}x")

    regs = [
        ("array input skew", 8 * K * (K - 1) // 2),
        ("array switch skew", 1 * K * (K - 1) // 2),
        ("array output deskew", ACC_W * M * (M - 1) // 2),
        ("PE shadow + active", K * M * 16),
        ("band window", 3 * ROWS * 3 * 8),
    ]
    print(f"\n{'Registers (not SRAM)':28}{'flops':>8}{'bytes':>9}")
    print("-" * 45)
    for label, flops in regs:
        print(f"{label:28}{flops:8,}{flops/8:9,.0f}")
    tf = sum(f for _, f in regs)
    print("-" * 45)
    print(f"{'TOTAL':28}{tf:8,}{tf/8:9,.0f}")
    print(f"\noutput deskew is {ACC_W * M * (M-1) // 2 / tf:.0%} of array registers "
          f"-- the first thing to revisit if synthesis says register-bound.")


if __name__ == "__main__":
    main()
