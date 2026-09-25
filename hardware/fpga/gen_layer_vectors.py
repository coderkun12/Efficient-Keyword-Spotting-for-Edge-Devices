"""
Generate the on-chip vectors for the stage C4 layer self-test.

WHAT THIS TARGETS
conv4, m-tile 0: output channels 0-15 of a 128 -> 128 layer, W=25, H=10, all
72 k-tiles accumulated, with BatchNorm folded into the requantiser, ReLU, and
2x2 max-pool fused into the write-back. That is one complete m-tile of a real
layer -- the whole fused datapath, not a slice of it.

WHY conv4 AND NOT conv2
rtl/sim/layer_cycles.py identifies conv4 as the weak layer: W=25 is below
2K=32, so its weight load cannot fully hide inside the sweep. Measuring the
layer that is hardest for this microarchitecture is worth more than measuring
the one most likely to look good. It is also the smallest band (25 columns),
which keeps the harness ROMs modest.

THE GOLDEN DATA COMES FROM ref_model.py
Not from a second implementation written here. conv_layer and
fused_writeback are the exact functions the 8 passing cocotb tests check
layer_top against, so if the board disagrees with this file, the board is
wrong -- there is no third model to arbitrate.

OUTPUTS (plain binary, one word per line, for $readmemb)
    layer_w.txt     18,432 x 8   weights in HOST ADDRESS ORDER
                                 kt*K*M + m*K + k, so the harness just counts
    layer_a.txt     32,000 x 8   feature map, flattened c*H*W + y*W + x
    layer_c.txt         16 x 50  requant config: {bias[27:0], mult[15:0], shift[5:0]}
    layer_g.txt         65 x 128 golden INT8 output vectors, M per word

    python fpga/gen_layer_vectors.py
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # hardware/
REPO_ROOT = ROOT.parent                                # repo root
sys.path.insert(0, str(ROOT / "rtl" / "tb"))
from ref_model import conv_layer, fused_writeback  # noqa: E402

# The repo root, not fpga/: the import below is by package path, so the
# directory that CONTAINS hardware/ has to be importable.
sys.path.insert(0, str(REPO_ROOT))
from hardware.fpga.gen_test_vectors import find_weight_source, write_mem  # noqa: E402

# Geometry of conv4 and of the array.
K, M       = 16, 16
ACC_W      = 28
MULT_W     = 16
CHANNELS   = 128          # input channels
HEIGHT     = 10
WIDTH      = 25
TAPS       = CHANNELS * 9  # 1152
KTILES     = -(-TAPS // K)  # 72


def parse_args():
    p = argparse.ArgumentParser(description="Vectors for the layer self-test")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--rows", type=int, default=HEIGHT,
                   help="output rows to generate; fewer shrinks the ROMs")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent))
    return p.parse_args()


def main():
    a = parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    height = a.rows

    # ---- Weights: real conv4 tile if the checkpoint is there --------------
    import numpy as np
    src = find_weight_source(REPO_ROOT, "conv4")
    if src is not None:
        flat, note = src
        scale = float(np.abs(flat).max()) / 127.0
        q = np.clip(np.round(flat / scale), -128, 127).astype(int)
        weights = [[int(q[m][t]) for t in range(TAPS)] for m in range(M)]
        print(f"weights   : REAL conv4 m-tile 0 from {note}, scale={scale:.6g}")
        print(f"            layer is {q.shape}, taking rows 0..{M-1}")
    else:
        weights = [[rng.randint(-8, 8) for _ in range(TAPS)] for _ in range(M)]
        print("weights   : random (no checkpoint found)")

    # ---- Activations ------------------------------------------------------
    # Non-negative: conv4's input is the output of conv3's ReLU, so a signed
    # range here would exercise a case the real layer never sees.
    fmap = [[[rng.randint(0, 127) for _ in range(WIDTH)]
             for _ in range(height)] for _ in range(CHANNELS)]
    print(f"activations: {CHANNELS}ch x {height} x {WIDTH}, post-ReLU range 0..127")

    # ---- Requantisation: BatchNorm folded in ------------------------------
    # Shifts are chosen so the accumulator lands inside INT8 rather than
    # saturating. A test whose every output clamps to +127 passes without
    # exercising the arithmetic -- that mistake was made once already on the
    # writeback unit tests.
    biases = [rng.randint(-2000, 2000) for _ in range(M)]
    mults  = [rng.randint(16384, 65535) for _ in range(M)]
    shifts = [rng.randint(26, 30) for _ in range(M)]

    # ---- Golden, from the SAME model the cocotb suite uses ---------------
    accs = conv_layer(fmap, weights, CHANNELS, height, WIDTH, M)
    want = fused_writeback(accs, biases, mults, shifts, height, WIDTH,
                           relu=True, pool=True)
    print(f"golden    : {len(want)} pooled output vectors "
          f"({height//2} rows x {WIDTH//2} cols)")

    sat = sum(1 for v in want for x in v if x in (127, -128))
    tot = sum(len(v) for v in want)
    print(f"            {sat}/{tot} values at the clamp "
          f"({sat/tot:.1%}) -- high means the shifts need raising")

    # ---- Pack -------------------------------------------------------------
    # Weights in host address order kt*K*M + m*K + k, so the harness FSM only
    # has to increment a counter.
    wbytes = []
    for kt in range(KTILES):
        for m in range(M):
            for k in range(K):
                tap = kt * K + k
                wbytes.append(weights[m][tap] & 0xFF if tap < TAPS else 0)

    abytes = [fmap[c][y][x] & 0xFF
              for c in range(CHANNELS) for y in range(height)
              for x in range(WIDTH)]

    cwords = []
    for m in range(M):
        w = ((biases[m] & ((1 << ACC_W) - 1)) << (MULT_W + 6)) \
            | ((mults[m] & ((1 << MULT_W) - 1)) << 6) \
            | (shifts[m] & 0x3F)
        cwords.append(w)

    gwords = []
    for vec in want:
        w = 0
        for m, v in enumerate(vec):
            w |= (int(v) & 0xFF) << (m * 8)
        gwords.append(w)

    write_mem(out / "layer_w.txt", 8, len(wbytes), wbytes)
    write_mem(out / "layer_a.txt", 8, len(abytes), abytes)
    write_mem(out / "layer_c.txt", ACC_W + MULT_W + 6, M, cwords)
    write_mem(out / "layer_g.txt", M * 8, len(gwords), gwords)

    bits = (len(wbytes) * 8 + len(abytes) * 8
            + M * (ACC_W + MULT_W + 6) + len(gwords) * M * 8)
    print()
    print(f"layer_w.txt {len(wbytes):7,} x   8 bit   weights")
    print(f"layer_a.txt {len(abytes):7,} x   8 bit   activations")
    print(f"layer_c.txt {M:7,} x {ACC_W+MULT_W+6:3d} bit   requant config")
    print(f"layer_g.txt {len(gwords):7,} x {M*8:3d} bit   golden output")
    print(f"total ROM   {bits:,} bits = {bits/8/1024:.1f} KiB "
          f"({bits/6635520:.2%} of the EP4CGX150's M9K)")
    print()
    print("harness parameters to match:")
    print(f"    NROWS  = {height}")
    print(f"    KTILES = {KTILES}")
    print(f"    NGOLD  = {len(gwords)}")
    print(f"    WIDTH  = {WIDTH}   CHANNELS = {CHANNELS}")


if __name__ == "__main__":
    main()
