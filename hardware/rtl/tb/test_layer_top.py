"""
cocotb tests for layer_top -- the integrated accelerator.

This is the first test that exercises the whole datapath together:

    band_sram  ->  mac_array  ->  row accumulator  ->  writeback  ->  INT8 out

and therefore the first that checks the pieces agree with each other rather
than each agreeing with its own reference in isolation. The golden model is a
plain Python 3x3 convolution followed by the same fused write-back model the
standalone tests use, so a bug has to appear in both to survive.

Checked here:
  I1  one output row, convolution only (pooling off), against the reference
  I2  multi-row layer with pooling on, end to end
  I3  k-tile accumulation: K > 16 taps must sum across tiles, not overwrite
  I4  ReLU and per-channel requantisation survive integration
  I5  back-to-back layers without a reset, which is the real tiling loop
  I6  no signal reads X or Z after a full layer
"""

import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware.rtl.tb.ref_model import (conv_layer, conv_row, fused_writeback,  # noqa: E402
                       to_signed)
from hardware.rtl.tb.xscan import assert_no_floating  # noqa: E402

CLK_NS = 2
ROWS = 3


def acc_w(dut):
    """Accumulator width, read from the DUT so ACC_W stays a parameter."""
    return len(dut.cfg_bias)


def is_high(sig):
    return str(sig.value) == "1"


def dims(dut):
    m = len(dut.out_vec) // 8
    k = len(dut.u_array.a_vec) // 8
    return k, m


# ---------------------------------------------------------------------------
# Golden model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Layer:
    def __init__(self, dut):
        self.dut = dut
        self.k, self.m = dims(dut)
        cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())

    async def reset(self):
        d = self.dut
        for sig in ("bnd_wr_en", "bnd_wr_bank", "bnd_wr_row", "bnd_wr_addr",
                    "bnd_wr_data", "wm_wr_en", "wm_wr_addr", "wm_wr_data",
                    "cfg_we", "cfg_ch", "cfg_bias", "cfg_mult", "cfg_shift",
                    "cfg_relu_en", "cfg_pool_en", "cfg_width", "cfg_ktiles",
                    "cfg_rd_bank", "cfg_ch_stride", "layer_start", "row_go"):
            getattr(d, sig).value = 0
        d.rst_n.value = 0
        for _ in range(4):
            await RisingEdge(d.clk)
        d.rst_n.value = 1
        await RisingEdge(d.clk)

    async def write_weights(self, weights, ktiles):
        """Flatten to the layout layer_top indexes: kt*K*M + m*K + k."""
        d = self.dut
        for kt in range(ktiles):
            for m in range(self.m):
                for k in range(self.k):
                    tap = kt * self.k + k
                    val = weights[m][tap] if tap < len(weights[m]) else 0
                    d.wm_wr_en.value = 1
                    d.wm_wr_addr.value = kt * self.k * self.m + m * self.k + k
                    d.wm_wr_data.value = val & 0xFF
                    await RisingEdge(d.clk)
        d.wm_wr_en.value = 0
        await RisingEdge(d.clk)

    async def write_requant(self, biases, mults, shifts):
        d = self.dut
        for ch in range(self.m):
            d.cfg_we.value = 1
            d.cfg_ch.value = ch
            d.cfg_bias.value = biases[ch] & ((1 << acc_w(self.dut)) - 1)
            d.cfg_mult.value = mults[ch]
            d.cfg_shift.value = shifts[ch]
            await RisingEdge(d.clk)
        d.cfg_we.value = 0
        await RisingEdge(d.clk)

    async def fill_band(self, band, win_channels, width, bank):
        """Write the three live rows.

        Channels beyond the layer's real count are zero-filled rather than left
        unwritten: an unwritten byte reads X, and X times a zero weight is still
        X in Verilog, so it would poison the accumulator.
        """
        d = self.dut
        d.bnd_wr_bank.value = bank
        for r in range(ROWS):
            d.bnd_wr_row.value = r
            for c in range(win_channels):
                for x in range(width):
                    d.bnd_wr_en.value = 1
                    d.bnd_wr_addr.value = c * width + x
                    value = band[r][c][x] if c < len(band[r]) else 0
                    d.bnd_wr_data.value = value & 0xFF
                    await RisingEdge(d.clk)
        d.bnd_wr_en.value = 0
        await RisingEdge(d.clk)

    async def run_layer(self, fmap, channels, height, width, ktiles,
                        relu, pool):
        """Drive a whole layer: fill each band, pulse row_go, collect output."""
        d = self.dut
        d.cfg_width.value = width
        d.cfg_ktiles.value = ktiles
        d.cfg_rd_bank.value = 0
        d.cfg_ch_stride.value = width
        d.cfg_relu_en.value = 1 if relu else 0
        d.cfg_pool_en.value = 1 if pool else 0

        d.layer_start.value = 1
        await RisingEdge(d.clk)
        d.layer_start.value = 0

        got = []
        stop = [False]

        async def monitor():
            while not stop[0]:
                await RisingEdge(d.clk)
                await Timer(1, unit="ps")
                if is_high(d.out_vld):
                    raw = d.out_vec.value.to_unsigned()
                    got.append([to_signed((raw >> (i * 8)) & 0xFF, 8)
                                for i in range(self.m)])

        mon = cocotb.start_soon(monitor())

        # +(WIN_CH-1): the window for the LAST k-tile reads channels
        # ch_base, ch_base+1 and ch_base+2, so the two channels past the
        # layer's last one are still addressed. Leaving them unwritten makes
        # them read X, and X times a zero weight is still X in Verilog, so the
        # poison reaches the accumulator. ceil(ktiles*K/9) alone is one short.
        win_channels = -(-(ktiles * self.k) // 9) + 2
        zero_row = [[0] * width for _ in range(win_channels)]

        for y in range(height):
            band = []
            for dy in (-1, 0, 1):
                yy = y + dy
                if 0 <= yy < height:
                    band.append([fmap[c][yy][:] if c < channels else [0] * width
                                 for c in range(win_channels)])
                else:
                    band.append([r[:] for r in zero_row])
            await self.fill_band(band, win_channels, width, bank=0)

            d.row_go.value = 1
            await RisingEdge(d.clk)
            d.row_go.value = 0
            await RisingEdge(d.clk)
            while is_high(d.busy):
                await RisingEdge(d.clk)
            for _ in range(8):
                await RisingEdge(d.clk)

        for _ in range(16):
            await RisingEdge(d.clk)
        stop[0] = True
        await mon
        return got


def rand_fmap(channels, height, width, rng):
    return [[[rng.randint(-128, 127) for _ in range(width)]
             for _ in range(height)] for _ in range(channels)]


def rand_weights(m_count, taps, rng):
    return [[rng.randint(-8, 8) for _ in range(taps)] for _ in range(m_count)]


async def run_case(dut, channels, height, width, relu, pool, rng, seed_note=""):
    lay = Layer(dut)
    await lay.reset()

    taps = channels * 9
    ktiles = -(-taps // lay.k)
    weights = rand_weights(lay.m, taps, rng)
    biases = [rng.randint(-2000, 2000) for _ in range(lay.m)]
    mults = [rng.randint(16384, 65535) for _ in range(lay.m)]
    shifts = [rng.randint(20, 26) for _ in range(lay.m)]

    await lay.write_weights(weights, ktiles)
    await lay.write_requant(biases, mults, shifts)

    fmap = rand_fmap(channels, height, width, rng)
    got = await lay.run_layer(fmap, channels, height, width, ktiles, relu, pool)

    accs = conv_layer(fmap, weights, channels, height, width, lay.m)
    want = fused_writeback(accs, biases, mults, shifts, height, width,
                           relu=relu, pool=pool)

    assert len(got) == len(want), (
        f"{seed_note}{channels}ch {height}x{width} pool={pool}: got "
        f"{len(got)} outputs, expected {len(want)}"
    )
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, (
            f"{seed_note}{channels}ch {height}x{width} pool={pool} output {i}:"
            f"\n  got  {g}\n  want {w}"
        )
    return got


@cocotb.test()
async def test_single_channel_no_pool(dut):
    """I1: one input channel, pooling off -- the simplest complete path."""
    rng = random.Random(0x1A1)
    await run_case(dut, channels=1, height=3, width=6, relu=True, pool=False,
                   rng=rng)


@cocotb.test()
async def test_ktile_accumulation(dut):
    """I3: more than 16 taps, so partial sums must accumulate across k-tiles.

    Two input channels is 18 taps, which needs 2 k-tiles. If the row
    accumulator overwrote instead of adding, only the last tile's contribution
    would appear and every output would be wrong by the first tile's share --
    a plausible-looking number, not an obvious failure.
    """
    rng = random.Random(0x2B2)
    await run_case(dut, channels=2, height=3, width=6, relu=True, pool=False,
                   rng=rng)


@cocotb.test()
async def test_pooling_end_to_end(dut):
    """I2: the full fused path, pooling included."""
    rng = random.Random(0x3C3)
    await run_case(dut, channels=2, height=4, width=8, relu=True, pool=True,
                   rng=rng)


@cocotb.test()
async def test_relu_disabled(dut):
    """I4: requantisation without ReLU still integrates correctly."""
    rng = random.Random(0x4D4)
    got = await run_case(dut, channels=1, height=4, width=6, relu=False,
                         pool=False, rng=rng)
    assert any(v < 0 for vec in got for v in vec), (
        "no negative outputs with ReLU off; this run proves nothing"
    )


@cocotb.test()
async def test_odd_geometry_pooling(dut):
    """I2: odd width and height, where pooling floors."""
    rng = random.Random(0x5E5)
    await run_case(dut, channels=2, height=5, width=7, relu=True, pool=True,
                   rng=rng)


@cocotb.test()
async def test_back_to_back_layers(dut):
    """I5: three layers in a row without a reset, the real tiling loop."""
    rng = random.Random(0x6F6)
    for i in range(3):
        await run_case(dut, channels=2, height=4, width=6, relu=True, pool=True,
                       rng=rng, seed_note=f"layer {i}: ")


@cocotb.test()
async def test_no_floating_signals(dut):
    """I6: nothing reads X or Z after a complete layer.

    Undriven nets simulate as X and only become a hard failure at
    place-and-route, so this is the cheap place to catch one.
    """
    rng = random.Random(0x7A7)
    await run_case(dut, channels=2, height=4, width=6, relu=True, pool=True,
                   rng=rng)

    assert_no_floating(dut, "layer_top")


@cocotb.test()
async def test_accumulator_no_overflow(dut):
    """I7: the narrowed accumulator must not wrap at the worst case.

    ACC_W was cut from a reflexive 32 to 28, which removes 12.5% of every
    accumulator register in the design. 26 is the hard bound -- conv4's 1152
    taps at 127 x 127 is 18,580,608 -- so this drives every operand to its
    maximum magnitude and checks the result against Python integers, which
    cannot wrap. A width that was too narrow shows up here as a sign flip, not
    as a small error.
    """
    lay = Layer(dut)
    await lay.reset()

    channels, height, width = 2, 4, 6
    taps = channels * 9
    ktiles = -(-taps // lay.k)

    # Every weight at -128 and every activation at +127: the largest magnitude
    # an INT8 pair can produce, accumulated over the full tap depth.
    weights = [[-128] * taps for _ in range(lay.m)]
    await lay.write_weights(weights, ktiles)
    biases = [0] * lay.m
    mults = [1] * lay.m
    shifts = [0] * lay.m
    await lay.write_requant(biases, mults, shifts)

    fmap = [[[127] * width for _ in range(height)] for _ in range(channels)]
    got = await lay.run_layer(fmap, channels, height, width, ktiles,
                              relu=False, pool=False)

    accs = conv_layer(fmap, weights, channels, height, width, lay.m)
    want = fused_writeback(accs, biases, mults, shifts, height, width,
                           relu=False, pool=False)

    worst = min(min(v) for v in accs)
    dut._log.info(f"worst accumulator value reached: {worst:,} "
                  f"(needs {abs(worst).bit_length() + 1} bits signed)")

    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, (
            f"output {i}: got {g}, want {w} -- the accumulator wrapped, "
            f"ACC_W={acc_w(dut)} is too narrow"
        )

    # The simulated geometry is small, so it only reaches ~20 bits. The width
    # that matters is conv4's, which no practical testbench can stream, so
    # assert the analytic bound for every real layer instead.
    width_bits = acc_w(dut)
    for name, in_ch in (("conv1", 1), ("conv2", 32), ("conv3", 64), ("conv4", 128)):
        taps = in_ch * 9
        bound = taps * 128 * 127          # worst INT8 pair, full tap depth
        needed = bound.bit_length() + 1   # +1 for the sign
        assert needed <= width_bits, (
            f"{name}: {taps} taps can reach {bound:,}, needing {needed} bits "
            f"signed, but ACC_W is {width_bits}"
        )
        dut._log.info(f"  {name}: {taps:5d} taps -> needs {needed:2d} bits, "
                      f"ACC_W={width_bits} OK")


# ---------------------------------------------------------------------------
# K-TILE DEPTH SWEEP
#
# Every case above uses 1 or 2 channels, which is at most TWO k-tiles -- even
# test_ktile_accumulation, whose 2 channels give 18 taps and so ceil(18/16)=2.
# The accelerator's real layers need 18 (conv2), 36 (conv3) and 72 (conv4).
#
# The FPGA harness runs 72 and mismatches. These walk the tile count up on a
# small, fast geometry to find the first depth that breaks, which is a far
# better question than "does conv4 work".
# ---------------------------------------------------------------------------

def build_fits(dut, channels, width):
    """Can this elaborated build hold a layer of this shape?

    run_layer.py builds small (MAX_KTILES=4, BAND_DEPTH=256) to keep the
    regression fast; run_layer_fpga.py builds at the real sizes. The deeper
    cases below are meaningless on the small build -- the weight memory and
    the band simply are not there -- so they skip rather than fail, which
    would otherwise report a configuration limit as an RTL defect.

    Bounds come from the port widths, which are $clog2 of the parameters, so
    they are upper bounds. That is the right direction: a case that fits the
    bound but not the parameter still fails loudly.
    """
    k, m = dims(dut)
    ktiles = -(-(channels * 9) // k)
    band_depth = 1 << len(dut.bnd_wr_addr)
    wmem_bytes = 1 << len(dut.wm_wr_addr)
    win_channels = -(-(ktiles * k) // 9) + 2
    return (ktiles * k * m <= wmem_bytes
            and win_channels * width <= band_depth)


async def _depth_case(dut, channels, note):
    if not build_fits(dut, channels, 8):
        dut._log.info(f"skipping {note}build too small for {channels} channels")
        return
    await run_case(dut, channels, 4, 8, relu=True, pool=True,
                   rng=random.Random(0xD0 + channels), seed_note=note)


@cocotb.test()
async def test_ktiles_3(dut):
    """4 channels = 36 taps = 3 k-tiles. One past what the suite covered."""
    await _depth_case(dut, 4, "3 k-tiles: ")


@cocotb.test()
async def test_ktiles_4(dut):
    """6 channels = 54 taps = 4 k-tiles."""
    await _depth_case(dut, 6, "4 k-tiles: ")


@cocotb.test()
async def test_ktiles_6(dut):
    """10 channels = 90 taps = 6 k-tiles."""
    await _depth_case(dut, 10, "6 k-tiles: ")


@cocotb.test()
async def test_ktiles_18(dut):
    """32 channels = 288 taps = 18 k-tiles -- conv2's real depth."""
    await _depth_case(dut, 32, "18 k-tiles: ")


# ---------------------------------------------------------------------------
# WIDTH SWEEP at fixed depth.
#
# 32 channels passes at W=8 but fails at W=25. W=40 also fails, but that one
# is expected: MAXW=32 sizes rowacc, and a row wider than MAXW has nowhere to
# accumulate. These walk W up to find where a LEGAL width starts failing.
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_width_12(dut):
    if not build_fits(dut, 32, 12):
        dut._log.info('skipping: build too small')
        return
    await run_case(dut, 32, 4, 12, relu=True, pool=True,
                   rng=random.Random(0x11), seed_note="W=12: ")


@cocotb.test()
async def test_width_16(dut):
    if not build_fits(dut, 32, 16):
        dut._log.info('skipping: build too small')
        return
    await run_case(dut, 32, 4, 16, relu=True, pool=True,
                   rng=random.Random(0x16), seed_note="W=16: ")


@cocotb.test()
async def test_width_20(dut):
    if not build_fits(dut, 32, 20):
        dut._log.info('skipping: build too small')
        return
    await run_case(dut, 32, 4, 20, relu=True, pool=True,
                   rng=random.Random(0x20), seed_note="W=20: ")


@cocotb.test()
async def test_width_25(dut):
    if not build_fits(dut, 32, 25):
        dut._log.info('skipping: build too small')
        return
    await run_case(dut, 32, 4, 25, relu=True, pool=True,
                   rng=random.Random(0x25), seed_note="W=25: ")


@cocotb.test()
async def test_width_20_nopool(dut):
    """W=20, 32 channels, pooling OFF.

    Splits the failure: if this passes, the convolution and k-tile
    accumulation are correct and the fault is in writeback's pooling stages.
    If it fails too, the accumulator path is already wrong before pooling
    ever sees it.
    """
    if not build_fits(dut, 32, 20):
        dut._log.info('skipping: build too small')
        return
    await run_case(dut, 32, 4, 20, relu=True, pool=False,
                   rng=random.Random(0x20), seed_note="W=20 nopool: ")


# ---------------------------------------------------------------------------
# COLUMN 18.
#
# With pooling off, W=20 / 32ch fails first at output 18 -- and with pooling
# on it fails at pooled column 9, which is exactly input columns 18-19. W=16
# passes only because it never reaches column 18. The width is not the
# variable; the column is.
#
# These hold W=20 fixed and vary only the k-tile count, which separates "one
# sweep goes wrong at column 18" from "the k-tile handover goes wrong".
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_col18_1ktile(dut):
    """1 channel = 9 taps = 1 k-tile, W=20. No handover at all."""
    await run_case(dut, 1, 4, 20, relu=True, pool=False,
                   rng=random.Random(0x18), seed_note="1kt W=20: ")


@cocotb.test()
async def test_col18_2ktile(dut):
    """2 channels = 18 taps = 2 k-tiles, W=20. One handover."""
    await run_case(dut, 2, 4, 20, relu=True, pool=False,
                   rng=random.Random(0x18), seed_note="2kt W=20: ")


@cocotb.test()
async def test_col18_4ktile(dut):
    """6 channels = 54 taps = 4 k-tiles, W=20. Three handovers."""
    if not build_fits(dut, 6, 20):
        dut._log.info('skipping: build too small')
        return
    await run_case(dut, 6, 4, 20, relu=True, pool=False,
                   rng=random.Random(0x18), seed_note="4kt W=20: ")


@cocotb.test()
async def test_col18_w17(dut):
    """2 k-tiles, W=17: highest column is 16."""
    await run_case(dut, 2, 4, 17, relu=True, pool=False,
                   rng=random.Random(0x18), seed_note="2kt W=17: ")


@cocotb.test()
async def test_col18_w18(dut):
    """2 k-tiles, W=18: highest column is 17."""
    await run_case(dut, 2, 4, 18, relu=True, pool=False,
                   rng=random.Random(0x18), seed_note="2kt W=18: ")


@cocotb.test()
async def test_col18_w19(dut):
    """2 k-tiles, W=19: highest column is 18 -- the first suspect column."""
    await run_case(dut, 2, 4, 19, relu=True, pool=False,
                   rng=random.Random(0x18), seed_note="2kt W=19: ")


# ---------------------------------------------------------------------------
# SEED SWEEP at the failing geometry.
#
# W=17, 18 and 19 all pass with 2 k-tiles; only W=20 fails, and by a single
# LSB on a single channel. That is not the shape of a structural fault -- it
# is the shape of a rare, data-dependent arithmetic edge case. If some seeds
# pass here, the width was never the variable.
# ---------------------------------------------------------------------------


@cocotb.test()
async def test_seed_0(dut):
    await run_case(dut, 2, 4, 20, relu=True, pool=False,
                   rng=random.Random(0xA1), seed_note="seed 0xa1: ")


@cocotb.test()
async def test_seed_1(dut):
    await run_case(dut, 2, 4, 20, relu=True, pool=False,
                   rng=random.Random(0xB2), seed_note="seed 0xb2: ")


@cocotb.test()
async def test_seed_2(dut):
    await run_case(dut, 2, 4, 20, relu=True, pool=False,
                   rng=random.Random(0xC3), seed_note="seed 0xc3: ")


@cocotb.test()
async def test_seed_3(dut):
    await run_case(dut, 2, 4, 20, relu=True, pool=False,
                   rng=random.Random(0xD4), seed_note="seed 0xd4: ")


@cocotb.test()
async def test_seed_4(dut):
    await run_case(dut, 2, 4, 20, relu=True, pool=False,
                   rng=random.Random(0xE5), seed_note="seed 0xe5: ")


@cocotb.test()
async def test_probe_rvld_vs_flush(dut):
    """Diagnostic: does a result land after the flush passed its column?

    S_DRAIN waits LATENCY+2 cycles from the last window before flushing
    rowacc. If that is short, a late result writes a rowacc entry the flush
    has already read, and that column keeps a partial sum -- which is exactly
    "the tail of the row is short by the last tile's contribution".

    Counts r_vld per row (should be ktiles*W) and flags any r_vld seen while
    the sequencer is in S_FLUSH or S_IDLE.
    """
    lay = Layer(dut)
    await lay.reset()

    channels, height, width = 2, 4, 20
    taps   = channels * 9
    ktiles = -(-taps // lay.k)
    rng    = random.Random(0x18)

    weights = rand_weights(lay.m, taps, rng)
    await lay.write_weights(weights, ktiles)
    await lay.write_requant([rng.randint(-2000, 2000) for _ in range(lay.m)],
                            [rng.randint(16384, 65535) for _ in range(lay.m)],
                            [rng.randint(20, 26) for _ in range(lay.m)])

    stats = {"rvld": 0, "late": 0, "in_flush": 0}

    async def probe():
        while True:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")
            if is_high(dut.r_vld):
                stats["rvld"] += 1
                st = int(dut.state.value)
                if st == 4:                      # S_FLUSH
                    stats["in_flush"] += 1
                elif st == 0:                    # S_IDLE
                    stats["late"] += 1

    cocotb.start_soon(probe())

    fmap = rand_fmap(channels, height, width, rng)
    await lay.run_layer(fmap, channels, height, width, ktiles,
                        relu=True, pool=False)

    expect = ktiles * width * height
    dut._log.info(f"r_vld pulses     : {stats['rvld']} (expect {expect})")
    dut._log.info(f"  during S_FLUSH : {stats['in_flush']}")
    dut._log.info(f"  during S_IDLE  : {stats['late']}")
    assert stats["rvld"] == expect, (
        f"result count wrong: {stats['rvld']} vs {expect} -- the oc/rkt "
        f"derivation counts r_vld pulses, so a miscount misassigns every "
        f"subsequent result to the wrong tile AND the wrong column"
    )
    assert stats["in_flush"] == 0 and stats["late"] == 0, (
        f"{stats['in_flush']} results arrived during S_FLUSH and "
        f"{stats['late']} during S_IDLE -- S_DRAIN's LATENCY+2 wait is short"
    )


@cocotb.test()
async def test_probe_rowacc(dut):
    """Diagnostic: is rowacc wrong, or only the requantised output?

    r_vld count and flush timing are both correct, so the mapping of results
    to tiles and columns is right and the values themselves must be wrong.
    This captures what writeback is actually fed -- rowacc[flush_x], the
    pre-requantisation accumulator -- and compares it against conv_layer().

    If these match, the fault is in writeback. If they differ, it is in the
    array or the k-tile accumulation, and the first differing column says
    where.
    """
    lay = Layer(dut)
    await lay.reset()

    channels, height, width = 2, 4, 20
    taps   = channels * 9
    ktiles = -(-taps // lay.k)
    rng    = random.Random(0x18)

    weights = rand_weights(lay.m, taps, rng)
    await lay.write_weights(weights, ktiles)
    await lay.write_requant([0] * lay.m, [1 << 14] * lay.m, [14] * lay.m)

    seen = []

    async def probe():
        while True:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")
            if int(dut.state.value) == 4:            # S_FLUSH
                raw = dut.u_wb.acc_vec.value.to_unsigned()
                w = acc_w(dut)
                seen.append((int(dut.flush_x.value),
                             [to_signed((raw >> (m * w)) & ((1 << w) - 1), w)
                              for m in range(lay.m)]))

    cocotb.start_soon(probe())

    fmap = rand_fmap(channels, height, width, rng)
    await lay.run_layer(fmap, channels, height, width, ktiles,
                        relu=True, pool=False)

    want = conv_layer(fmap, weights, channels, height, width, lay.m)

    dut._log.info(f"captured {len(seen)} flush beats, expected {height*width}")
    bad = []
    for i, (fx, got) in enumerate(seen):
        exp = want[i]
        if got != exp:
            bad.append((i, fx, got, exp))

    if bad:
        dut._log.info(f"{len(bad)} of {len(seen)} accumulator values wrong")
        for i, fx, got, exp in bad[:3]:
            diff = [g - e for g, e in zip(got, exp)]
            dut._log.info(f"  beat {i} flush_x={fx}")
            dut._log.info(f"    got  {got}")
            dut._log.info(f"    want {exp}")
            dut._log.info(f"    diff {diff}")
    assert not bad, f"{len(bad)} accumulator values wrong; first at beat {bad[0][0]}"
