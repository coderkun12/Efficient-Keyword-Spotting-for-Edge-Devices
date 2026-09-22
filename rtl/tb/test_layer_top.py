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
from ref_model import fused_writeback, to_signed  # noqa: E402
from xscan import assert_no_floating  # noqa: E402

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

def conv_row(band, weights, channels, width, m_count):
    """One output row of a 3x3 same-padded convolution.

    band[r][c][x] holds the three input rows already selected by the caller,
    with rows outside the map passed in as zeros.
    """
    out = []
    for x in range(width):
        acc = []
        for m in range(m_count):
            total = 0
            for c in range(channels):
                for r in range(ROWS):
                    for s in range(3):
                        col = x - 1 + s
                        if 0 <= col < width:
                            total += weights[m][c * 9 + r * 3 + s] * band[r][c][col]
            acc.append(total)
        out.append(acc)
    return out


def conv_layer(fmap, weights, channels, height, width, m_count):
    """Full layer accumulator stream, row-major, before requantisation."""
    zero_row = [[0] * width for _ in range(channels)]
    accs = []
    for y in range(height):
        band = []
        for dy in (-1, 0, 1):
            yy = y + dy
            band.append([fmap[c][yy][:] for c in range(channels)]
                        if 0 <= yy < height else [r[:] for r in zero_row])
        accs.extend(conv_row(band, weights, channels, width, m_count))
    return accs


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

        win_channels = -(-(ktiles * self.k) // 9)      # ceil, channels touched
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
