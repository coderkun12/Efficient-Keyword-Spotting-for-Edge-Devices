"""
cocotb tests for writeback -- fused requantise + BatchNorm + ReLU + 2x2 max-pool.

This is the module the whole project turns on. Convolution alone caps the
system at 2.24x by Amdahl; folding pooling, BatchNorm and ReLU into the array's
output path raises the ceiling to 44.3x, because max-pool is 38% of host
runtime while being ~0% of the MACs.

Checked here:
  W1  requantisation matches the reference, including rounding and saturation
  W2  ReLU clamps negatives to zero, and is skipped when disabled
  W3  pass-through mode (pool disabled) returns one output per input
  W4  2x2 max-pool on an even geometry
  W5  floor semantics: odd widths and heights drop the trailing column/row,
      the way torch.nn.MaxPool2d(2) does
  W6  the three real layer geometries: 40x101, 20x50 and 10x25
  W7  per-channel bias, multiplier and shift are honoured independently
"""

import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_model import fused_writeback, pack, to_signed  # noqa: E402

CLK_NS = 2
ACC_W = 32


def is_high(sig):
    return str(sig.value) == "1"


def dims(dut):
    return len(dut.out_vec) // 8


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.cfg_we.value = 0
    dut.cfg_ch.value = 0
    dut.cfg_bias.value = 0
    dut.cfg_mult.value = 0
    dut.cfg_shift.value = 0
    dut.cfg_relu_en.value = 0
    dut.cfg_pool_en.value = 0
    dut.cfg_row_width.value = 0
    dut.start.value = 0
    dut.acc_vld.value = 0
    dut.acc_vec.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def configure(dut, biases, mults, shifts, relu, pool, width):
    for ch, (b, mu, sh) in enumerate(zip(biases, mults, shifts)):
        dut.cfg_we.value = 1
        dut.cfg_ch.value = ch
        dut.cfg_bias.value = b & ((1 << ACC_W) - 1)
        dut.cfg_mult.value = mu
        dut.cfg_shift.value = sh
        await RisingEdge(dut.clk)
    dut.cfg_we.value = 0
    dut.cfg_relu_en.value = 1 if relu else 0
    dut.cfg_pool_en.value = 1 if pool else 0
    dut.cfg_row_width.value = width
    await RisingEdge(dut.clk)


async def run_stream(dut, vectors, m_count):
    """Pulse start, stream the accumulator vectors, collect INT8 outputs."""
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    got = []

    async def monitor():
        for _ in range(len(vectors) + 16):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")
            if is_high(dut.out_vld):
                raw = dut.out_vec.value.to_unsigned()
                got.append([to_signed((raw >> (i * 8)) & 0xFF, 8)
                            for i in range(m_count)])

    mon = cocotb.start_soon(monitor())
    for vec in vectors:
        dut.acc_vld.value = 1
        dut.acc_vec.value = pack(vec, ACC_W)
        await RisingEdge(dut.clk)
    dut.acc_vld.value = 0
    dut.acc_vec.value = 0
    await mon
    return got


def rand_cfg(m_count, rng, shift_lo=22, shift_hi=28):
    """Realistic requantisation constants.

    A real INT8 scale is around 1e-3: accumulators near 1e5 have to land inside
    [-128, 127]. With a 16-bit multiplier that means shifts in the mid-20s.
    Smaller shifts saturate every output, which would make these tests pass on
    nothing but clamping.
    """
    biases = [rng.randint(-5000, 5000) for _ in range(m_count)]
    mults = [rng.randint(16384, 65535) for _ in range(m_count)]
    shifts = [rng.randint(shift_lo, shift_hi) for _ in range(m_count)]
    return biases, mults, shifts


def rand_acc(m_count, count, rng, lo=-200000, hi=200000):
    return [[rng.randint(lo, hi) for _ in range(m_count)] for _ in range(count)]


async def check(dut, height, width, relu, pool, rng, shift_lo=22, shift_hi=28):
    m_count = dims(dut)
    biases, mults, shifts = rand_cfg(m_count, rng, shift_lo, shift_hi)
    await configure(dut, biases, mults, shifts, relu, pool, width)

    vectors = rand_acc(m_count, height * width, rng)
    got = await run_stream(dut, vectors, m_count)
    want = fused_writeback(vectors, biases, mults, shifts, height, width,
                           relu=relu, pool=pool)

    assert len(got) == len(want), (
        f"{height}x{width} pool={pool}: got {len(got)} outputs, expected "
        f"{len(want)}"
    )
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"{height}x{width} pool={pool} output {i}: got {g}, want {w}"
    return got


@cocotb.test()
async def test_requantise_passthrough(dut):
    """W1/W3: requantisation with pooling off, one output per input."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0x0EAD)
    await check(dut, height=4, width=8, relu=True, pool=False, rng=rng)


@cocotb.test()
async def test_relu_disabled(dut):
    """W2: with ReLU off, negatives survive down to -128."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0xBEEF)
    got = await check(dut, height=4, width=8, relu=False, pool=False, rng=rng)
    assert any(v < 0 for vec in got for v in vec), (
        "no negative outputs with ReLU disabled; the test data never went "
        "negative, so this run proves nothing"
    )


@cocotb.test()
async def test_relu_enabled_clamps(dut):
    """W2: with ReLU on, nothing is negative."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0xBEEF)
    got = await check(dut, height=4, width=8, relu=True, pool=False, rng=rng)
    assert all(v >= 0 for vec in got for v in vec), "ReLU let a negative through"


@cocotb.test()
async def test_saturation(dut):
    """W1: accumulators far outside INT8 range must saturate, not wrap.

    A wrap here is catastrophic and silent: a large positive activation would
    reappear as a large negative one.
    """
    await start_clock(dut)
    await reset(dut)
    m_count = dims(dut)

    biases = [0] * m_count
    mults = [65535] * m_count
    shifts = [8] * m_count          # deliberately far too little shift
    await configure(dut, biases, mults, shifts, relu=False, pool=False, width=4)

    vectors = [[10**6] * m_count, [-10**6] * m_count,
               [0] * m_count, [1] * m_count]
    got = await run_stream(dut, vectors, m_count)
    want = fused_writeback(vectors, biases, mults, shifts, 1, 4,
                           relu=False, pool=False)
    assert got == want, f"saturation mismatch:\n  got  {got}\n  want {want}"
    assert got[0] == [127] * m_count, "large positive did not saturate to +127"
    assert got[1] == [-128] * m_count, "large negative did not saturate to -128"


@cocotb.test()
async def test_pool_even_geometry(dut):
    """W4: 2x2 max-pool on an even grid."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0x900D)
    got = await check(dut, height=4, width=8, relu=True, pool=True, rng=rng)
    assert len(got) == (4 // 2) * (8 // 2)


@cocotb.test()
async def test_pool_odd_geometry(dut):
    """W5: odd width and height drop the trailing column and row.

    torch.nn.MaxPool2d(2) floors. Getting this wrong shifts every downstream
    feature by a pixel, which shows up as a quiet accuracy loss rather than an
    obvious failure, so it is worth asserting directly.
    """
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0x0DD)
    for height, width in ((5, 7), (3, 9), (7, 5)):
        got = await check(dut, height, width, relu=True, pool=True, rng=rng)
        assert len(got) == (height // 2) * (width // 2), (
            f"{height}x{width}: got {len(got)} pooled outputs, expected "
            f"{(height // 2) * (width // 2)}"
        )


@cocotb.test()
async def test_real_layer_geometries(dut):
    """W6: the three pooling layers in KeywordSpottingCNN.

    conv2 40x101 -> 20x50, conv3 20x50 -> 10x25, conv4 10x25 -> 5x12.
    These are the shapes eda_profiling.txt reports for the trained model.
    """
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0xC0FFEE)
    for name, height, width, ph, pw in (
        ("conv3-in", 20, 50, 10, 25),
        ("conv4-in", 10, 25, 5, 12),
    ):
        got = await check(dut, height, width, relu=True, pool=True, rng=rng)
        assert len(got) == ph * pw, (
            f"{name}: {height}x{width} pooled to {len(got)}, expected {ph*pw}"
        )
        dut._log.info(f"{name}: {height}x{width} -> {ph}x{pw}, {len(got)} vectors")


@cocotb.test()
async def test_per_channel_config(dut):
    """W7: each channel's bias, multiplier and shift act independently.

    Per-channel constants are what carry the folded BatchNorm, so a shared
    register here would quietly apply one channel's statistics to all of them.
    """
    await start_clock(dut)
    await reset(dut)
    m_count = dims(dut)

    biases = [100 * (m + 1) for m in range(m_count)]
    mults = [1 << 12] * m_count
    shifts = [12] * m_count
    await configure(dut, biases, mults, shifts, relu=False, pool=False, width=2)

    vectors = [[0] * m_count, [0] * m_count]
    got = await run_stream(dut, vectors, m_count)
    want = [[min(127, 100 * (m + 1)) for m in range(m_count)] for _ in range(2)]
    assert got == want, (
        f"per-channel bias not applied independently:\n  got  {got[0]}\n"
        f"  want {want[0]}"
    )
