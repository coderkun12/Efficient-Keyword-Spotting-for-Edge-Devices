"""
cocotb tests for band_sram -- the row-banded activation scratchpad.

The module holds ROWS rows of a feature map for many channels, double-buffered,
and emits a 3x3 im2col window per output column as it sweeps a row band.

Checked here:
  B1  a full sweep matches the Python window model, including edge padding
  B2  left and right padding are zero, asserted explicitly at x=0 and x=W-1
  B3  the sweep emits exactly W valid windows, no more and no fewer
  B4  double buffering: filling one bank does not disturb a sweep of the other
  B5  a second sweep over the same bank reproduces the same windows
  B6  conv2, conv3 and conv4 band geometries all fit and sweep correctly
"""

import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))

CLK_NS = 2
ROWS = 3
WIN_CH = 3


def is_high(sig):
    return str(sig.value) == "1"


def expected_window(band, ch_base, width, x):
    """Golden 3x3 im2col window, packed as [((c*ROWS + r)*3 + t)].

    band[row][channel][col]. Columns outside [0, width) are zero, which is what
    padding=1 means for every convolution in this model.
    """
    out = []
    for c in range(WIN_CH):
        for r in range(ROWS):
            for t in range(3):
                col = x - 1 + t
                if 0 <= col < width:
                    out.append(band[r][ch_base + c][col])
                else:
                    out.append(0)
    return out


def unpack_window(dut):
    raw = dut.win.value.to_unsigned()
    return [(raw >> (i * 8)) & 0xFF for i in range(WIN_CH * ROWS * 3)]


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.wr_bank.value = 0
    dut.wr_row.value = 0
    dut.wr_addr.value = 0
    dut.wr_data.value = 0
    dut.rd_bank.value = 0
    dut.sweep_start.value = 0
    dut.rd_base.value = 0
    dut.ch_stride.value = 0
    dut.row_width.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def make_band(channels, width, rng):
    """band[row][channel][col] of random bytes."""
    return [[[rng.randint(0, 255) for _ in range(width)]
             for _ in range(channels)] for _ in range(ROWS)]


async def fill(dut, band, channels, width, bank):
    """Write a whole band into one buffer, address = channel*width + column."""
    dut.wr_bank.value = bank
    for r in range(ROWS):
        dut.wr_row.value = r
        for c in range(channels):
            for x in range(width):
                dut.wr_en.value = 1
                dut.wr_addr.value = c * width + x
                dut.wr_data.value = band[r][c][x]
                await RisingEdge(dut.clk)
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)


async def sweep(dut, bank, ch_base, width, max_beats=None):
    """Run one sweep, returning the list of (x, window) beats observed."""
    dut.rd_bank.value = bank
    dut.rd_base.value = ch_base * width
    dut.ch_stride.value = width
    dut.row_width.value = width
    dut.sweep_start.value = 1
    await RisingEdge(dut.clk)
    dut.sweep_start.value = 0

    beats = []
    limit = max_beats if max_beats is not None else width + 12
    for _ in range(limit):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if is_high(dut.win_vld):
            beats.append((int(dut.win_x.value), unpack_window(dut)))
    return beats


@cocotb.test()
async def test_sweep_matches_model(dut):
    """B1: every window in a full sweep matches the Python model."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0x5EED)

    channels, width = 8, 12
    band = make_band(channels, width, rng)
    await fill(dut, band, channels, width, bank=0)

    beats = await sweep(dut, bank=0, ch_base=0, width=width)
    assert len(beats) == width, f"expected {width} windows, got {len(beats)}"

    for x, got in beats:
        want = expected_window(band, 0, width, x)
        assert got == want, f"column x={x}:\n  got  {got}\n  want {want}"


@cocotb.test()
async def test_edge_padding(dut):
    """B2: the first and last columns pad with zeros, not with wrapped data.

    A wrap here would be silent: the arithmetic stays plausible and only the
    accuracy moves, which is the hardest kind of bug to find downstream.
    """
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(2)

    channels, width = 4, 9
    band = make_band(channels, width, rng)
    await fill(dut, band, channels, width, bank=0)
    beats = dict(await sweep(dut, bank=0, ch_base=0, width=width))

    first = beats[0]
    last = beats[width - 1]
    for c in range(WIN_CH):
        for r in range(ROWS):
            left = first[(c * ROWS + r) * 3 + 0]
            right = last[(c * ROWS + r) * 3 + 2]
            assert left == 0, f"x=0 ch{c} row{r} left tap is {left}, expected pad 0"
            assert right == 0, (
                f"x={width-1} ch{c} row{r} right tap is {right}, expected pad 0"
            )


@cocotb.test()
async def test_sweep_length(dut):
    """B3: exactly W valid beats, so the downstream vector count is right."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(3)

    for width in (4, 7, 25, 50):
        channels = 3
        band = make_band(channels, width, rng)
        await fill(dut, band, channels, width, bank=0)
        beats = await sweep(dut, bank=0, ch_base=0, width=width)
        assert len(beats) == width, (
            f"width={width}: got {len(beats)} valid windows, expected {width}"
        )
        xs = [x for x, _ in beats]
        assert xs == list(range(width)), f"width={width}: win_x sequence {xs}"


@cocotb.test()
async def test_double_buffering(dut):
    """B4: filling the far bank must not perturb a sweep of the near one.

    This is the property the whole banding scheme rests on: the next row band
    loads while the current one computes. If the banks interfered, the overlap
    would have to be given up and the buffer would double.
    """
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0xDB)

    channels, width = 5, 10
    band_a = make_band(channels, width, rng)
    band_b = make_band(channels, width, rng)
    await fill(dut, band_a, channels, width, bank=0)

    # Start a sweep of bank 0, then write bank 1 underneath it.
    dut.rd_bank.value = 0
    dut.rd_base.value = 0
    dut.ch_stride.value = width
    dut.row_width.value = width
    dut.sweep_start.value = 1
    await RisingEdge(dut.clk)
    dut.sweep_start.value = 0

    writer = cocotb.start_soon(fill(dut, band_b, channels, width, bank=1))

    beats = []
    for _ in range(width + 12):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if is_high(dut.win_vld):
            beats.append((int(dut.win_x.value), unpack_window(dut)))
    await writer

    assert len(beats) == width, f"expected {width} windows, got {len(beats)}"
    for x, got in beats:
        want = expected_window(band_a, 0, width, x)
        assert got == want, (
            f"bank 0 sweep corrupted at x={x} by concurrent bank 1 fill:\n"
            f"  got  {got}\n  want {want}"
        )

    # And the freshly written bank reads back correctly.
    beats_b = await sweep(dut, bank=1, ch_base=0, width=width)
    for x, got in beats_b:
        assert got == expected_window(band_b, 0, width, x), (
            f"bank 1 wrong at x={x} after concurrent write"
        )


@cocotb.test()
async def test_repeat_sweep_and_channel_offset(dut):
    """B5/B6: sweeps repeat, and a non-zero channel base selects the right group."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(7)

    channels, width = 9, 11
    band = make_band(channels, width, rng)
    await fill(dut, band, channels, width, bank=0)

    for ch_base in (0, 3, 6):
        for attempt in range(2):
            beats = await sweep(dut, bank=0, ch_base=ch_base, width=width)
            assert len(beats) == width
            for x, got in beats:
                want = expected_window(band, ch_base, width, x)
                assert got == want, (
                    f"ch_base={ch_base} attempt={attempt} x={x}:\n"
                    f"  got  {got}\n  want {want}"
                )


@cocotb.test()
async def test_real_layer_geometries(dut):
    """B6: the band shapes the plan sized the SRAM for.

    conv2 32ch x 101 wide, conv3 64ch x 50, conv4 128ch x 25. All three must
    fit inside one DEPTH-byte row bank and sweep correctly, because DEPTH was
    chosen as the max of channels*width across exactly these layers.
    """
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0xC04F)

    depth = 2 ** len(dut.wr_addr)
    geometries = [("conv2", 32, 101), ("conv3", 64, 50), ("conv4", 128, 25)]

    for name, channels, width in geometries:
        assert channels * width <= depth, (
            f"{name} needs {channels * width} B per row bank, DEPTH is {depth}"
        )
        # Only the channels the window actually reads need real data; filling
        # all 128 channels x 3 rows would dominate the runtime for no coverage.
        used = WIN_CH
        band = make_band(channels, width, rng)
        dut.wr_bank.value = 0
        for r in range(ROWS):
            dut.wr_row.value = r
            for c in range(used):
                for x in range(width):
                    dut.wr_en.value = 1
                    dut.wr_addr.value = c * width + x
                    dut.wr_data.value = band[r][c][x]
                    await RisingEdge(dut.clk)
        dut.wr_en.value = 0
        await RisingEdge(dut.clk)

        beats = await sweep(dut, bank=0, ch_base=0, width=width)
        assert len(beats) == width, (
            f"{name}: expected {width} windows, got {len(beats)}"
        )
        for x, got in beats:
            want = expected_window(band, 0, width, x)
            assert got == want, f"{name} x={x}:\n  got  {got}\n  want {want}"
        dut._log.info(
            f"{name}: {channels}ch x {width} wide = "
            f"{channels * width * ROWS:,} B band, swept {len(beats)} columns"
        )
