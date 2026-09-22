"""
cocotb tests for pe_int8 -- the single INT8 multiply-accumulate cell.

Checked here:
  T1  reset clears the weight, activation and partial-sum registers
  T2  weight shift chain: value in, value out one cycle later
  T3  MAC correctness against ref_model over the full INT8 corner set
  T4  MAC correctness over randomised operands
  T5  activation pass-through has exactly one cycle of latency
  T6  the array's accumulate chain: psum_in adds in, unregistered weights hold
"""

import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_model import pe_step, to_signed  # noqa: E402

ACC_W = int(os.environ.get("ACC_W", 32))
CLK_NS = 2  # 500 MHz, the design target


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.w_shift_en.value = 0
    dut.w_switch.value = 0
    dut.w_in.value = 0
    dut.a_in.value = 0
    dut.psum_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weight(dut, w):
    """Push one weight into the shadow register, then commit it to active.

    M3 split the single weight register into shadow + active so the next tile
    can shift in while the current one streams. A load is therefore two steps,
    and forgetting the commit leaves the PE multiplying by the previous tile.
    """
    dut.w_shift_en.value = 1
    dut.w_in.value = w & 0xFF
    await RisingEdge(dut.clk)
    dut.w_shift_en.value = 0
    dut.w_switch.value = 1
    await RisingEdge(dut.clk)
    dut.w_switch.value = 0
    await Timer(1, unit="ps")   # let the NBA update settle before reading


def rd_signed(handle, width):
    return to_signed(handle.value.to_unsigned(), width)


@cocotb.test()
async def test_reset(dut):
    """T1: reset clears every register."""
    await start_clock(dut)
    dut.rst_n.value = 0
    dut.w_shift_en.value = 1
    dut.w_switch.value = 1
    dut.w_in.value = 0x7F
    dut.a_in.value = 0x55
    dut.psum_in.value = 12345
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    assert rd_signed(dut.a_out, 8) == 0, "a_out not cleared by reset"
    assert rd_signed(dut.psum_out, ACC_W) == 0, "psum_out not cleared by reset"
    assert rd_signed(dut.w_out, 8) == 0, "w_shadow not cleared by reset"


@cocotb.test()
async def test_weight_shift(dut):
    """T2: the shift chain moves one weight per enabled cycle, and only then.

    w_out exposes the SHADOW register, which is what the chain hands to the PE
    below -- not the active weight being multiplied.
    """
    await start_clock(dut)
    await reset(dut)

    for w in (1, -1, 127, -128, 42):
        await load_weight(dut, w)
        assert rd_signed(dut.w_out, 8) == w, (
            f"w_out={rd_signed(dut.w_out, 8)} after loading {w}"
        )

    # With w_shift_en low the weight must hold, which is the whole point of
    # weight-stationary: it survives every activation cycle that follows.
    held = rd_signed(dut.w_out, 8)
    dut.w_in.value = 0x00
    for _ in range(5):
        await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    assert rd_signed(dut.w_out, 8) == held, "weight moved with w_shift_en low"


@cocotb.test()
async def test_mac_corners(dut):
    """T3: INT8 corner cases, where sign handling and width usually break."""
    await start_clock(dut)
    await reset(dut)

    corners = (-128, -127, -1, 0, 1, 126, 127)
    psums = (0, 1, -1, 100000, -100000)

    for w in corners:
        await load_weight(dut, w)
        for a in corners:
            for p in psums:
                dut.a_in.value = a & 0xFF
                dut.psum_in.value = p & ((1 << ACC_W) - 1)
                await RisingEdge(dut.clk)
                await Timer(1, unit="ps")
                got = rd_signed(dut.psum_out, ACC_W)
                want = pe_step(w, a, p)
                assert got == want, (
                    f"w={w} a={a} psum_in={p}: got {got}, expected {want}"
                )


@cocotb.test()
async def test_mac_random(dut):
    """T4: randomised operands, 400 vectors."""
    await start_clock(dut)
    await reset(dut)
    rng = random.Random(0xC0FFEE)

    for _ in range(400):
        w = rng.randint(-128, 127)
        a = rng.randint(-128, 127)
        p = rng.randint(-(2**20), 2**20)
        await load_weight(dut, w)
        dut.a_in.value = a & 0xFF
        dut.psum_in.value = p & ((1 << ACC_W) - 1)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        got = rd_signed(dut.psum_out, ACC_W)
        want = pe_step(w, a, p)
        assert got == want, f"w={w} a={a} psum={p}: got {got}, expected {want}"


@cocotb.test()
async def test_activation_passthrough(dut):
    """T5: a_out is a REGISTERED copy of a_in, one clock edge behind.

    The array's skew calculation rests on this being exactly one register deep:
    a value presented to row k during cycle t is multiplied by PE(k,0) at edge
    t, by PE(k,1) at edge t+1, and so by PE(k,m) at edge t+m. That t+m is the
    "+m" in the k+m diagonal that mac_array.sv skews for. So this test checks
    both halves: the output does NOT move combinationally when a_in changes,
    and it DOES take the new value after the next edge.
    """
    await start_clock(dut)
    await reset(dut)

    dut.psum_in.value = 0
    prev = 0
    for a in (5, -5, 127, -128, 0, 99):
        dut.a_in.value = a & 0xFF
        await Timer(1, unit="ps")
        assert rd_signed(dut.a_out, 8) == prev, (
            f"a_out moved to {rd_signed(dut.a_out, 8)} without a clock edge; "
            f"expected it to hold {prev}. The path is combinational."
        )
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        assert rd_signed(dut.a_out, 8) == a, (
            f"a_out={rd_signed(dut.a_out, 8)} after the edge, expected {a}"
        )
        prev = a


@cocotb.test()
async def test_accumulate_chain(dut):
    """T6: a running sum through psum_in, the way a column actually works."""
    await start_clock(dut)
    await reset(dut)

    w = 7
    await load_weight(dut, w)

    acc = 0
    for a in range(-64, 64, 7):
        dut.a_in.value = a & 0xFF
        dut.psum_in.value = acc & ((1 << ACC_W) - 1)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        acc = pe_step(w, a, acc)
        assert rd_signed(dut.psum_out, ACC_W) == acc, (
            f"running accumulate diverged at a={a}"
        )
