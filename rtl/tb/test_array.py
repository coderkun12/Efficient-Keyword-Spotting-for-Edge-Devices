"""
cocotb tests for mac_array -- the K x M weight-stationary INT8 systolic array.

The array is a streaming matrix-vector multiplier: present an aligned K-vector
on any cycle, get the aligned M-vector W @ X back LATENCY = K + M - 1 cycles
later, one vector per cycle with no stalls.

Checked here:
  A1  weight shift order leaves PE(k,m) holding W[m][k]
  A2  single vector, hand-checkable weights
  A3  dense stream of random vectors, back to back, against ref_model
  A4  gaps in the input stream do not corrupt valid data (valid tracking)
  A5  latency is exactly K + M - 1, asserted directly
  A6  INT8 corner operands, including the -128 asymmetry
  A7  real conv2 weights and activations from the trained checkpoint, if present

K and M are read from the DUT's port widths so the same file covers every
elaboration the runner builds.
"""

import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_model import (  # noqa: E402
    matvec, pack, random_activations, random_weights, to_signed,
    weight_shift_order,
)

ACC_W = 32
CLK_NS = 2  # 500 MHz


def is_high(sig):
    """True when a 1-bit signal reads as logic 1.

    cocotb 2.0 returns a `Logic` for scalar signals and a `LogicArray` for
    vectors; only the latter has to_unsigned(), so scalars need their own path.
    """
    return str(sig.value) == "1"


def dims(dut):
    """Recover K and M from port widths, so the test adapts to the build."""
    k = len(dut.a_vec) // 8
    m = len(dut.r_vec) // ACC_W
    return k, m


def pipe_depth(dut):
    """Read the PIPE parameter; fall back to 1 if the simulator hides it."""
    try:
        return int(dut.PIPE.value)
    except Exception:
        return 1


def latency_of(dut):
    k, m = dims(dut)
    return pipe_depth(dut) * k + m - 1


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.w_shift_en.value = 0
    dut.w_switch.value = 0
    dut.w_top.value = 0
    dut.a_vld.value = 0
    dut.a_vec.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")


async def load_weights(dut, weights, k_count):
    """Shift a full M x K tile into the shadow registers, then commit it.

    The commit must lead the data by exactly one cycle. w_switch injected at
    cycle T reaches row k at T + PIPE*k, and an activation injected at T
    reaches row k at the same T + PIPE*k -- but at that edge the PE multiplies
    with the OLD active weight and only then registers the new one. Pulsing
    w_switch one cycle before the first vector keeps that one-cycle lead at
    every row, because both travel down the array at the same rate.
    """
    for column_bytes in weight_shift_order(weights, k_count):
        dut.w_shift_en.value = 1
        dut.w_top.value = pack(column_bytes, 8)
        await RisingEdge(dut.clk)
    dut.w_shift_en.value = 0
    dut.w_top.value = 0
    dut.w_switch.value = 1
    await RisingEdge(dut.clk)
    dut.w_switch.value = 0
    await Timer(1, unit="ps")


def read_results(dut, m_count):
    raw = dut.r_vec.value.to_unsigned()
    mask = (1 << ACC_W) - 1
    return [to_signed((raw >> (i * ACC_W)) & mask, ACC_W) for i in range(m_count)]


async def collect(dut, m_count, cycles, sink, stamps=None):
    """Sample the result bus for `cycles` edges, recording valid beats."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if is_high(dut.r_vld):
            sink.append(read_results(dut, m_count))
            if stamps is not None:
                stamps.append(cocotb.utils.get_sim_time("ns"))


async def drive_stream(dut, vectors, k_count, gaps=None):
    """Drive aligned activation vectors, optionally with idle cycles between."""
    for i, vec in enumerate(vectors):
        if gaps and i in gaps:
            dut.a_vld.value = 0
            dut.a_vec.value = 0
            for _ in range(gaps[i]):
                await RisingEdge(dut.clk)
        dut.a_vld.value = 1
        dut.a_vec.value = pack(vec, 8)
        await RisingEdge(dut.clk)
    dut.a_vld.value = 0
    dut.a_vec.value = 0


async def run_case(dut, weights, vectors, gaps=None):
    """Shared body: load weights, stream vectors, return collected results."""
    k_count, m_count = dims(dut)
    await reset(dut)
    await load_weights(dut, weights, k_count)

    got = []
    latency = latency_of(dut)
    total_gap = sum(gaps.values()) if gaps else 0
    monitor = cocotb.start_soon(
        collect(dut, m_count, len(vectors) + total_gap + latency + 4, got)
    )
    await drive_stream(dut, vectors, k_count, gaps)
    await monitor
    return got


@cocotb.test()
async def test_weight_load_order(dut):
    """A1: after K shifts, column m holds W[m][k] at row k.

    Verified behaviourally: drive a one-hot activation vector selecting tap k,
    so the output must be exactly W[m][k] for every m. If the shift order were
    reversed this fails immediately on k=0.
    """
    await start_clock(dut)
    k_count, m_count = dims(dut)
    rng = random.Random(1)
    weights = random_weights(m_count, k_count, rng)

    for k in range(k_count):
        one_hot = [0] * k_count
        one_hot[k] = 1
        got = await run_case(dut, weights, [one_hot])
        assert len(got) == 1, f"tap {k}: expected 1 result beat, got {len(got)}"
        want = [weights[m][k] for m in range(m_count)]
        assert got[0] == want, (
            f"tap {k}: got {got[0]}, expected {want}. "
            "Weight shift order is wrong -- see ref_model.weight_shift_order."
        )


@cocotb.test()
async def test_single_vector(dut):
    """A2: one vector with hand-checkable values."""
    await start_clock(dut)
    k_count, m_count = dims(dut)

    # W[m][k] = m + 1 for every k, so Y[m] = (m+1) * sum(X).
    weights = [[(m + 1) for _ in range(k_count)] for m in range(m_count)]
    vec = [2] * k_count
    got = await run_case(dut, weights, [vec])

    assert len(got) == 1
    want = [(m + 1) * 2 * k_count for m in range(m_count)]
    assert got[0] == want, f"got {got[0]}, expected {want}"
    assert got[0] == matvec(weights, vec), "RTL and ref_model disagree"


@cocotb.test()
async def test_dense_stream(dut):
    """A3: 64 random vectors back to back, one result per cycle."""
    await start_clock(dut)
    k_count, m_count = dims(dut)
    rng = random.Random(0xBEEF)

    weights = random_weights(m_count, k_count, rng)
    vectors = random_activations(k_count, 64, rng)
    got = await run_case(dut, weights, vectors)

    assert len(got) == len(vectors), (
        f"expected {len(vectors)} result beats, got {len(got)}. "
        "The array should sustain one vector per cycle with no stalls."
    )
    for n, (g, v) in enumerate(zip(got, vectors)):
        want = matvec(weights, v)
        assert g == want, f"vector {n}: got {g}, expected {want}"


@cocotb.test()
async def test_stream_with_gaps(dut):
    """A4: idle cycles between vectors must not corrupt the valid ones.

    This is the test that catches a mistracked valid pipeline: the datapath
    keeps computing on stale activations during the gap, and if r_vld is even
    one cycle off, garbage diagonals get reported as results.
    """
    await start_clock(dut)
    k_count, m_count = dims(dut)
    rng = random.Random(0x1234)

    weights = random_weights(m_count, k_count, rng)
    vectors = random_activations(k_count, 12, rng)
    gaps = {2: 1, 5: 3, 9: 7}
    got = await run_case(dut, weights, vectors, gaps)

    assert len(got) == len(vectors), (
        f"expected {len(vectors)} beats, got {len(got)}; gaps leaked into r_vld"
    )
    for n, (g, v) in enumerate(zip(got, vectors)):
        want = matvec(weights, v)
        assert g == want, f"vector {n} after gaps: got {g}, expected {want}"


@cocotb.test()
async def test_latency(dut):
    """A5: the first result lands exactly K + M - 1 cycles after its input."""
    await start_clock(dut)
    k_count, m_count = dims(dut)
    expected_latency = latency_of(dut)

    weights = [[1] * k_count for _ in range(m_count)]
    await reset(dut)
    await load_weights(dut, weights, k_count)

    dut.a_vld.value = 1
    dut.a_vec.value = pack([1] * k_count, 8)
    await RisingEdge(dut.clk)
    dut.a_vld.value = 0
    dut.a_vec.value = 0

    for cycle in range(1, expected_latency + 5):
        await Timer(1, unit="ps")
        if is_high(dut.r_vld):
            assert cycle == expected_latency, (
                f"r_vld asserted after {cycle} cycles, expected "
                f"{expected_latency} (PIPE={pipe_depth(dut)} * K={k_count} "
                f"+ M={m_count} - 1)"
            )
            assert read_results(dut, m_count) == [k_count] * m_count
            return
        await RisingEdge(dut.clk)

    raise AssertionError(f"r_vld never asserted within {expected_latency + 4} cycles")


@cocotb.test()
async def test_int8_corners(dut):
    """A6: saturated operands, including the asymmetric -128.

    -128 has no positive counterpart in INT8, so a sign-extension bug in the
    multiplier or the accumulator widening shows up here and nowhere else.
    """
    await start_clock(dut)
    k_count, m_count = dims(dut)

    cases = [
        ([[-128] * k_count for _ in range(m_count)], [-128] * k_count),
        ([[-128] * k_count for _ in range(m_count)], [127] * k_count),
        ([[127] * k_count for _ in range(m_count)], [-128] * k_count),
        ([[127] * k_count for _ in range(m_count)], [127] * k_count),
    ]
    for weights, vec in cases:
        got = await run_case(dut, weights, [vec])
        want = matvec(weights, vec)
        assert got[0] == want, (
            f"corner w={weights[0][0]} a={vec[0]}: got {got[0]}, expected {want}"
        )


@cocotb.test()
async def test_real_conv2_weights(dut):
    """A7: real trained weights, quantised to INT8, rather than random data.

    Skipped when the checkpoint is not on this machine, so the suite still runs
    standalone. Uses conv2 because it is the largest single MAC contributor at
    40.0% of the model.
    """
    # cocotb 2.0 removed cocotb.result.TestSuccess, so a runtime skip is just
    # an early return with a clear log line.
    ckpt = (Path(__file__).resolve().parents[2]
            / "results" / "checkpoints" / "baseline_best.pt")
    if not ckpt.exists():
        dut._log.warning(
            f"SKIPPED: no checkpoint at {ckpt}. "
            "Run this suite on the machine holding results/checkpoints/."
        )
        return
    try:
        import torch
    except ImportError:
        dut._log.warning("SKIPPED: torch not installed in this environment.")
        return

    await start_clock(dut)
    k_count, m_count = dims(dut)

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"] if isinstance(state, dict) else state
    w = sd["features.3.weight"]                       # conv2: (64, 32, 3, 3)
    flat = w.reshape(w.shape[0], -1)                  # (M=64, K=288)

    # Symmetric per-tensor INT8 quantisation, matching what QAT produces.
    scale = flat.abs().max().item() / 127.0
    q = torch.clamp(torch.round(flat / scale), -128, 127).to(torch.int64)

    weights = [[int(q[m][k]) for k in range(k_count)] for m in range(m_count)]
    rng = random.Random(7)
    vectors = random_activations(k_count, 16, rng)
    got = await run_case(dut, weights, vectors)

    assert len(got) == len(vectors)
    for n, (g, v) in enumerate(zip(got, vectors)):
        want = matvec(weights, v)
        assert g == want, f"real-weight vector {n}: got {g}, expected {want}"
    dut._log.info(
        f"conv2 tile verified: {m_count}x{k_count} real INT8 weights, "
        f"scale={scale:.6g}"
    )
