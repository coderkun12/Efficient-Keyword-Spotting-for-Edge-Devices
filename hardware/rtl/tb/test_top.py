"""
cocotb tests for tile_top -- the M2 single-tile accelerator behind real AXI.

Driven with cocotbext-axi's stock AXI4-Lite master and AXI4-Stream source/sink,
so the handshakes are exercised by an independent protocol implementation
rather than by testbench code written to match this RTL's assumptions.

Checked here:
  S1  AXI4-Lite identity and configuration registers
  S2  weight staging writes read back byte-exact
  S3  end-to-end: load weights, stream activations, results match ref_model
  S4  AXI4-Lite result readback agrees with the result stream
  S5  back-pressure on the result stream loses nothing and sets no overflow
  S6  input back-pressure while weights are shifting (tready low during load)
  S7  a second weight tile can be loaded and used without a reset
  S8  writing an unmapped address returns SLVERR rather than silently passing
  S9  M3: the next tile's weights load WHILE the current tile streams
"""

import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import (
    AxiLiteBus, AxiLiteMaster, AxiStreamBus, AxiStreamSink, AxiStreamSource,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_model import matvec, random_activations, random_weights, to_signed  # noqa: E402
from xscan import assert_no_floating  # noqa: E402

CLK_NS = 2  # 500 MHz

ADDR_CTRL = 0x0000
ADDR_STATUS = 0x0004
ADDR_ID = 0x0008
ADDR_CFG = 0x000C
WEIGHT_BASE = 0x1000
RESULT_BASE = 0x2000

ID_VALUE = 0x4B575301
ACC_W = 32


class Harness:
    def __init__(self, dut):
        self.dut = dut
        self.k = len(dut.s_axis_tdata) // 8
        self.m = len(dut.m_axis_tdata) // ACC_W
        self.axil = AxiLiteMaster(
            AxiLiteBus.from_prefix(dut, "s_axil"), dut.clk, dut.rst_n,
            reset_active_level=False,
        )
        self.source = AxiStreamSource(
            AxiStreamBus.from_prefix(dut, "s_axis"), dut.clk, dut.rst_n,
            reset_active_level=False,
        )
        self.sink = AxiStreamSink(
            AxiStreamBus.from_prefix(dut, "m_axis"), dut.clk, dut.rst_n,
            reset_active_level=False,
        )
        cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())

    async def reset(self):
        self.dut.rst_n.value = 0
        for _ in range(5):
            await RisingEdge(self.dut.clk)
        self.dut.rst_n.value = 1
        for _ in range(5):
            await RisingEdge(self.dut.clk)

    async def stage_weights(self, weights):
        """Write a tile into the staging array without starting a load."""
        flat = bytearray(self.k * self.m)
        for m in range(self.m):
            for k in range(self.k):
                flat[m * self.k + k] = weights[m][k] & 0xFF
        for word in range(len(flat) // 4):
            value = int.from_bytes(flat[word * 4:word * 4 + 4], "little")
            await self.axil.write_dword(WEIGHT_BASE + word * 4, value)

    async def wait_loaded(self, timeout=256):
        """Poll STATUS until the shift chain has settled."""
        for _ in range(timeout):
            if (await self.axil.read_dword(ADDR_STATUS)) & 0x1:
                return
        raise AssertionError("STATUS.w_loaded never set; the shift never finished")

    async def write_weights(self, weights):
        """Stage a full M x K INT8 tile, then pulse CTRL.w_load."""
        flat = bytearray(self.k * self.m)
        for m in range(self.m):
            for k in range(self.k):
                flat[m * self.k + k] = weights[m][k] & 0xFF
        for word in range(len(flat) // 4):
            value = int.from_bytes(flat[word * 4:word * 4 + 4], "little")
            await self.axil.write_dword(WEIGHT_BASE + word * 4, value)

        await self.axil.write_dword(ADDR_CTRL, 1)
        # The load FSM takes K cycles; poll STATUS.w_loaded rather than guess.
        for _ in range(self.k + 32):
            status = await self.axil.read_dword(ADDR_STATUS)
            if status & 0x1:
                return
        raise AssertionError("STATUS.w_loaded never set after CTRL.w_load")

    def pack_vectors(self, vectors):
        out = bytearray()
        for vec in vectors:
            for value in vec:
                out.append(value & 0xFF)
        return bytes(out)

    def unpack_results(self, data):
        stride = self.m * 4
        assert len(data) % stride == 0, (
            f"result frame is {len(data)} bytes, not a multiple of {stride}"
        )
        out = []
        for beat in range(len(data) // stride):
            chunk = data[beat * stride:(beat + 1) * stride]
            out.append([
                to_signed(int.from_bytes(chunk[i * 4:(i + 1) * 4], "little"), 32)
                for i in range(self.m)
            ])
        return out

    async def run_vectors(self, vectors):
        await self.source.send(self.pack_vectors(vectors))
        frame = await self.sink.recv()
        return self.unpack_results(bytes(frame.tdata))


@cocotb.test()
async def test_id_and_config(dut):
    """S1: the identity and config registers describe this elaboration."""
    tb = Harness(dut)
    await tb.reset()

    ident = await tb.axil.read_dword(ADDR_ID)
    assert ident == ID_VALUE, f"ID register read 0x{ident:08X}, expected 0x{ID_VALUE:08X}"

    cfg = await tb.axil.read_dword(ADDR_CFG)
    assert cfg & 0xFF == tb.k, f"CFG.K={cfg & 0xFF}, port width implies {tb.k}"
    assert (cfg >> 8) & 0xFF == tb.m, f"CFG.M={(cfg >> 8) & 0xFF}, expected {tb.m}"
    assert (cfg >> 16) & 0xFF == ACC_W


@cocotb.test()
async def test_weight_staging_readback(dut):
    """S2: staged weight bytes survive the write path exactly."""
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(5)

    flat = [rng.randint(0, 255) for _ in range(tb.k * tb.m)]
    for word in range(len(flat) // 4):
        value = int.from_bytes(bytes(flat[word * 4:word * 4 + 4]), "little")
        await tb.axil.write_dword(WEIGHT_BASE + word * 4, value)

    for word in range(len(flat) // 4):
        got = await tb.axil.read_dword(WEIGHT_BASE + word * 4)
        want = int.from_bytes(bytes(flat[word * 4:word * 4 + 4]), "little")
        assert got == want, (
            f"weight word {word}: read 0x{got:08X}, wrote 0x{want:08X}"
        )


@cocotb.test()
async def test_end_to_end(dut):
    """S3: the whole flow, against the Python reference."""
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(0xA11CE)

    weights = random_weights(tb.m, tb.k, rng)
    await tb.write_weights(weights)

    vectors = random_activations(tb.k, 32, rng)
    got = await tb.run_vectors(vectors)

    assert len(got) == len(vectors), (
        f"sent {len(vectors)} activation beats, received {len(got)} result beats"
    )
    for n, (g, v) in enumerate(zip(got, vectors)):
        want = matvec(weights, v)
        assert g == want, f"vector {n}: got {g}, expected {want}"


@cocotb.test()
async def test_result_readback_register(dut):
    """S4: the AXI4-Lite result latch matches the final streamed vector."""
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(11)

    weights = random_weights(tb.m, tb.k, rng)
    await tb.write_weights(weights)
    vectors = random_activations(tb.k, 8, rng)
    got = await tb.run_vectors(vectors)

    for _ in range(8):
        await RisingEdge(dut.clk)

    last = got[-1]
    for m in range(tb.m):
        raw = await tb.axil.read_dword(RESULT_BASE + m * 4)
        assert to_signed(raw, 32) == last[m], (
            f"RESULT[{m}] register reads {to_signed(raw, 32)}, "
            f"stream gave {last[m]}"
        )


@cocotb.test()
async def test_result_backpressure(dut):
    """S5: a slow consumer must cost throughput, never data.

    The array cannot stall, so this is the test that proves axis_result_fifo is
    actually absorbing the gap rather than the design quietly dropping beats.
    """
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(0xBADC0DE)

    weights = random_weights(tb.m, tb.k, rng)
    await tb.write_weights(weights)

    # Accept roughly one beat in three.
    tb.sink.set_pause_generator(iter(lambda: rng.random() < 0.66, None))

    vectors = random_activations(tb.k, 12, rng)
    got = await tb.run_vectors(vectors)

    assert len(got) == len(vectors), (
        f"back-pressure lost beats: sent {len(vectors)}, got {len(got)}"
    )
    for n, (g, v) in enumerate(zip(got, vectors)):
        assert g == matvec(weights, v), f"vector {n} corrupted under back-pressure"

    status = await tb.axil.read_dword(ADDR_STATUS)
    assert not (status & 0x10), (
        "STATUS.overflow set: the result FIFO dropped data. Either deepen "
        "FIFO_DEPTH or slow the producer."
    )


@cocotb.test()
async def test_tready_low_during_weight_load(dut):
    """S6: the tile refuses activations while the weight tile is shifting.

    Without this the host could stream into a half-loaded array and get
    plausible-looking but wrong results, which is the hardest class of bug to
    notice downstream.
    """
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(3)

    weights = random_weights(tb.m, tb.k, rng)
    flat = bytearray(tb.k * tb.m)
    for m in range(tb.m):
        for k in range(tb.k):
            flat[m * tb.k + k] = weights[m][k] & 0xFF
    for word in range(len(flat) // 4):
        await tb.axil.write_dword(
            WEIGHT_BASE + word * 4,
            int.from_bytes(flat[word * 4:word * 4 + 4], "little"),
        )

    await tb.axil.write_dword(ADDR_CTRL, 1)

    saw_busy = False
    for _ in range(tb.k + 8):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if str(dut.s_axis_tready.value) == "0":
            saw_busy = True
            break
    assert saw_busy, "s_axis_tready never went low during the weight load"

    while str(dut.s_axis_tready.value) == "0":
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")

    vectors = random_activations(tb.k, 4, rng)
    got = await tb.run_vectors(vectors)
    for n, (g, v) in enumerate(zip(got, vectors)):
        assert g == matvec(weights, v), f"vector {n} wrong after weight load"


@cocotb.test()
async def test_second_weight_tile(dut):
    """S7: reloading weights between bursts, which is the real tiling loop.

    conv3 needs 8 x 36 tiles on a 16x16 array, so the weight reload path runs
    288 times per layer. It has to work without a reset.
    """
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(0xFEED)

    for round_idx in range(3):
        weights = random_weights(tb.m, tb.k, rng)
        await tb.write_weights(weights)
        vectors = random_activations(tb.k, 6, rng)
        got = await tb.run_vectors(vectors)
        for n, (g, v) in enumerate(zip(got, vectors)):
            assert g == matvec(weights, v), (
                f"tile {round_idx}, vector {n}: stale weights from the previous tile?"
            )


@cocotb.test()
async def test_unmapped_write_returns_slverr(dut):
    """S8: an unmapped write must be reported, not silently accepted.

    cocotbext-axi's AXI4-Lite master does not raise on a non-OKAY response, it
    hands the response back, so this checks BRESP directly. An accelerator that
    answers OKAY to a write it dropped is worse than one that fails loudly --
    a mistyped weight base address would look like an arithmetic bug.
    """
    tb = Harness(dut)
    await tb.reset()

    ok = await tb.axil.write(ADDR_CTRL, (0).to_bytes(4, "little"))
    assert int(ok.resp) == 0, f"mapped write returned BRESP={int(ok.resp)}, expected OKAY"

    bad = await tb.axil.write(0x0F00, (0xDEADBEEF).to_bytes(4, "little"))
    assert int(bad.resp) == 2, (
        f"write to unmapped 0x0F00 returned BRESP={int(bad.resp)}, expected "
        "SLVERR (2). The address decoder is accepting writes it cannot store."
    )


@cocotb.test()
async def test_overlapped_weight_load(dut):
    """S9: load the next tile's weights WHILE the current tile streams.

    This is the M3 pipelining win. The shift chain writes each PE's SHADOW
    register, which the array is not multiplying by, so a load no longer has to
    stop the stream -- s_axis_tready stays high throughout a non-blocking load.
    rtl/sim/tile_schedule.py measures what that removes: 93.9% -> 99.5% array
    utilisation, 1.550 ms -> 1.462 ms per inference.

    Software contract, and it matters:
        CTRL[0] load + auto-commit (blocking, simple path)
        CTRL[1] load only, non-blocking -- overlaps the current tile's stream
        CTRL[2] commit shadow -> active
    A commit must wait for STATUS.w_loaded. The hardware defers an early one
    rather than splicing tiles (see S10), but the host still has to not stream
    the NEW tile's vectors until the commit has actually landed.
    """
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(0x0FFA1)

    weights_a = random_weights(tb.m, tb.k, rng)
    weights_b = random_weights(tb.m, tb.k, rng)
    vectors_a = random_activations(tb.k, 24, rng)
    vectors_b = random_activations(tb.k, 24, rng)

    await tb.write_weights(weights_a)
    await tb.stage_weights(weights_b)

    # Start shifting B, then immediately stream A. The two overlap: tready must
    # stay high and A's results must be untouched by B's shift.
    await tb.axil.write_dword(ADDR_CTRL, 0b010)
    got_a = await tb.run_vectors(vectors_a)

    assert len(got_a) == len(vectors_a), (
        f"tile A returned {len(got_a)} beats, expected {len(vectors_a)}; "
        "the non-blocking load should never lower s_axis_tready"
    )
    for n, (g, v) in enumerate(zip(got_a, vectors_a)):
        assert g == matvec(weights_a, v), (
            f"tile A vector {n} wrong: shifting B's weights disturbed the "
            "active weights A was still using"
        )

    await tb.wait_loaded()
    await tb.axil.write_dword(ADDR_CTRL, 0b100)      # commit B
    for _ in range(4):
        await RisingEdge(dut.clk)

    got_b = await tb.run_vectors(vectors_b)
    for n, (g, v) in enumerate(zip(got_b, vectors_b)):
        want_b = matvec(weights_b, v)
        if g != want_b:
            which = ("still tile A -- the commit never landed"
                     if g == matvec(weights_a, v)
                     else "SPLICED -- holds neither tile cleanly")
            raise AssertionError(
                "tile B vector {}: {} | got {} | want B {}".format(
                    n, which, g[:4], want_b[:4]))


@cocotb.test()
async def test_commit_during_shift_is_deferred(dut):
    """S10: a commit issued mid-shift must not splice two weight tiles.

    Committing while the shift chain is still moving would leave the rows
    already shifted holding the new tile and the rest holding the old one. The
    array then computes with weights that match NEITHER tile, and the result is
    still plausible-looking numbers -- nothing downstream flags it. The
    hardware defers such a commit to the end of the shift instead.

    This is a real hazard, not a hypothetical: it is how the overlapped-load
    test first failed, reporting "NEITHER A NOR B".
    """
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(0x5B1CE)

    weights_a = random_weights(tb.m, tb.k, rng)
    weights_b = random_weights(tb.m, tb.k, rng)
    await tb.write_weights(weights_a)

    # Shift B non-blocking, then commit IMMEDIATELY -- mid-shift by design.
    await tb.stage_weights(weights_b)
    await tb.axil.write_dword(ADDR_CTRL, 0b010)
    await tb.axil.write_dword(ADDR_CTRL, 0b100)

    for _ in range(tb.k + 16):
        await RisingEdge(dut.clk)

    vectors = random_activations(tb.k, 6, rng)
    got = await tb.run_vectors(vectors)

    for n, (g, v) in enumerate(zip(got, vectors)):
        want_b = matvec(weights_b, v)
        if g != want_b:
            want_a = matvec(weights_a, v)
            which = ("held tile A" if g == want_a
                     else "SPLICED: holds neither tile cleanly")
            raise AssertionError(
                "vector {}: {} | got {} | want B {}".format(
                    n, which, g[:4], want_b[:4]))


@cocotb.test()
async def test_no_floating_signals(dut):
    """S11: no scalar or vector signal reads X or Z after a full operation.

    An undriven net simulates as X and only becomes a hard problem at
    place-and-route, so this is the cheapest place to catch one. Memory arrays
    with unwritten locations are reported but not failed -- RAM powers up
    undefined in silicon too, and reading what you never wrote is a software
    bug. See tb/xscan.py.
    """
    tb = Harness(dut)
    await tb.reset()
    rng = random.Random(0xF10A7)

    weights = random_weights(tb.m, tb.k, rng)
    await tb.write_weights(weights)
    await tb.run_vectors(random_activations(tb.k, 8, rng))
    for _ in range(8):
        await RisingEdge(dut.clk)

    assert_no_floating(dut, "tile_top")
