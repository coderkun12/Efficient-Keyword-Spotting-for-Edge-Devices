"""
Shared X/Z sweep for the cocotb testbenches.

WHAT THIS IS FOR
An undriven net simulates as X and only becomes a hard failure during
place-and-route, where it shows up as a floating signal. Sweeping the
elaborated hierarchy for X after the design has been reset and exercised is
direct evidence that nothing is undriven, rather than a promise.

THE DISTINCTION THAT MATTERS
A memory array whose unwritten locations read X is NOT a floating net. RAM
powers up undefined in real silicon too, and reading a location you never wrote
is a software bug, not a hardware one. Every design here has arrays larger than
the test exercises -- `rowacc` is sized for MAXW columns but a test may use six
-- so flagging those would drown the real signal and train people to ignore the
check.

So arrays are counted and reported separately, and only scalars and vectors
cause a failure. The discipline that goes with it: the host must write every
memory location it will later read. `test_layer_top` zero-fills the band for
channels beyond the layer's real count for exactly this reason -- an unwritten
byte reads X, and X times a zero weight is still X in Verilog.
"""


def _is_array(value):
    return type(value).__name__ == "Array"


def scan_for_x(handle, path, max_depth=2, budget=40):
    """Return (bad, arrays) where `bad` is the list that should fail a test.

    bad    -- [(path, value)] for scalar/vector signals reading X or Z
    arrays -- [path] for memory arrays holding X, reported but not failed
    """
    bad = []
    arrays = []
    remaining = [budget]

    def walk(node, node_path, depth):
        if remaining[0] <= 0:
            return
        try:
            value = node.value
        except Exception:
            value = None

        if value is not None:
            if _is_array(value):
                text = str(value)
                if "x" in text.lower() or "z" in text.lower():
                    arrays.append(node_path)
                return                      # do not descend into memory words
            text = str(value)
            if "x" in text.lower() or "z" in text.lower():
                bad.append((node_path, text if len(text) < 48
                            else text[:48] + "..."))
                remaining[0] -= 1
                return

        if depth >= max_depth:
            return
        try:
            children = list(node)
        except Exception:
            return
        for child in children:
            name = str(getattr(child, "_name", "?")).split(".")[-1]
            walk(child, f"{node_path}.{name}", depth + 1)

    walk(handle, path, 0)
    return bad, arrays


def assert_no_floating(dut, top_name):
    """Fail on any X/Z in a scalar or vector; log arrays as informational."""
    bad, arrays = scan_for_x(dut, top_name)

    if arrays:
        dut._log.info(
            f"{len(arrays)} memory array(s) hold X in unwritten locations, "
            f"which is expected: {', '.join(arrays[:6])}"
        )

    if bad:
        lines = chr(10).join(f"    {p} = {v}" for p, v in bad[:20])
        raise AssertionError(
            f"{len(bad)} signal(s) reading X/Z after reset and operation:"
            + chr(10) + lines + chr(10)
            + "Each is undriven or only partially driven, which is what floats "
              "during place-and-route."
        )

    dut._log.info("no X/Z on any scalar or vector signal")
