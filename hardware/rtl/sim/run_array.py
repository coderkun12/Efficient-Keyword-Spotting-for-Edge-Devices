"""Run the mac_array cocotb suite under Icarus Verilog.

Builds the array at several shapes and pipeline depths, because different bugs
surface at different elaborations: a 1x1 build isolates the PE wiring, 4x4
makes the systolic skew hand-checkable, 16x16 is the design point, and PIPE=2
exercises the doubled row skew that M3 added for timing closure.

    python rtl/sim/run_array.py              # all shapes
    python rtl/sim/run_array.py 16 16        # one shape, PIPE=1
    python rtl/sim/run_array.py 16 16 2      # one shape, PIPE=2
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent      # rtl/
RTL = ROOT / "rtl_design"
TB = ROOT / "tb"

SHAPES = [(1, 1, 1), (4, 4, 1), (16, 16, 1), (4, 4, 2), (16, 16, 2)]


def run_shape(k, m, pipe=1):
    tag = f"{k}x{m}_p{pipe}"
    bar = "=" * 70
    print("")
    print(bar)
    print(f"  mac_array  K={k} (reduction depth)  M={m} (output channels)  PIPE={pipe}")
    print(bar)

    build_dir = ROOT / "sim" / f"build_array_{tag}"
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=[RTL / "pe_int8.sv", RTL / "mac_array.sv"],
        hdl_toplevel="mac_array",
        parameters={"K": k, "M": m, "PIPE": pipe},
        always=True,
        build_args=["-g2012"],
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        waves=True,
    )
    return runner.test(
        hdl_toplevel="mac_array",
        test_module="test_array",
        test_dir=TB,
        build_dir=build_dir,
        results_xml=f"results_array_{tag}.xml",
        timescale=("1ns", "1ps"),
        waves=True,
    )


def main():
    if len(sys.argv) >= 3:
        pipe = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        shapes = [(int(sys.argv[1]), int(sys.argv[2]), pipe)]
    else:
        shapes = SHAPES
    for k, m, pipe in shapes:
        run_shape(k, m, pipe)
    print("\nAll shapes built and run. Check the per-shape summaries above.")


if __name__ == "__main__":
    sys.exit(main())
