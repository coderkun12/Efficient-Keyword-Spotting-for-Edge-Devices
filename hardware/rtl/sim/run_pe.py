"""Run the pe_int8 cocotb suite under Icarus Verilog.

    python rtl/sim/run_pe.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent      # rtl/
RTL = ROOT / "rtl_design"
TB = ROOT / "tb"


def main():
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=[RTL / "pe_int8.sv"],
        hdl_toplevel="pe_int8",
        always=True,
        build_args=["-g2012"],
        build_dir=ROOT / "sim" / "build_pe",
        waves=True,
        timescale=("1ns", "1ps"),
    )
    results = runner.test(
        hdl_toplevel="pe_int8",
        test_module="test_pe",
        test_dir=TB,
        build_dir=ROOT / "sim" / "build_pe",
        waves=True,
        timescale=("1ns", "1ps"),
    )
    print(f"\nResults XML: {results}")


if __name__ == "__main__":
    sys.exit(main())
