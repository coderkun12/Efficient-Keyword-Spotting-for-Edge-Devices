"""Run the writeback (requantise + BN + ReLU + max-pool) suite under Icarus.

    python rtl/sim/run_writeback.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl_design"
TB = ROOT / "tb"


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    build_dir = ROOT / "sim" / f"build_wb_{m}"
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=[RTL / "writeback.sv"],
        hdl_toplevel="writeback",
        parameters={"M": m},
        always=True,
        build_args=["-g2012"],
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        waves=True,
    )
    runner.test(
        hdl_toplevel="writeback",
        test_module="test_writeback",
        test_dir=TB,
        build_dir=build_dir,
        results_xml=f"results_wb_{m}.xml",
        timescale=("1ns", "1ps"),
        waves=True,
    )


if __name__ == "__main__":
    sys.exit(main())
