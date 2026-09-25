"""Run the band_sram cocotb suite under Icarus Verilog.

    python rtl/sim/run_band.py

DEPTH defaults to 3232 bytes per row bank, the max of channels x width across
conv2 (32 x 101), conv3 (64 x 50) and conv4 (128 x 25). Three row banks, double
buffered, is 19,392 B for the input band.
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl_design"
TB = ROOT / "tb"


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3232
    build_dir = ROOT / "sim" / f"build_band_{depth}"
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=[RTL / "band_sram.sv"],
        hdl_toplevel="band_sram",
        parameters={"DEPTH": depth},
        always=True,
        build_args=["-g2012"],
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        waves=True,
    )
    runner.test(
        hdl_toplevel="band_sram",
        test_module="test_band_sram",
        test_dir=TB,
        build_dir=build_dir,
        results_xml=f"results_band_{depth}.xml",
        timescale=("1ns", "1ps"),
        waves=True,
    )


if __name__ == "__main__":
    sys.exit(main())
