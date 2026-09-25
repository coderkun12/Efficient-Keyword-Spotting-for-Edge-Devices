"""Run the layer_top suite at the EXACT parameters the FPGA build uses.

run_layer.py exercises the logic with small memories -- MAX_KTILES=4,
BAND_DEPTH=256 -- because that is enough to prove the sequencing and it keeps
the regression fast. Those are not the sizes that go on the board.

The difference matters. Address widths are $clog2 of these parameters, so
BAND_DEPTH 256 -> 3232 widens every band address from 8 bits to 12, and
MAX_KTILES 4 -> 72 widens the weight word address from 6 bits to 11. Width
changes are exactly where off-by-one truncation hides, and a design that
passes at one size can fail at another without a single line of logic
differing.

So this runs the same 8 tests against the real configuration:

    MAXW       = 32     conv4 is W=25; sizes rowacc at 32 x 448 flops
    MAX_KTILES = 72     conv4 has 1152 taps = 72 k-tiles (the maximum)
    BAND_DEPTH = 3232   max channels x width across conv2/3/4

    python rtl/sim/run_layer_fpga.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl_design"
TB = ROOT / "tb"

SOURCES = ["pe_int8.sv", "mac_array.sv", "byte_ram.sv", "band_sram.sv", "writeback.sv",
           "layer_top.sv"]

PARAMS = {"PIPE": 1, "MAXW": 32, "MAX_KTILES": 72, "BAND_DEPTH": 3232}


def main():
    build_dir = ROOT / "sim" / "build_layer_fpga"
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=[RTL / s for s in SOURCES],
        hdl_toplevel="layer_top",
        parameters=PARAMS,
        always=True,
        build_args=["-g2012"],
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        waves=True,
    )
    runner.test(
        hdl_toplevel="layer_top",
        test_module="test_layer_top",
        test_dir=TB,
        build_dir=build_dir,
        results_xml="results_layer_fpga.xml",
        timescale=("1ns", "1ps"),
        waves=True,
    )


if __name__ == "__main__":
    main()
