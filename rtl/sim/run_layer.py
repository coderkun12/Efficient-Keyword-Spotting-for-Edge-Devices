"""Run the integrated layer_top suite (band_sram + array + writeback).

    python rtl/sim/run_layer.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl_design"
TB = ROOT / "tb"

SOURCES = ["pe_int8.sv", "mac_array.sv", "band_sram.sv", "writeback.sv",
           "layer_top.sv"]


def main():
    pipe = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    build_dir = ROOT / "sim" / f"build_layer_p{pipe}"
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=[RTL / s for s in SOURCES],
        hdl_toplevel="layer_top",
        parameters={"PIPE": pipe, "MAXW": 32, "MAX_KTILES": 4,
                    "BAND_DEPTH": 256},
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
        results_xml=f"results_layer_p{pipe}.xml",
        timescale=("1ns", "1ps"),
        waves=True,
    )


if __name__ == "__main__":
    sys.exit(main())
