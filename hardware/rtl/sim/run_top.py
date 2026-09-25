"""Run the tile_top cocotb suite (AXI4-Lite + AXI4-Stream) under Icarus.

    python rtl/sim/run_top.py            # K=16, M=16, PIPE=1, the design point
    python rtl/sim/run_top.py 4 4        # smaller build, faster to debug
    python rtl/sim/run_top.py 16 16 2    # M3 two-stage PE
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent      # rtl/
RTL = ROOT / "rtl_design"
TB = ROOT / "tb"

SOURCES = ["pe_int8.sv", "mac_array.sv", "axis_result_fifo.sv", "tile_top.sv"]


def main():
    if len(sys.argv) >= 3:
        k, m = int(sys.argv[1]), int(sys.argv[2])
        pipe = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    else:
        k, m, pipe = 16, 16, 1

    tag = f"{k}x{m}_p{pipe}"
    bar = "=" * 70
    print("")
    print(bar)
    print(f"  tile_top  K={k}  M={m}  PIPE={pipe}   (AXI4-Lite + AXI4-Stream)")
    print(bar)

    build_dir = ROOT / "sim" / f"build_top_{tag}"
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=[RTL / s for s in SOURCES],
        hdl_toplevel="tile_top",
        parameters={"K": k, "M": m, "PIPE": pipe},
        always=True,
        build_args=["-g2012"],
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        waves=True,
    )
    results = runner.test(
        hdl_toplevel="tile_top",
        test_module="test_top",
        test_dir=TB,
        build_dir=build_dir,
        results_xml=f"results_top_{tag}.xml",
        timescale=("1ns", "1ps"),
        waves=True,
    )
    print(f"\nResults XML: {results}")


if __name__ == "__main__":
    sys.exit(main())
