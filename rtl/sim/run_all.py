"""Run the whole M2 + M3 regression and write rtl/sim/final_run.log.

    python rtl/sim/run_all.py

Covers, in order of increasing integration:
    pe_int8    single MAC cell
    mac_array  1x1, 4x4 and 16x16 systolic arrays
    tile_top   4x4 and 16x16 behind AXI4-Lite and AXI4-Stream
"""

import io
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # rtl/
SIM = ROOT / "sim"

JOBS = [
    # --- M2: the compute tile ---------------------------------------------
    ("pe_int8  single MAC cell", [sys.executable, str(SIM / "run_pe.py")]),
    ("mac_array  1x1  PIPE=1", [sys.executable, str(SIM / "run_array.py"), "1", "1", "1"]),
    ("mac_array  4x4  PIPE=1", [sys.executable, str(SIM / "run_array.py"), "4", "4", "1"]),
    ("mac_array  16x16 PIPE=1", [sys.executable, str(SIM / "run_array.py"), "16", "16", "1"]),
    ("tile_top  4x4  PIPE=1", [sys.executable, str(SIM / "run_top.py"), "4", "4", "1"]),
    ("tile_top  16x16 PIPE=1", [sys.executable, str(SIM / "run_top.py"), "16", "16", "1"]),
    # --- M3: deeper pipeline and the banded scratchpad ---------------------
    ("mac_array  4x4  PIPE=2", [sys.executable, str(SIM / "run_array.py"), "4", "4", "2"]),
    ("mac_array  16x16 PIPE=2", [sys.executable, str(SIM / "run_array.py"), "16", "16", "2"]),
    ("tile_top  16x16 PIPE=2", [sys.executable, str(SIM / "run_top.py"), "16", "16", "2"]),
    ("tile_top  4x4  PIPE=2", [sys.executable, str(SIM / "run_top.py"), "4", "4", "2"]),
    ("band_sram  scratchpad", [sys.executable, str(SIM / "run_band.py")]),
    ("writeback  BN+ReLU+pool", [sys.executable, str(SIM / "run_writeback.py")]),
    # --- integration: the whole datapath under one sequencer ---------------
    ("layer_top  INTEGRATED", [sys.executable, str(SIM / "run_layer.py")]),
]


def main():
    log = io.StringIO()
    summary = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    log.write(f"M2 regression -- started {started}\n")

    overall_ok = True
    for name, cmd in JOBS:
        print(f"--- {name} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(ROOT.parent))
        out = proc.stdout + proc.stderr
        log.write(f"\n{'=' * 78}\n{name}\n{'=' * 78}\n{out}")

        tally = [ln for ln in out.splitlines() if "TESTS=" in ln]
        passed = failed = 0
        for ln in tally:
            for token in ln.replace("*", " ").split():
                if token.startswith("PASS="):
                    passed += int(token[5:])
                elif token.startswith("FAIL="):
                    failed += int(token[5:])
        ok = failed == 0 and passed > 0 and proc.returncode == 0
        overall_ok &= ok
        summary.append((name, passed, failed, ok))
        print(f"    {'PASS' if ok else 'FAIL'}  ({passed} passed, {failed} failed)")

    header = ["", "=" * 78, "M2 + M3 SUMMARY", "=" * 78,
              f"{'block':<28}{'passed':>8}{'failed':>8}   status"]
    total_p = total_f = 0
    for name, p, f, ok in summary:
        header.append(f"{name:<28}{p:>8}{f:>8}   {'PASS' if ok else 'FAIL'}")
        total_p += p
        total_f += f
    header += ["-" * 78,
               f"{'TOTAL':<28}{total_p:>8}{total_f:>8}   "
               f"{'ALL PASS' if overall_ok else 'FAILURES PRESENT'}", "=" * 78]

    text = log.getvalue() + "\n".join(header) + "\n"
    (SIM / "final_run.log").write_text(text, encoding="utf-8")
    print("\n".join(header))
    print(f"\nFull log: {SIM / 'final_run.log'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
