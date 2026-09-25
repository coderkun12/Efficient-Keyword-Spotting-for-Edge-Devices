"""
Host machine vitals for the roofline analysis.

Captures everything needed to derive the two "spec sheet" numbers the roofline
depends on -- CPU peak FLOP/s and peak memory bandwidth -- plus a measured
sustained throughput point, and writes it all to Profiling/host_vitals.txt.

Run this on the machine that produced the trained checkpoints and the
benchmark latencies. Those numbers and this report must come from the SAME
machine or the roofline is internally inconsistent.

Requirements:
    pip install torch torchinfo

Usage:
    python Profiling/host_vitals.py
    python Profiling/host_vitals.py --measure-iters 1200
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_MISSING = []
try:
    import torch
except ImportError:
    _MISSING.append("torch")
try:
    from torchinfo import summary
except ImportError:
    _MISSING.append("torchinfo")
if _MISSING:
    sys.exit("Missing package(s): " + ", ".join(_MISSING)
             + "\nInstall with:  pip install " + " ".join(_MISSING))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from software.src.model import KeywordSpottingCNN  # noqa: E402

try:
    import software.src.dataset as dataset
    N_MELS, NUM_CLASSES = dataset.N_MELS, dataset.NUM_CLASSES
    DATASET_NOTE = "src/dataset.py imported cleanly"
except Exception as exc:
    N_MELS, NUM_CLASSES = 40, 12
    DATASET_NOTE = f"src/dataset.py not importable ({type(exc).__name__}); using defaults 40 / 12"


def parse_args():
    p = argparse.ArgumentParser(description="Capture host vitals for the roofline")
    p.add_argument("--time-frames", type=int, default=101)
    p.add_argument("--measure-iters", type=int, default=800,
                   help="timed batch-1 inferences per thread setting")
    p.add_argument("--warmup-s", type=float, default=5.0,
                   help="seconds of spinning before timing, to settle boost clocks")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "host_vitals.txt"))
    return p.parse_args()


def run(cmd):
    """Run a shell command, return stdout or a short error marker."""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        text = (out.stdout or "").strip()
        return text if text else f"(no output; rc={out.returncode})"
    except Exception as exc:
        return f"(failed: {type(exc).__name__}: {exc})"


def hardware_probe():
    """Platform-specific CPU and memory queries. Raw output, interpreted later."""
    blocks = []
    if platform.system() == "Windows":
        blocks.append((
            "CPU (Win32_Processor)",
            run('powershell -NoProfile -Command "Get-CimInstance Win32_Processor | '
                'Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,'
                'CurrentClockSpeed,L2CacheSize,L3CacheSize | Format-List"')))
        blocks.append((
            "Memory modules (Win32_PhysicalMemory)",
            run('powershell -NoProfile -Command "Get-CimInstance Win32_PhysicalMemory | '
                'Select-Object BankLabel,DeviceLocator,Capacity,Speed,ConfiguredClockSpeed,'
                'SMBIOSMemoryType | Format-List"')))
        blocks.append((
            "Total physical memory",
            run('powershell -NoProfile -Command "(Get-CimInstance Win32_ComputerSystem).'
                'TotalPhysicalMemory"')))
    elif platform.system() == "Linux":
        blocks.append(("CPU (lscpu)", run("lscpu")))
        blocks.append(("Memory (/proc/meminfo, first 5)", run("head -5 /proc/meminfo")))
        blocks.append(("Memory modules (dmidecode, may need root)",
                       run("sudo -n dmidecode -t memory 2>/dev/null | "
                           "grep -E 'Size|Speed|Type:|Locator' | head -40")))
    elif platform.system() == "Darwin":
        blocks.append(("CPU (sysctl)", run("sysctl -n machdep.cpu.brand_string; "
                                           "sysctl -n hw.physicalcpu hw.logicalcpu")))
        blocks.append(("Memory", run("sysctl -n hw.memsize")))
    return blocks


def isa_flags():
    """Vector ISA support -- decides FLOP/cycle/core for the peak calculation."""
    try:
        import cpuinfo  # py-cpuinfo, optional
        info = cpuinfo.get_cpu_info()
        flags = [f for f in info.get("flags", [])
                 if f.startswith(("avx", "sse4", "fma", "amx"))]
        return info.get("brand_raw", "?"), " ".join(sorted(flags)) or "(none reported)"
    except Exception:
        pass
    if platform.system() == "Linux":
        out = run("grep -m1 '^flags' /proc/cpuinfo")
        flags = [f for f in out.split() if f.startswith(("avx", "sse4", "fma", "amx"))]
        return platform.processor(), " ".join(sorted(flags)) or "(none reported)"
    return platform.processor(), "(install py-cpuinfo for ISA flags: pip install py-cpuinfo)"


def measure(model, input_shape, total_flops, threads, iters, warmup_s):
    """Sustained batch-1 latency at a given thread count.

    Reports best / median / p90 rather than one number. Laptop parts boost then
    throttle, so the spread is itself a result: a wide spread means the median
    is contaminated and the best case is the more reproducible figure.
    """
    torch.set_num_threads(threads)
    x = torch.randn(*input_shape)
    model.eval()
    with torch.no_grad():
        t_end = time.perf_counter() + warmup_s
        while time.perf_counter() < t_end:
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    times.sort()
    best = times[0] * 1e3
    med = times[len(times) // 2] * 1e3
    p90 = times[int(len(times) * 0.9)] * 1e3
    return best, med, p90, total_flops / (best / 1e3) / 1e9


def main():
    args = parse_args()
    torch.manual_seed(42)

    model = KeywordSpottingCNN(num_classes=NUM_CLASSES)
    input_shape = (1, 1, N_MELS, args.time_frames)
    res = summary(model, input_size=input_shape, verbose=0)
    total_macs = sum(li.macs for li in res.summary_list
                     if li.is_leaf_layer and li.macs > 0)
    total_flops = 2 * total_macs

    brand, flags = isa_flags()
    L = []
    w = L.append
    bar = "=" * 90

    w(bar)
    w("HOST VITALS -- Efficient Keyword Spotting for Edge Devices")
    w("Inputs for the roofline's CPU peak, memory bandwidth and measured point")
    w(bar)
    w(f"Generated            : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Machine (hostname)   : {platform.node()}")
    w(f"Platform             : {platform.platform()}")
    w(f"Processor (platform) : {platform.processor()}")
    w(f"Processor (brand)    : {brand}")
    w(f"Vector ISA flags     : {flags}")
    w(f"Logical CPUs (os)    : {os.cpu_count()}")
    w(f"Python               : {platform.python_version()}")
    w(f"torch                : {torch.__version__}")
    w(f"torch default threads: {torch.get_num_threads()}")
    try:
        import torchaudio
        w(f"torchaudio           : {torchaudio.__version__}")
    except Exception:
        w("torchaudio           : NOT INSTALLED")
    w(f"dataset config       : {DATASET_NOTE}")
    w(f"Model input shape    : {tuple(input_shape)}  (B, C, n_mels, time)")
    w(f"Model params         : {res.total_params:,}")
    w(f"Model MACs/inference : {total_macs:,}  ({total_flops:,} FLOP)")
    w("")

    w(bar)
    w("SECTION 1 -- HARDWARE PROBE (raw tool output)")
    w("Needed for: CPU peak FLOP/s (cores x FLOP-per-cycle x clock) and")
    w("            peak memory bandwidth (module speed x channels x 8 bytes).")
    w(bar)
    for title, body in hardware_probe():
        w("")
        w(f"--- {title} " + "-" * max(0, 74 - len(title)))
        w(body)
    w("")

    w(bar)
    w("SECTION 2 -- MEASURED SUSTAINED THROUGHPUT (batch 1)")
    w(f"Warmup {args.warmup_s:g} s of continuous inference before timing, so boost clocks")
    w(f"have settled. {args.measure_iters} timed iterations per thread setting.")
    w(bar)
    w("")
    w(f"{'threads':>8} | {'best ms':>9} | {'median ms':>10} | {'p90 ms':>9} | "
      f"{'GFLOP/s @best':>13} | {'spread':>7}")
    w("-" * 78)
    thread_settings = sorted({1, 2, os.cpu_count() or 1, torch.get_num_threads()})
    for t in thread_settings:
        if t < 1:
            continue
        best, med, p90, gflops = measure(model, input_shape, total_flops, t,
                                         args.measure_iters, args.warmup_s)
        w(f"{t:>8} | {best:>9.2f} | {med:>10.2f} | {p90:>9.2f} | "
          f"{gflops:>13.1f} | {p90 / best:>6.1f}x")
    w("")
    w("Reading the spread column: 1.0-1.3x means a quiet, well-cooled machine and")
    w("the median is trustworthy. Above ~2x means the part is throttling or the")
    w("machine is loaded, and only the best-case column is reproducible.")
    w("")
    w("NOTE: the single-threaded row is the one that matters. Keyword spotting is")
    w("an always-on streaming workload classifying one 1 s window at a time, and")
    w("src/benchmark.py measures latency the same way.")
    w("")
    w(bar)
    w("SECTION 3 -- WHAT WE STILL NEED FROM THE SPEC SHEET")
    w(bar)
    w("From Section 1, look up and confirm by hand:")
    w("  1. All-core sustained boost clock (GHz) -- not just the single-core max.")
    w("  2. FMA units per core and vector width, to get FLOP per cycle per core.")
    w("     AVX2 + 1 FMA  = 16 FLOP/cycle/core;  AVX2 + 2 FMA = 32 FLOP/cycle/core.")
    w("  3. Number of populated memory channels (count the modules in Section 1).")
    w("     Peak BW = module speed (MT/s) x channels x 8 bytes/transfer.")
    w(bar)

    out_path = Path(args.out)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
