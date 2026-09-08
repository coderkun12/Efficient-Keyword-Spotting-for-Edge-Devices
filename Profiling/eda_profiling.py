"""
Layer profiling for the keyword-spotting CNN.

Dumps a torchinfo layer summary (output shapes, params, mult-adds) plus the
machine it was measured on to Profiling/eda_profiling.txt.

Runs standalone -- synthetic input, no dataset download, no checkpoint needed.

Requirements:
    pip install torch torchinfo
    (torchaudio is optional here -- it is only used to read the mel config from
     src/dataset.py; without it the script falls back to those same constants.)

Usage:
    python Profiling/eda_profiling.py
    python Profiling/eda_profiling.py --batch-size 1 --device cuda
"""

import argparse
import contextlib
import io
import platform
import sys
import time
from pathlib import Path

# --- dependency check, so a missing package gives a useful message ----------
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
    sys.exit(
        "Missing required package(s): " + ", ".join(_MISSING) + "\n"
        "Install with:  pip install " + " ".join(_MISSING) + "\n"
        "(for torch, see https://pytorch.org/get-started/locally/ to get the "
        "right CUDA build for your machine)"
    )

# Make src/ importable regardless of where the script is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import KeywordSpottingCNN  # noqa: E402


def load_dataset_config():
    """
    src/dataset.py holds the input-shape config (n_mels, num_classes) but
    importing it pulls in torchaudio. Fall back to a mirror of its constants
    so the model-side profiling still runs without torchaudio installed.
    """
    try:
        import dataset
        return dataset.N_MELS, dataset.NUM_CLASSES, True
    except Exception:
        return 40, 12, False


N_MELS, NUM_CLASSES, DATASET_IMPORTED = load_dataset_config()

SEPARATOR = "=" * 90


def parse_args():
    p = argparse.ArgumentParser(description="Profile the KWS model layer by layer")
    p.add_argument("--batch-size", type=int, default=1,
                   help="batch size for the profiled forward pass (default: 1)")
    p.add_argument("--time-frames", type=int, default=101,
                   help="mel time frames per clip (101 for a 1 s clip at 10 ms hop)")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                   help="force a device (default: cuda if available, else cpu)")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).resolve().parent / "eda_profiling.txt"))
    return p.parse_args()


def environment_lines(device, input_shape):
    lines = [
        f"Generated           : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Machine             : {platform.node()}",
        f"Platform            : {platform.platform()}",
        f"Processor           : {platform.processor() or 'unknown'}",
        f"Python              : {platform.python_version()}",
        f"torch               : {torch.__version__}",
        f"Profiling device    : {device}",
    ]
    if device.type == "cuda":
        lines.append(f"CUDA device         : {torch.cuda.get_device_name(0)}")
    else:
        lines.append(f"torch threads       : {torch.get_num_threads()}")
    lines.append(f"Model input shape   : {tuple(input_shape)}  (B, C, n_mels, time)")
    lines.append(f"Output classes      : {NUM_CLASSES}")
    if not DATASET_IMPORTED:
        lines.append("NOTE                : src/dataset.py not importable (torchaudio "
                     "missing); input shape uses its default constants.")
    return lines


def main():
    args = parse_args()
    torch.manual_seed(42)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = KeywordSpottingCNN(num_classes=NUM_CLASSES).to(device)
    model.eval()

    input_shape = (args.batch_size, 1, N_MELS, args.time_frames)

    # Capture stdout rather than passing print_fn: some torchinfo versions
    # forward unexpected kwargs to the model call when print_fn is used.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary(
            model,
            input_size=input_shape,
            col_names=("output_size", "num_params", "mult_adds"),
            device=device,
            verbose=1,
        )

    report = "\n".join([
        SEPARATOR,
        "EDA PROFILING REPORT -- Efficient Keyword Spotting for Edge Devices",
        "Model: KeywordSpottingCNN (baseline FP32, randomly initialised weights)",
        SEPARATOR,
        *environment_lines(device, input_shape),
        "",
        buf.getvalue(),
    ]) + "\n"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    # torchinfo draws the table with box characters; a legacy console codepage
    # (cp1252 on Windows) cannot encode them. The file is always written as
    # UTF-8 -- only the echo to stdout needs the fallback.
    try:
        print(report)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(report.encode(enc, errors="replace").decode(enc))

    print(f"Wrote profiling report to {out_path}")


if __name__ == "__main__":
    main()
