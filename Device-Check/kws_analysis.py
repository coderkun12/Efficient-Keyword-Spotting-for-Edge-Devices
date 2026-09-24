"""
Layer analysis for the keyword-spotting CNN: top layers by MAC count and the
arithmetic intensity of the most MAC-intensive layer.

Writes Profiling/kws_analysis.md.

Requirements:
    pip install torch torchinfo

Usage:
    python Profiling/kws_analysis.py
    python Profiling/kws_analysis.py --top 5 --batch-size 1
"""

import argparse
import sys
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
    sys.exit(
        "Missing required package(s): " + ", ".join(_MISSING) + "\n"
        "Install with:  pip install " + " ".join(_MISSING) + "\n"
        "(for torch, see https://pytorch.org/get-started/locally/ to get the "
        "right CUDA build for your machine)"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import KeywordSpottingCNN  # noqa: E402


def load_dataset_config():
    """Input-shape config from src/dataset.py, with a torchaudio-free fallback."""
    try:
        import dataset
        return dataset.N_MELS, dataset.NUM_CLASSES
    except Exception:
        return 40, 12


N_MELS, NUM_CLASSES = load_dataset_config()

BYTES_PER_ELEM = 4  # float32


def parse_args():
    p = argparse.ArgumentParser(description="Analyse KWS model layers by MAC count")
    p.add_argument("--top", type=int, default=5, help="how many layers to rank (default: 5)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--time-frames", type=int, default=101,
                   help="mel time frames per clip (101 for a 1 s clip at 10 ms hop)")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).resolve().parent / "kws_analysis.md"))
    return p.parse_args()


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main():
    args = parse_args()
    torch.manual_seed(42)

    model = KeywordSpottingCNN(num_classes=NUM_CLASSES)
    model.eval()

    input_shape = (args.batch_size, 1, N_MELS, args.time_frames)
    result = summary(model, input_size=input_shape, verbose=0)

    # Keep only leaf layers with non-zero MACs
    layers = [li for li in result.summary_list if li.is_leaf_layer and li.macs > 0]

    total_macs = sum(li.macs for li in layers)
    top = sorted(layers, key=lambda li: li.macs, reverse=True)[:args.top]

    lines = [
        "# Keyword-Spotting CNN Layer Analysis",
        "",
        f"**Model:** `KeywordSpottingCNN` ({result.total_params:,} parameters, "
        f"{NUM_CLASSES} classes)  ",
        f"**Input:** `{list(input_shape)}` (batch, channel, n_mels, time frames) "
        "-- one 1-second clip as a 40-bin log-mel spectrogram",
        "",
        f"## Top {args.top} Layers by MAC Count",
        "",
        "| Rank | Layer Name | MACs | FLOPs (2xMACs) | Parameters | % of Model MACs |",
        "| ---- | ---------- | ---- | -------------- | ---------- | --------------- |",
    ]

    for rank, li in enumerate(top, 1):
        name = li.get_layer_name(show_var_name=True, show_depth=True)
        share = 100.0 * li.macs / total_macs if total_macs else 0.0
        lines.append(
            f"| {rank} | `{name}` | {li.macs:,} | {li.macs * 2:,} | "
            f"{li.num_params:,} | {share:.1f}% |"
        )

    lines += [
        "",
        f"Total MACs across all layers: **{total_macs:,}** "
        f"({total_macs / 1e6:.2f} M) per inference.",
    ]

    # --- Arithmetic Intensity for the #1 MAC layer ---
    top1 = top[0]
    top_name = top1.get_layer_name(show_var_name=True, show_depth=True)

    input_elems = numel(top1.input_size)
    output_elems = numel(top1.output_size)
    weight_elems = top1.num_params

    input_bytes = input_elems * BYTES_PER_ELEM
    output_bytes = output_elems * BYTES_PER_ELEM
    weight_bytes = weight_elems * BYTES_PER_ELEM
    total_bytes = input_bytes + output_bytes + weight_bytes

    flops_val = top1.macs * 2
    arith_intensity = flops_val / total_bytes

    lines += [
        "",
        "---",
        "",
        "## Arithmetic Intensity -- Most MAC-Intensive Layer",
        "",
        f"**Layer:** `{top_name}`",
        "",
        "### Assumptions",
        "- All weights and activations are loaded from DRAM with **no reuse**.",
        "- Data type: **float32** (4 bytes per element).",
        "",
        "### Memory Traffic",
        "",
        "| Tensor | Shape | Elements | Bytes (x4) |",
        "| ------ | ----- | -------- | ---------- |",
        f"| Input activations  | `{list(top1.input_size)}`  | {input_elems:,}  | {input_bytes:,}  |",
        f"| Output activations | `{list(top1.output_size)}` | {output_elems:,} | {output_bytes:,} |",
        f"| Weights            | --                         | {weight_elems:,} | {weight_bytes:,} |",
        f"| **Total**          |                            |                  | **{total_bytes:,}** |",
        "",
        "### Calculation",
        "",
        "```",
        f"FLOPs  = 2 x MACs = 2 x {top1.macs:,} = {flops_val:,}",
        "",
        "DRAM bytes = input + output + weights",
        f"           = {input_bytes:,} + {output_bytes:,} + {weight_bytes:,}",
        f"           = {total_bytes:,} bytes",
        "",
        "Arithmetic Intensity = FLOPs / DRAM bytes",
        f"                     = {flops_val:,} / {total_bytes:,}",
        f"                     = {arith_intensity:.2f} FLOPs/byte",
        "```",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote analysis to {out_path}")


if __name__ == "__main__":
    main()
