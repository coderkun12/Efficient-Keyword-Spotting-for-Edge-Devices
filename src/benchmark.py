"""
Benchmark model variants: size, parameter count, MACs, CPU latency, accuracy.
 
This is the single source of truth for comparing baseline / pruned / quantized
checkpoints on the same footing. Because the final target is a real FPGA
implementation (built and measured by a teammate on actual hardware), this
script is also responsible for producing a "hardware handoff report" per
model: per-layer shapes, MAC counts, and weight value ranges. Those numbers
are what someone would need to size a fixed-point/quantized datapath (DSP
slice count, BRAM for weights, expected dynamic range) even before our own
quantize.py exists. The CPU latency numbers here are a *software reference
point*, not a promise of what the FPGA will hit -- the FPGA build is measured
independently on real hardware. Its value is (a) a sanity check that
compression is actually shrinking the model, and (b) an apples-to-apples
number across variants that doesn't depend on the FPGA toolchain.
 
Note on latency methodology: keyword spotting is a streaming, always-on
workload -- one ~1s window is classified at a time, not batches. So latency
is measured at batch size 1, single-threaded, which mirrors how an
always-on accelerator (FPGA or microcontroller) actually gets used, unlike
a throughput-oriented batch-32 benchmark.
 
--------------------------------------------------------------------------
CHECKPOINT CONTRACT (train.py / prune.py / quantize.py all need to follow
this so benchmark.py can load anything generically):
 
Every checkpoint file in --checkpoints-dir must be loadable in one of three
ways. benchmark.py auto-detects which one applies:
 
1. "state_dict" (used by train.py today, and by prune.py for UNSTRUCTURED
   pruning, since the architecture doesn't change):
       torch.save({
           "variant": "baseline",              # human-readable name
           "format": "state_dict",             # optional; inferred if absent
           "model_state_dict": model.state_dict(),
           "num_classes": 12,                  # optional, defaults to NUM_CLASSES
           "in_channels": 1,                   # optional, defaults to 1
           "val_loss": ..., "val_acc": ...,    # optional, just carried through
       }, path)
   IMPORTANT: if unstructured pruning is applied via torch.nn.utils.prune,
   call prune.remove(module, "weight") on every pruned module BEFORE saving,
   so the checkpoint contains plain zeroed weight tensors and loads straight
   into a vanilla KeywordSpottingCNN via load_state_dict.
 
2. "full_model" (needed for STRUCTURED pruning, since channel counts change
   and a fresh KeywordSpottingCNN(num_classes) default constructor no
   longer matches the saved shapes):
       torch.save(model, path)   # the nn.Module object itself, no wrapper dict
 
3. "torchscript" (used by quantize.py for PTQ/QAT, since quantized modules
   have a different structure than the float model):
       scripted = torch.jit.script(quantized_model)
       torch.jit.save(scripted, path)
 
Files are auto-detected in that order (state_dict dict -> full_model object
-> torchscript). Variant name defaults to the filename stem if the
checkpoint doesn't carry a "variant" key.
--------------------------------------------------------------------------
 
Usage:
    python src/benchmark.py
    python src/benchmark.py --checkpoints-dir results/checkpoints --latency-runs 200

"""

import argparse
import json
import time
import warnings
from pathlib import Path
 
import torch
import torch.nn as nn
 
from dataset import get_dataloaders, NUM_CLASSES
from model import KeywordSpottingCNN, count_parameters
 
 
# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
 
def load_variant(path: Path):
    """Load a checkpoint of unknown format and return (model, meta_dict).
 
    `model` is always something callable as model(x) -> logits, in eval mode.
    `meta_dict` carries whatever bookkeeping info was available.
    """
    meta = {"variant": path.stem, "format": None, "path": str(path)}
 
    # Try torchscript first -- torch.jit.load fails fast on non-TS files.
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        meta["format"] = "torchscript"
        return model, meta
    except Exception:
        pass
 
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
 
    if isinstance(obj, nn.Module):
        obj.eval()
        meta["format"] = "full_model"
        return obj, meta
 
    if isinstance(obj, dict) and "model_state_dict" in obj:
        num_classes = obj.get("num_classes", NUM_CLASSES)
        in_channels = obj.get("in_channels", 1)
        model = KeywordSpottingCNN(num_classes=num_classes, in_channels=in_channels)
        model.load_state_dict(obj["model_state_dict"])
        model.eval()
        meta["format"] = "state_dict"
        for key in ("variant", "epoch", "val_loss", "val_acc", "args"):
            if key in obj:
                meta[key] = obj[key]
        return model, meta
 
    raise ValueError(
        f"Could not determine checkpoint format for {path}. "
        f"Expected a torchscript file, an nn.Module, or a dict with "
        f"'model_state_dict'. See the CHECKPOINT CONTRACT docstring at the "
        f"top of this file."
    )
 
 
# ---------------------------------------------------------------------------
# Size / parameter metrics
# ---------------------------------------------------------------------------
 
def file_size_bytes(path: Path) -> int:
    return path.stat().st_size
 
 
def count_nonzero_parameters(model) -> int:
    """Count non-zero weight/bias elements. Equal to total params for a dense
    (unpruned) model; meaningfully smaller after unstructured pruning, which
    zeros weights without shrinking tensor shapes. Skipped for torchscript
    models where parameter iteration isn't reliably available."""
    total_nonzero = 0
    try:
        for p in model.parameters():
            total_nonzero += int(torch.count_nonzero(p))
    except Exception:
        return None
    return total_nonzero
 
 
def total_parameters(model) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None
 
 
# ---------------------------------------------------------------------------
# MAC (multiply-accumulate) counting -- the number that actually matters for
# FPGA resource planning (roughly proportional to DSP slice / cycle budget).
# ---------------------------------------------------------------------------
 
def _conv_macs(module: nn.Conv2d, out_shape) -> int:
    _, out_c, out_h, out_w = out_shape
    in_c_per_group = module.in_channels // module.groups
    kh, kw = module.kernel_size
    return out_c * out_h * out_w * in_c_per_group * kh * kw
 
 
def _linear_macs(module: nn.Linear) -> int:
    return module.in_features * module.out_features
 
 
def compute_layer_profile(model: nn.Module, input_shape):
    """Run one forward pass with hooks to record, per Conv2d/Linear layer:
    module type, weight shape, output shape, MACs, and param count. Works off
    actual tensor shapes so it's correct even after structured pruning
    changes channel counts. Returns (layer_records, total_macs). Returns
    (None, None) if the model isn't hookable (e.g. some torchscript/quantized
    graphs) -- MAC estimation for quantized models is left as a follow-up
    once quantize.py defines the exact module types used.
    """
    records = []
    hooks = []
 
    def make_hook(name, module):
        def hook(mod, inp, out):
            record = {
                "layer": name,
                "type": type(mod).__name__,
                "output_shape": list(out.shape),
            }
            if isinstance(mod, nn.Conv2d):
                record["weight_shape"] = list(mod.weight.shape)
                record["macs"] = _conv_macs(mod, out.shape)
                record["params"] = mod.weight.numel() + (
                    mod.bias.numel() if mod.bias is not None else 0
                )
            elif isinstance(mod, nn.Linear):
                record["weight_shape"] = list(mod.weight.shape)
                record["macs"] = _linear_macs(mod)
                record["params"] = mod.weight.numel() + (
                    mod.bias.numel() if mod.bias is not None else 0
                )
            else:
                return
            records.append(record)
 
        return hook
 
    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_hook(name, module)))
 
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            model(dummy)
    except Exception as e:
        warnings.warn(f"Layer profiling failed ({e}); skipping MAC/layer report.")
        return None, None
    finally:
        for h in hooks:
            h.remove()
 
    total_macs = sum(r["macs"] for r in records)
    return records, total_macs
 
 
def weight_range_report(model: nn.Module):
    """Per-layer weight min/max/dtype -- useful for a teammate picking a
    fixed-point format (e.g. Q8.8) for an FPGA datapath, independent of
    whether we've quantized yet."""
    ranges = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.detach()
                ranges.append({
                    "layer": name,
                    "dtype": str(w.dtype),
                    "min": float(w.min()),
                    "max": float(w.max()),
                })
    except Exception:
        return None
    return ranges
 
 
# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
 
def measure_cpu_latency(model, input_shape, num_runs: int, warmup: int, num_threads: int):
    """Single-sample (batch=1), single-threaded CPU latency in milliseconds.
    Single-threaded because the deployment targets (MCU-class chip today,
    FPGA eventually) don't get to lean on an 8-core desktop scheduler --
    this keeps the number honest as a lower bound on real-world latency."""
    torch.set_num_threads(num_threads)
    dummy = torch.randn(*input_shape)
 
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
 
        timings = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            timings.append((time.perf_counter() - start) * 1000.0)  # ms
 
    timings = torch.tensor(timings)
    return float(timings.mean()), float(timings.std())
 
 
# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
 
def evaluate_accuracy(model, loader):
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples
 
 
# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
 
def format_size(num_bytes: int) -> str:
    if num_bytes is None:
        return "N/A"
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    return f"{kb / 1024:.2f} MB"
 
 
def print_summary_table(results):
    headers = ["Variant", "Size", "Params", "Sparsity", "MACs", "Latency (ms)", "Accuracy"]
    rows = []
    for r in results:
        sparsity = "N/A"
        if r["params_total"] and r["params_nonzero"] is not None:
            sparsity = f"{(1 - r['params_nonzero'] / r['params_total']) * 100:.1f}%"
        rows.append([
            r["variant"],
            format_size(r["size_bytes"]),
            f"{r['params_total']:,}" if r["params_total"] else "N/A",
            sparsity,
            f"{r['macs']:,}" if r["macs"] else "N/A",
            f"{r['latency_mean_ms']:.3f} ± {r['latency_std_ms']:.3f}" if r["latency_mean_ms"] else "N/A",
            f"{r['accuracy'] * 100:.2f}%" if r["accuracy"] is not None else "N/A",
        ])
 
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
 
 
def write_markdown_table(results, path: Path):
    lines = ["| Model Variant | Size | Accuracy | CPU Latency | Params | Sparsity | MACs |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        sparsity = "—"
        if r["params_total"] and r["params_nonzero"] is not None:
            sparsity = f"{(1 - r['params_nonzero'] / r['params_total']) * 100:.1f}%"
 
        acc_str = f"{r['accuracy'] * 100:.2f}%" if r["accuracy"] is not None else "—"
        latency_str = f"{r['latency_mean_ms']:.3f} ms" if r["latency_mean_ms"] else "—"
        params_str = f"{r['params_total']:,}" if r["params_total"] else "—"
        macs_str = f"{r['macs']:,}" if r["macs"] else "—"
 
        lines.append(
            f"| {r['variant']} | {format_size(r['size_bytes'])} | {acc_str} "
            f"| {latency_str} | {params_str} | {sparsity} | {macs_str} |"
        )
    path.write_text("\n".join(lines) + "\n")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark keyword-spotting model variants")
    parser.add_argument("--checkpoints-dir", type=str, default="results/checkpoints")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for the accuracy pass")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--input-shape", type=str, default="1,1,40,101",
        help="dummy input shape as batch,channels,mel_bins,time_frames; "
             "must match dataset.py's mel-spectrogram config",
    )
    parser.add_argument("--latency-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=1, help="CPU threads for latency measurement")
    parser.add_argument("--skip-accuracy", action="store_true", help="skip the (slower) test-set accuracy pass")
    parser.add_argument("--hardware-report", action="store_true",
                         help="also dump per-layer MAC/weight-range JSON per variant, for FPGA handoff")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
 
    checkpoints_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
 
    checkpoint_paths = sorted(
        p for p in checkpoints_dir.glob("*") if p.suffix in (".pt", ".pth")
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No .pt/.pth files found in {checkpoints_dir}. Run train.py / "
            f"prune.py / quantize.py first."
        )
 
    test_loader = None
    if not args.skip_accuracy:
        print("Loading test data...")
        _, _, test_loader = get_dataloaders(
            root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        # NOTE: this assumes get_dataloaders returns (train, val, test) in
        # that order, matching how train.py unpacks it. Flag this in review
        # of dataset.py in case the third loader is something else.
 
    all_results = []
    for path in checkpoint_paths:
        print(f"\n=== {path.name} ===")
        try:
            model, meta = load_variant(path)
        except Exception as e:
            print(f"  Skipping ({e})")
            continue
 
        variant = meta.get("variant", path.stem)
        size_bytes = file_size_bytes(path)
        params_total = total_parameters(model)
        params_nonzero = count_nonzero_parameters(model)
 
        macs = None
        if meta["format"] != "torchscript":
            _, macs = compute_layer_profile(model, input_shape)
 
        latency_mean, latency_std = measure_cpu_latency(
            model, input_shape, args.latency_runs, args.warmup_runs, args.num_threads
        )
 
        accuracy = None
        if test_loader is not None:
            try:
                _, accuracy = evaluate_accuracy(model, test_loader)
            except Exception as e:
                warnings.warn(f"Accuracy eval failed for {variant}: {e}")
 
        result = {
            "variant": variant,
            "format": meta["format"],
            "checkpoint": str(path),
            "size_bytes": size_bytes,
            "params_total": params_total,
            "params_nonzero": params_nonzero,
            "macs": macs,
            "latency_mean_ms": latency_mean,
            "latency_std_ms": latency_std,
            "accuracy": accuracy,
            "stored_val_loss": meta.get("val_loss"),
            "stored_val_acc": meta.get("val_acc"),
        }
        all_results.append(result)
 
        print(f"  format={meta['format']}  size={format_size(size_bytes)}  "
              f"params={params_total}  latency={latency_mean:.3f}±{latency_std:.3f} ms  "
              f"accuracy={accuracy}")
 
        if args.hardware_report and meta["format"] != "torchscript":
            layer_records, _ = compute_layer_profile(model, input_shape)
            ranges = weight_range_report(model)
            hw_report = {"variant": variant, "layers": layer_records, "weight_ranges": ranges}
            hw_path = results_dir / f"hw_report_{variant}.json"
            with open(hw_path, "w") as f:
                json.dump(hw_report, f, indent=2)
            print(f"  Hardware handoff report written to {hw_path}")
 
    print("\n" + "=" * 80)
    print_summary_table(all_results)
 
    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    write_markdown_table(all_results, results_dir / "benchmark_table.md")
 
    print(f"\nFull results: {results_dir / 'benchmark_results.json'}")
    print(f"Markdown table (drop straight into README): {results_dir / 'benchmark_table.md'}")
 
 
if __name__ == "__main__":
    main()