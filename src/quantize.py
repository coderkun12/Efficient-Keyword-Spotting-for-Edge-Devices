"""
Quantization experiments: post-training static quantization (PTQ) and
quantization-aware training (QAT), both to INT8, using PyTorch's eager-mode
quantization API (matches the tech stack table in the README).

Usage:
    # PTQ: quantize the trained baseline, calibrate on a slice of training data
    python src/quantize.py --mode ptq --checkpoint results/checkpoints/baseline_best.pt

    # QAT: fine-tune with fake-quantization inserted, then convert to real INT8
    python src/quantize.py --mode qat --checkpoint results/checkpoints/baseline_best.pt --qat-epochs 5

    # Combined pipeline: quantize an already-pruned model (structured or
    # unstructured), satisfying the "prune + quantize together" project goal
    python src/quantize.py --mode qat --checkpoint results/checkpoints/pruned_structured_70-70-70-70.pt

Saves to results/checkpoints/<variant>.pt as a torchscript file (see the
checkpoint contract in benchmark.py) -- quantized eager-mode models have a
different module structure (quantized Conv2d/Linear, fake-quant/observer
modules) than the plain float model, so a plain state_dict load into
KeywordSpottingCNN won't work. torch.jit.trace + torch.jit.save packages the
model in a self-contained, directly loadable form.

WHY PTQ AND QAT ARE BOTH WORTH DOING:
PTQ is cheap (no training) but the model was never made aware, during
training, that its activations would get rounded to 8-bit integers -- so it
tends to lose more accuracy, especially at lower bit widths or on more
sensitive layers. QAT fine-tunes with fake-quantization ops inserted into
the forward pass, so the network's own weights adapt to quantization noise
before it's ever actually converted. Comparing PTQ vs. QAT accuracy recovery
is exactly the "which one recovers more" comparison the README goals ask for.

NOTE ON BACKEND: --backend controls which reference CPU kernels PyTorch uses
to produce and evaluate the INT8 model (fbgemm for x86, qnnpack for ARM /
Raspberry Pi). Neither is the actual FPGA datapath -- your teammate's FPGA
implementation will use its own fixed-point logic -- but the resulting INT8
weights, scales, and zero-points are the same numbers that would inform that
implementation, and --hardware-report style layer stats (see benchmark.py)
are meant to be read off of the fp32 model as a proxy for that reason.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders
from benchmark import load_variant, evaluate_accuracy, total_parameters, format_size


# Same block layout as prune.py / model.py: each tuple is the (conv, bn, relu)
# indices inside model.features for one block, used for Conv+BN+ReLU fusion.
FUSE_GROUPS = [
    ["0", "1", "2"],
    ["3", "4", "5"],
    ["7", "8", "9"],
    ["11", "12", "13"],
]


class QuantizableKeywordSpottingCNN(nn.Module):
    """Wraps a trained (float) KeywordSpottingCNN with Quant/DeQuant stubs,
    which is all eager-mode static quantization needs on top of an existing
    model. The wrapped model is otherwise architecturally identical --
    fuse_model() only fuses Conv+BN+ReLU into single ops in place."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = base_model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # fuse_modules wants module names relative to `self`, and needs the
        # model in eval mode (it reads BatchNorm's running stats to fold
        # them into the preceding conv).
        self.eval()
        torch.quantization.fuse_modules(
            self.model.features,
            FUSE_GROUPS,
            inplace=True,
        )

# Shared train/eval loop for QAT fine-tuning.

def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train(mode=train)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def calibrate(model, loader, num_batches: int):
    """Run representative data through the prepared (observer-inserted)
    model so the observers can record activation ranges for PTQ. No
    gradients, no labels needed beyond unpacking the batch."""
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(inputs)


# PTQ / QAT pipelines

def run_ptq(base_model, train_loader, backend: str, calibration_batches: int):
    wrapped = QuantizableKeywordSpottingCNN(base_model)
    wrapped.eval()
    wrapped.fuse_model()

    wrapped.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(wrapped, inplace=True)

    print(f"Calibrating on {calibration_batches} batches of training data...")
    calibrate(wrapped, train_loader, calibration_batches)

    torch.quantization.convert(wrapped, inplace=True)
    return wrapped


def run_qat(base_model, train_loader, val_loader, backend: str, epochs: int, lr: float):
    wrapped = QuantizableKeywordSpottingCNN(base_model)
    wrapped.fuse_model()  # done in eval mode internally, then...
    wrapped.train()

    wrapped.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(wrapped, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=lr)

    print(f"QAT fine-tuning for {epochs} epoch(s) (fake-quant active during train and eval)...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(wrapped, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(wrapped, val_loader, criterion, optimizer, train=False)
        print(f"  epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    wrapped.eval()
    torch.quantization.convert(wrapped, inplace=True)
    return wrapped

# Main

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize the keyword-spotting CNN (PTQ or QAT)")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/baseline_best.pt",
                         help="fp32 model to quantize -- baseline or an already-pruned "
                              "checkpoint (state_dict or full_model format, not torchscript)")
    parser.add_argument("--mode", type=str, choices=["ptq", "qat"], required=True)
    parser.add_argument("--backend", type=str, choices=["fbgemm", "qnnpack"], default="fbgemm",
                         help="fbgemm for x86 dev machines, qnnpack for ARM/Raspberry Pi")
    parser.add_argument("--calibration-batches", type=int, default=20, help="[ptq only]")
    parser.add_argument("--qat-epochs", type=int, default=5, help="[qat only]")
    parser.add_argument("--qat-lr", type=float, default=1e-4, help="[qat only]")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-shape", type=str, default="1,1,40,101",
                         help="dummy input shape for tracing: batch,channels,mel_bins,time_frames")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--variant", type=str, default=None, help="override output checkpoint name")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.quantized.engine = args.backend
    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.checkpoint)
    print(f"Loading {source_path} ...")
    base_model, meta = load_variant(source_path)
    if meta["format"] == "torchscript":
        raise ValueError(
            "Input checkpoint is already a quantized torchscript model -- "
            "quantize a float model, not another quantized one."
        )
    print(f"  loaded ({meta['format']}), params={total_parameters(base_model):,}")
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    # Default variant name always encodes the source checkpoint's stem, so
    # quantizing baseline vs. a pruned model never collides on the same
    # output filename even if you forget --variant. E.g. quantizing
    # "pruned_structured_70-70-70-70.pt" with --mode qat defaults to
    # "quantized_qat_fbgemm__from_pruned_structured_70-70-70-70".
    default_variant = f"quantized_{args.mode}_{args.backend}__from_{source_path.stem}"
    if args.mode == "ptq":
        variant = args.variant or default_variant
        quantized = run_ptq(base_model, train_loader, args.backend, args.calibration_batches)
    else:
        variant = args.variant or default_variant
        quantized = run_qat(base_model, train_loader, val_loader, args.backend, args.qat_epochs, args.qat_lr)
    print("Evaluating final INT8 model on the test set...")
    test_loss, test_acc = evaluate_accuracy(quantized, test_loader)
    print(f"  test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    example_input = torch.randn(*input_shape)
    traced = torch.jit.trace(quantized, example_input)
    out_path = checkpoints_dir / f"{variant}.pt"
    torch.jit.save(traced, str(out_path))
    # Torchscript files can't carry an arbitrary metadata dict the way state_dict checkpoints can (see the CHECKPOINT CONTRACT in
    # benchmark.py), so provenance -- which exact checkpoint was quantized, with what settings, and what it scored -- gets written to a small
    # sidecar JSON next to it instead. Otherwise this information is only ever visible in stdout at the moment you run this script, and is lost
    # the second your terminal scrollback clears.
    sidecar = {
        "variant": variant,
        "source_checkpoint": str(source_path),
        "mode": args.mode,
        "backend": args.backend,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "args": vars(args),
    }
    sidecar_path = checkpoints_dir / f"{variant}.meta.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nSaved: {out_path} ({format_size(out_path.stat().st_size)})")
    print(f"Provenance sidecar: {sidecar_path}")
    print("Run benchmark.py to compare size/latency/accuracy against other variants: "
          "python src/benchmark.py")


if __name__ == "__main__":
    main()