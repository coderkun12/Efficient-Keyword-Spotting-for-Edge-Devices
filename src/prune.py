# Pruning experiments (structured and unstructured)

import argparse
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from dataset import get_dataloaders, NUM_CLASSES
from model import KeywordSpottingCNN
from benchmark import load_variant, count_nonzero_parameters, total_parameters, format_size

# Conv/BN index pairs:
CONV_BN_PAIRS=[(0,1),(3,4),(7,8),(11,12)]

def run_epoch(model,loader,criterion,optimizer,train:bool):
    model.train(mode=train)
    total_loss,total_correct,total_samples=0.0,0,0
    context=torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            if train:
                loss.backward()
                optimizer.step()
            batch_size=labels.size(0)
            total_loss+=loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples

def finetune(model, train_loader, val_loader, epochs, lr, weight_decay):
    if epochs <= 0:
        val_loss, val_acc = run_epoch(model, val_loader, nn.CrossEntropyLoss(), None, train=False)
        return copy.deepcopy(model.state_dict()), val_loss, val_acc
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    best_state, best_val_loss, best_val_acc = None, float("inf"), 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, train=False)
        print(f"  finetune epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc = val_loss, val_acc
            best_state = copy.deepcopy(model.state_dict())
    return best_state, best_val_loss, best_val_acc 

# Unstructured Pruning
def apply_unstructured_pruning(model: nn.Module, amount: float):
    parameters_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount,
    )
    return parameters_to_prune

def remove_pruning_reparametrization(parameters_to_prune):
    for module, name in parameters_to_prune:
        prune.remove(module, name)

# Structured (channel) pruning
def _channel_importance(conv: nn.Conv2d)->torch.Tensor:
    return conv.weight.detach().abs().sum(dim=(1,2,3))

def _keep_indices(importance: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    num_keep = max(1, round(len(importance) * keep_ratio))
    keep = torch.topk(importance, num_keep).indices
    return torch.sort(keep).values

def _slim_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, keep_out, keep_in=None):
    in_channels = conv.in_channels if keep_in is None else len(keep_in)
    new_conv = nn.Conv2d(
        in_channels, len(keep_out), kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
        bias=conv.bias is not None,
    )
    weight = conv.weight.detach()[keep_out]
    if keep_in is not None:
        weight = weight[:, keep_in]
    new_conv.weight.data.copy_(weight)
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.detach()[keep_out])
 
    new_bn = nn.BatchNorm2d(
        len(keep_out), eps=bn.eps, momentum=bn.momentum,
        affine=bn.affine, track_running_stats=bn.track_running_stats,
    )
    if bn.affine:
        new_bn.weight.data.copy_(bn.weight.detach()[keep_out])
        new_bn.bias.data.copy_(bn.bias.detach()[keep_out])
    if bn.track_running_stats:
        new_bn.running_mean.copy_(bn.running_mean[keep_out])
        new_bn.running_var.copy_(bn.running_var[keep_out])
 
    return new_conv, new_bn
 
 
def apply_structured_pruning(model: KeywordSpottingCNN, ratios):
    assert len(ratios) == len(CONV_BN_PAIRS), (
        f"Expected {len(CONV_BN_PAIRS)} ratios (one per conv block), got {len(ratios)}"
    )
 
    keep_in = None  # block 1's conv has fixed in_channels (raw spectrogram channel)
    for (conv_idx, bn_idx), keep_ratio in zip(CONV_BN_PAIRS, ratios):
        conv = model.features[conv_idx]
        bn = model.features[bn_idx]
        importance = _channel_importance(conv)
        keep_out = _keep_indices(importance, keep_ratio)
        new_conv, new_bn = _slim_conv_bn(conv, bn, keep_out, keep_in)
        model.features[conv_idx] = new_conv
        model.features[bn_idx] = new_bn
        keep_in = keep_out
 
    # Final conv block's kept output channels are what feeds the classifier
    # after global average pooling -- slim the classifier's input dimension
    # to match, inheriting the corresponding weight columns.
    old_classifier = model.classifier
    new_classifier = nn.Linear(len(keep_in), old_classifier.out_features)
    new_classifier.weight.data.copy_(old_classifier.weight.detach()[:, keep_in])
    new_classifier.bias.data.copy_(old_classifier.bias.detach())
    model.classifier = new_classifier
 
    return model
 
# Main

def parse_args():
    parser = argparse.ArgumentParser(description="Prune the keyword-spotting CNN")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/baseline_best.pt",
                         help="model to prune -- baseline by default, but can be any "
                              "checkpoint benchmark.py's load_variant understands "
                              "(e.g. an already-pruned model, to chain compression steps)")
    parser.add_argument("--mode", type=str, choices=["unstructured", "structured"], required=True)
    parser.add_argument("--amount", type=float, default=0.5,
                         help="[unstructured] global fraction of weights to zero out")
    parser.add_argument("--structured-ratios", type=str, default="0.3,0.3,0.3,0.3",
                         help="[structured] comma-separated KEEP ratio per conv block "
                              "(4 values, one per block, e.g. '1.0,0.7,0.7,0.7' to leave "
                              "block 1 untouched and remove 30% of channels elsewhere)")
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--variant", type=str, default=None, help="override output checkpoint name")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
 
    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Loading {args.checkpoint} ...")
    model, meta = load_variant(Path(args.checkpoint))
    if meta["format"] == "torchscript":
        raise ValueError(
            "Cannot prune a torchscript/quantized checkpoint with this script. "
            "Prune before quantizing, not after."
        )
    print(f"  loaded ({meta['format']}), params={total_parameters(model):,}")
 
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )
 
    parameters_to_prune = None  # only set for unstructured mode
 
    if args.mode == "unstructured":
        variant = args.variant or f"pruned_unstructured_{int(args.amount * 100)}"
        print(f"Applying global unstructured pruning (amount={args.amount}) ...")
        parameters_to_prune = apply_unstructured_pruning(model, args.amount)
    else:
        ratios = [float(x) for x in args.structured_ratios.split(",")]
        variant = args.variant or "pruned_structured_" + "-".join(
            str(int(r * 100)) for r in ratios
        )
        print(f"Applying structured (channel) pruning, keep-ratios={ratios} ...")
        model = apply_structured_pruning(model, ratios)
        print(f"  params after channel removal: {total_parameters(model):,}")
 
    print(f"Fine-tuning for {args.finetune_epochs} epoch(s) to recover accuracy ...")
    best_state, val_loss, val_acc = finetune(
        model, train_loader, val_loader,
        args.finetune_epochs, args.finetune_lr, args.finetune_weight_decay,
    )
    model.load_state_dict(best_state)
 
    if args.mode == "unstructured":
        remove_pruning_reparametrization(parameters_to_prune)
        params_total = total_parameters(model)
        params_nonzero = count_nonzero_parameters(model)
        sparsity = 1 - params_nonzero / params_total
        print(f"Achieved sparsity: {sparsity * 100:.1f}% "
              f"({params_nonzero:,}/{params_total:,} nonzero)")

        out_path = checkpoints_dir / f"{variant}.pt"
        torch.save({
            "variant": variant,
            "format": "state_dict",
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "args": vars(args),
        }, out_path)
    else:
        params_total = total_parameters(model)
        print(f"Final structured-pruned param count: {params_total:,}")
        out_path = checkpoints_dir / f"{variant}.pt"
        torch.save(model, out_path)

    print(f"\nSaved: {out_path} ({format_size(out_path.stat().st_size)})")
    print(f"Final val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    print("Run benchmark.py to get size/latency/accuracy on the test set: "
          "python src/benchmark.py")
 
if __name__ == "__main__":
    main()