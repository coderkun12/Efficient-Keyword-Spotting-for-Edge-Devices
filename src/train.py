"""
Baseline training loop.

Usage:
    python src/train.py
    python src/train.py --epochs 60 --batch-size 128 --lr 1e-3

Saves:
    results/checkpoints/baseline_best.pt   -- best model (by val LOSS, not accuracy --
                                               val accuracy is noisier on a small val set)
    results/baseline_history.json          -- per-epoch train/val loss & accuracy
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders, NUM_CLASSES
from model import KeywordSpottingCNN, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline keyword-spotting CNN")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=10, help="epochs between LR decay steps")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="LR multiplier at each decay step")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience (epochs)")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train(mode=train)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Using device: {device} | AMP: {use_amp}")

    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data (this will download Speech Commands on first run)...")
    train_loader, val_loader, _ = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = KeywordSpottingCNN(num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Fixed step-decay schedule (matches the reference KWS training recipe:
    # halve LR every N epochs) rather than ReduceLROnPlateau, which reacts
    # to val_acc -- a metric that's noisy on a validation set this size and
    # can trigger premature or inconsistent LR drops.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    scaler = torch.amp.GradScaler('amp') if use_amp else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc_at_best_loss = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc_at_best_loss = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                checkpoints_dir / "baseline_best.pt",
            )
            print(f"  -> New best model saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    with open(results_dir / "baseline_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation loss: {best_val_loss:.4f} (val_acc at that point: {best_val_acc_at_best_loss:.4f})")
    print(f"Checkpoint saved to: {checkpoints_dir / 'baseline_best.pt'}")
    print(f"History saved to: {results_dir / 'baseline_history.json'}")


if __name__ == "__main__":
    main()