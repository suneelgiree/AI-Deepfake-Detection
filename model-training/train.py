"""
Training loop for the deepfake visual-branch classifier.
Adjusted for 4GB VRAM (e.g. GTX 1650 Ti): small batch size + gradient
accumulation to simulate a larger effective batch size without OOM crashes.

Usage:
    python3 train.py --data_dir dataset/real_vs_fake/real-vs-fake --epochs 10
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from dataset import DeepfakeFaceDataset
from model import EfficientNetClassifier


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return avg_loss, acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset root containing train/valid/test folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Physical batch size — kept small for 4GB VRAM GPUs")
    parser.add_argument("--accum_steps", type=int, default=4,
                         help="Gradient accumulation steps. Effective batch size = "
                              "batch_size * accum_steps (default 8*4=32)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected — training will be very slow. "
              "Check your CUDA/PyTorch install (see README.md).")

    print("Loading datasets...")
    train_ds = DeepfakeFaceDataset(args.data_dir, split="train")
    valid_ds = DeepfakeFaceDataset(args.data_dir, split="valid")
    print(f"Train samples: {len(train_ds)} | Valid samples: {len(valid_ds)}")
    print(f"Physical batch size: {args.batch_size} | Accumulation steps: {args.accum_steps} "
          f"| Effective batch size: {args.batch_size * args.accum_steps}")

    # num_workers=2 is plenty given 16GB system RAM; higher won't help since
    # the GPU (not the CPU data loading) is the bottleneck on this hardware
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    model = EfficientNetClassifier(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.time()
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            # Normalize loss by accumulation steps so gradients average correctly
            (loss / args.accum_steps).backward()

            if (i + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * images.size(0)

            if i % 50 == 0:
                print(f"  Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, valid_loader, device, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} done in {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save checkpoint every epoch 
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  New best model saved (val_acc={val_acc:.4f})")

        # Free up VRAM between epochs — helps on tight-memory GPUs
        torch.cuda.empty_cache()

    # Plot and save training curve — a real artifact for report/slides
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("\nTraining curve saved to training_curve.png")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
