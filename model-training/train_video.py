"""
Training loop for the CNN+LSTM video-sequence deepfake classifier.

IMPORTANT — memory note for 4GB VRAM GPUs (e.g. GTX 1650 Ti):
Video sequences are far more memory-hungry than single images, since each
batch item is [seq_len, C, H, W] instead of just [C, H, W]. batch_size=2 with
seq_len=8 is already roughly equivalent to batch_size=16 for a single-image
model. Start SMALL and only increase if you don't hit CUDA out-of-memory.

Usage:
    python3 train_video.py --data_dir frames --epochs 10 --batch_size 2 --seq_len 8
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from video_dataset import VideoSequenceDataset
from model_video import CNNLSTMClassifier


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * sequences.size(0)
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
                         help="Path to frames/ directory produced by extract_frames.py")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2,
                         help="Kept very small — video sequences use much more VRAM than images")
    parser.add_argument("--accum_steps", type=int, default=8,
                         help="Effective batch size = batch_size * accum_steps (default 2*8=16)")
    parser.add_argument("--seq_len", type=int, default=8,
                         help="Number of frames per sequence (must be <= frames_per_video used in extraction)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_video")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected — video training on CPU will be extremely slow.")

    print("Loading video sequence datasets (this can take a moment)...")
    train_ds = VideoSequenceDataset(args.data_dir, split="train", seq_len=args.seq_len)
    valid_ds = VideoSequenceDataset(args.data_dir, split="valid", seq_len=args.seq_len)
    print(f"Train videos: {len(train_ds)} | Valid videos: {len(valid_ds)}")
    print(f"Batch size: {args.batch_size} | Accum steps: {args.accum_steps} "
          f"| Effective batch: {args.batch_size * args.accum_steps} | Seq len: {args.seq_len}")

    # Celeb-DF-v2 is heavily class-imbalanced (far more fake than real videos).
    # Compute inverse-frequency class weights so the loss doesn't just learn
    # to always predict the majority class.
    train_labels = [label for _, label in train_ds.samples]
    num_real = train_labels.count(0)
    num_fake = train_labels.count(1)
    print(f"Class balance in train split — real: {num_real}, fake: {num_fake}")
    total = num_real + num_fake
    weight_real = total / (2 * max(num_real, 1))
    weight_fake = total / (2 * max(num_fake, 1))
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(device)
    print(f"Applying class weights — real: {weight_real:.3f}, fake: {weight_fake:.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    model = CNNLSTMClassifier(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)
            (loss / args.accum_steps).backward()

            if (i + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * sequences.size(0)

            if i % 20 == 0:
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

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"  New best model saved (val_acc={val_acc:.4f})")

        torch.cuda.empty_cache()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("CNN+LSTM Training and Validation Loss")
    ax1.legend()

    ax2.plot(history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("CNN+LSTM Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curve_video.png", dpi=150)
    print("\nTraining curve saved to training_curve_video.png")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
