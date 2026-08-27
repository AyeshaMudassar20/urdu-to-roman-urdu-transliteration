"""
Training script for the Urdu -> Roman Urdu Seq2Seq LSTM model.

Reconstructed to match the architecture verified in the original trained
checkpoint (see model.py for details). Trains with teacher forcing and saves
the best model (by validation loss) to best_seq2seq_model.pkl.

Usage:
    python train.py --epochs 20 --batch-size 64 --lr 0.001
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PAD_IDX, TransliterationDataset, collate_batch, load_tokenizers
from model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0.0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        # Flatten for CrossEntropyLoss, skipping the <s> token at position 0.
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg, teacher_forcing_ratio=0.0)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_flat)
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train the Urdu -> Roman Urdu Seq2Seq model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="best_seq2seq_model.pkl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ur_sp, en_sp = load_tokenizers()

    train_ds = TransliterationDataset("train", ur_sp, en_sp)
    val_ds = TransliterationDataset("val", ur_sp, en_sp)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.clip)
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model to {args.out} (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
