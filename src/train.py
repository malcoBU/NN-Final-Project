"""
train.py
--------
Training loop for AlphaSound (dual-head CNN: letter + language).

Usage
-----
    # From the project root:
    python src/train.py --data_dir data/processed

    # With options:
    python src/train.py \\
        --data_dir  data/processed \\
        --epochs    60             \\
        --batch_size 32            \\
        --lr        1e-3           \\
        --num_workers 4            \\
        --checkpoint_dir checkpoints

Device
------
Auto-detected: CUDA → MPS (Apple Silicon) → CPU.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure src/ is on the path when called from the project root
sys.path.insert(0, os.path.dirname(__file__))

from dataset import build_dataloaders, save_label_maps, VOWELS
from model   import AudioLetterClassifier, DualTaskLoss


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Classification accuracy from raw logits."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ── Single training epoch ─────────────────────────────────────────────────────

def train_one_epoch(
    model: AudioLetterClassifier,
    loader,
    criterion: DualTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Run one full training epoch.

    Returns
    -------
    dict with loss_total, loss_letter, loss_lang, acc_letter, acc_lang
    """
    model.train()

    total_loss   = 0.0
    total_l_loss = 0.0
    total_la_loss = 0.0
    total_l_acc  = 0.0
    total_la_acc = 0.0
    n_batches    = len(loader)

    for batch_idx, (mel, letter_targets, lang_targets) in enumerate(loader):
        mel            = mel.to(device)
        letter_targets = letter_targets.to(device)
        lang_targets   = lang_targets.to(device)

        optimizer.zero_grad()

        letter_logits, lang_logits = model(mel)

        loss, l_loss, la_loss = criterion(
            letter_logits, lang_logits,
            letter_targets, lang_targets,
        )

        loss.backward()

        # Gradient clipping: prevents gradient explosion
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metric accumulation
        total_loss    += loss.item()
        total_l_loss  += l_loss.item()
        total_la_loss += la_loss.item()
        total_l_acc   += accuracy(letter_logits, letter_targets)
        total_la_acc  += accuracy(lang_logits,   lang_targets)

        # Log every 10 batches (or at the end)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                f"loss={loss.item():.4f}  "
                f"acc_letter={accuracy(letter_logits, letter_targets):.3f}  "
                f"acc_lang={accuracy(lang_logits, lang_targets):.3f}",
                end="\r",
            )

    print()  # newline after \r

    return {
        "loss_total":  total_loss   / n_batches,
        "loss_letter": total_l_loss / n_batches,
        "loss_lang":   total_la_loss / n_batches,
        "acc_letter":  total_l_acc  / n_batches,
        "acc_lang":    total_la_acc / n_batches,
    }


# ── Validation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: AudioLetterClassifier,
    loader,
    criterion: DualTaskLoss,
    device: torch.device,
) -> dict:
    """
    Run the full validation set without gradients.

    Returns
    -------
    dict with loss_total, loss_letter, loss_lang, acc_letter, acc_lang
    """
    model.eval()

    total_loss    = 0.0
    total_l_loss  = 0.0
    total_la_loss = 0.0
    total_l_acc   = 0.0
    total_la_acc  = 0.0
    n_batches     = len(loader)

    for mel, letter_targets, lang_targets in loader:
        mel            = mel.to(device)
        letter_targets = letter_targets.to(device)
        lang_targets   = lang_targets.to(device)

        letter_logits, lang_logits = model(mel)

        loss, l_loss, la_loss = criterion(
            letter_logits, lang_logits,
            letter_targets, lang_targets,
        )

        total_loss    += loss.item()
        total_l_loss  += l_loss.item()
        total_la_loss += la_loss.item()
        total_l_acc   += accuracy(letter_logits, letter_targets)
        total_la_acc  += accuracy(lang_logits,   lang_targets)

    return {
        "loss_total":  total_loss    / n_batches,
        "loss_letter": total_l_loss  / n_batches,
        "loss_lang":   total_la_loss / n_batches,
        "acc_letter":  total_l_acc   / n_batches,
        "acc_lang":    total_la_acc  / n_batches,
    }


# ── Checkpoint saving ─────────────────────────────────────────────────────────

def save_checkpoint(
    model: AudioLetterClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """
    Save the full model state.

    The checkpoint includes:
      • model_state     → model weights
      • optimizer_state → optimizer state (useful for resuming)
      • epoch           → epoch at which it was saved
      • metrics         → validation metrics for that epoch

    To load the model in evaluate.py or for inference:
        ckpt = torch.load("checkpoints/best_model.pt")
        model.load_state_dict(ckpt["model_state"])
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch":           epoch,
        "metrics":         metrics,
    }, path)


# ── Main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = get_device()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n── Loading data ────────────────────────────────────────")

    # Letter subset: --vowels_only or --letters_subset a e i o u
    letters_subset = None
    if args.vowels_only:
        letters_subset = VOWELS
        print(f"  Mode: vowels only {VOWELS}")
    elif args.letters_subset:
        letters_subset = [l.lower() for l in args.letters_subset]
        print(f"  Mode: custom subset {letters_subset}")

    train_loader, val_loader, _ = build_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_prob=args.augment_prob,
        normalize=True,
        use_weighted_sampler=True,
        seed=args.seed,
        letters_subset=letters_subset,
    )

    # Save label maps for later inference
    save_label_maps("./data/label_maps.json")

    # n_letters is derived from the actual dataset so it always matches the subset
    n_letters_actual = train_loader.dataset.n_letters()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n── Initialising model (EfficientNet-B0, backbone frozen) ───")
    print(f"   Letter classes : {n_letters_actual}")
    model = AudioLetterClassifier(
        n_letters=n_letters_actual,
        n_langs=args.n_langs,
        dropout=args.dropout,
        freeze_backbone=True,   # backbone always frozen
    ).to(device)

    trainable, total = model.count_trainable_vs_total()
    print(f"Trainable parameters : {trainable:,} / {total:,}  "
          f"({100*trainable/total:.1f}%)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = DualTaskLoss(
        letter_weight=args.letter_weight,
        lang_weight=args.lang_weight,
        label_smoothing=args.label_smoothing,
    )

    # ── Optimizer and scheduler ───────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Checkpoint setup ──────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    last_ckpt_path = os.path.join(args.checkpoint_dir, "last_model.pt")
    history        = []

    print(f"\n── Training: {args.epochs} epochs ──────────────────────────")
    print(f"   LR            : {args.lr}")
    print(f"   Batch size    : {args.batch_size}")
    print(f"   Dropout       : {args.dropout}")
    print(f"   Label smooth  : {args.label_smoothing}")
    print(f"   Backbone      : FROZEN (no phase 2)\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs}  [{format_time(elapsed)}]  "
            f"LR={scheduler.get_last_lr()[0]:.2e}\n"
            f"  TRAIN → loss={train_metrics['loss_total']:.4f}  "
            f"acc_letter={train_metrics['acc_letter']:.4f}  "
            f"acc_lang={train_metrics['acc_lang']:.4f}\n"
            f"  VAL   → loss={val_metrics['loss_total']:.4f}  "
            f"acc_letter={val_metrics['acc_letter']:.4f}  "
            f"acc_lang={val_metrics['acc_lang']:.4f}"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            best_val_acc  = val_metrics["acc_letter"]
            save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        save_checkpoint(model, optimizer, epoch, val_metrics, last_ckpt_path)
        history.append({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })
        print()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("═" * 60)
    print(f"Training complete.")
    print(f"Best val_loss   : {best_val_loss:.4f}")
    print(f"Best acc_letter : {best_val_acc:.4f}")
    print(f"Checkpoint saved at: {best_ckpt_path}")
    print("═" * 60)

    # Save per-epoch metrics history to a simple CSV
    _save_history(history, args.checkpoint_dir)


def _save_history(history: list[dict], out_dir: str) -> None:
    """Save the per-epoch metrics history to a CSV file."""
    if not history:
        return
    path = os.path.join(out_dir, "training_history.csv")
    os.makedirs(out_dir, exist_ok=True)
    keys = list(history[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    print(f"History saved to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the AlphaSound classifier (letter + language)"
    )

    # Data
    p.add_argument("--data_dir",      default="data/processed",
                   help="Root directory containing the .npy files (english/ and spanish/)")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Where to save checkpoints")
    p.add_argument("--num_workers",   type=int, default=4,
                   help="DataLoader workers (set to 0 on Windows)")
    p.add_argument("--seed",          type=int, default=42)

    # Model
    p.add_argument("--n_letters",  type=int,   default=26,
                   help="Number of letter classes")
    p.add_argument("--n_langs",    type=int,   default=2,
                   help="Number of language classes")
    p.add_argument("--dropout",    type=float, default=0.1,
                   help="Dropout before the heads (kept low to avoid blocking learning)")

    # Training
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--augment_prob", type=float, default=0.0,
                   help="Online augmentation probability (0.0 if offline_augment.py was already run)")

    # Loss
    p.add_argument("--letter_weight",   type=float, default=0.7)
    p.add_argument("--lang_weight",     type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing (0.0 = no smoothing, clean gradient signal)")

    # Letter subset
    p.add_argument("--vowels_only",     action="store_true",
                   help="Train with the 5 vowels only (a,e,i,o,u). "
                        "Equivalent to --letters_subset a e i o u.")
    p.add_argument("--letters_subset",  nargs="+", default=None, metavar="LETTER",
                   help="List of letters to use (e.g. --letters_subset a b c d e). "
                        "If not specified, all letters are used.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
