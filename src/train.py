"""
train.py
--------
Bucle de entrenamiento para AlphaSound (CNN dual-head: letra + idioma).

Uso
---
    # Desde la raíz del proyecto:
    python src/train.py --data_dir data/processed

    # Con opciones:
    python src/train.py \\
        --data_dir  data/processed \\
        --epochs    60             \\
        --batch_size 32            \\
        --lr        1e-3           \\
        --num_workers 4            \\
        --checkpoint_dir checkpoints

Dispositivo
-----------
Se detecta automáticamente: CUDA → MPS (Apple Silicon) → CPU.
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

# Asegurar que src/ está en el path cuando se llama desde la raíz
sys.path.insert(0, os.path.dirname(__file__))

from dataset import build_dataloaders, save_label_maps
from model   import AudioLetterClassifier, DualTaskLoss


# ── Utilidades ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Dispositivo: {device}")
    return device


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy de clasificación a partir de logits crudos."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ── Un paso de entrenamiento ──────────────────────────────────────────────────

def train_one_epoch(
    model: AudioLetterClassifier,
    loader,
    criterion: DualTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Pasa una época completa de entrenamiento.

    Devuelve
    --------
    dict con loss_total, loss_letter, loss_lang, acc_letter, acc_lang
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

        # Gradient clipping: evita explosión de gradientes
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Acumulación de métricas
        total_loss    += loss.item()
        total_l_loss  += l_loss.item()
        total_la_loss += la_loss.item()
        total_l_acc   += accuracy(letter_logits, letter_targets)
        total_la_acc  += accuracy(lang_logits,   lang_targets)

        # Log cada 10 batches (o al final)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                f"loss={loss.item():.4f}  "
                f"acc_letter={accuracy(letter_logits, letter_targets):.3f}  "
                f"acc_lang={accuracy(lang_logits, lang_targets):.3f}",
                end="\r",
            )

    print()  # salto de línea tras el \r

    return {
        "loss_total":  total_loss   / n_batches,
        "loss_letter": total_l_loss / n_batches,
        "loss_lang":   total_la_loss / n_batches,
        "acc_letter":  total_l_acc  / n_batches,
        "acc_lang":    total_la_acc / n_batches,
    }


# ── Un paso de validación ─────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: AudioLetterClassifier,
    loader,
    criterion: DualTaskLoss,
    device: torch.device,
) -> dict:
    """
    Pasa el conjunto de validación completo sin gradientes.

    Devuelve
    --------
    dict con loss_total, loss_letter, loss_lang, acc_letter, acc_lang
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


# ── Guardado de checkpoint ────────────────────────────────────────────────────

def save_checkpoint(
    model: AudioLetterClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """
    Guarda el estado completo del modelo.

    El checkpoint incluye:
      • model_state     → pesos del modelo
      • optimizer_state → estado del optimizador (útil para reanudar)
      • epoch           → época en la que se guardó
      • metrics         → métricas de validación de esa época

    Para cargar el modelo en evaluate.py o inferencia:
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


# ── Bucle principal ───────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = get_device()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n── Cargando datos ──────────────────────────────────────")
    train_loader, val_loader, _ = build_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_prob=args.augment_prob,
        normalize=True,
        use_weighted_sampler=True,
        seed=args.seed,
    )

    # Guardar label maps para inferencia posterior
    save_label_maps(os.path.join(os.path.dirname(args.data_dir), "data", "label_maps.json"))

    # ── Modelo ────────────────────────────────────────────────────────────────
    print("\n── Inicializando modelo ─────────────────────────────────")
    model = AudioLetterClassifier(
        n_letters=args.n_letters,
        n_langs=args.n_langs,
        dropout=args.dropout,
    ).to(device)
    print(f"Parámetros entrenables: {model.count_parameters():,}")

    # ── Pérdida, optimizador y scheduler ─────────────────────────────────────
    criterion = DualTaskLoss(
        letter_weight=args.letter_weight,
        lang_weight=args.lang_weight,
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # CosineAnnealingLR: reduce el LR suavemente hasta eta_min a lo largo de
    # todas las épocas. Favorece convergencia final más estable que StepLR.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,  # LR mínimo = 1% del inicial
    )

    # ── Preparar checkpoint ───────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    last_ckpt_path = os.path.join(args.checkpoint_dir, "last_model.pt")

    history = []  # lista de dicts con métricas por época

    print(f"\n── Entrenamiento: {args.epochs} épocas ──────────────────────")
    print(f"   LR inicial   : {args.lr}")
    print(f"   Batch size   : {args.batch_size}")
    print(f"   Augment prob : {args.augment_prob}")
    print(f"   Checkpoints  : {args.checkpoint_dir}/\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Entrenamiento ─────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # ── Validación ────────────────────────────────────────────────────────
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0

        # ── Log de época ──────────────────────────────────────────────────────
        print(
            f"Época {epoch:03d}/{args.epochs}  [{format_time(elapsed)}]  "
            f"LR={scheduler.get_last_lr()[0]:.2e}\n"
            f"  TRAIN → loss={train_metrics['loss_total']:.4f}  "
            f"acc_letter={train_metrics['acc_letter']:.4f}  "
            f"acc_lang={train_metrics['acc_lang']:.4f}\n"
            f"  VAL   → loss={val_metrics['loss_total']:.4f}  "
            f"acc_letter={val_metrics['acc_letter']:.4f}  "
            f"acc_lang={val_metrics['acc_lang']:.4f}"
        )

        # ── Guardar mejor modelo ──────────────────────────────────────────────
        is_best = val_metrics["loss_total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss_total"]
            best_val_acc  = val_metrics["acc_letter"]
            save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt_path)
            print(f"  ✓ Nuevo mejor modelo guardado (val_loss={best_val_loss:.4f})")

        # Checkpoint de la última época (útil para reanudar)
        save_checkpoint(model, optimizer, epoch, val_metrics, last_ckpt_path)

        # Historial
        history.append({
            "epoch": epoch,
            "lr":    scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })

        print()

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("═" * 60)
    print(f"Entrenamiento completado.")
    print(f"Mejor val_loss   : {best_val_loss:.4f}")
    print(f"Mejor acc_letter : {best_val_acc:.4f}")
    print(f"Checkpoint guardado en: {best_ckpt_path}")
    print("═" * 60)

    # Guardar historial en CSV sencillo
    _save_history(history, args.checkpoint_dir)


def _save_history(history: list[dict], out_dir: str) -> None:
    """Guarda el historial de métricas por época en un CSV."""
    if not history:
        return
    path = os.path.join(out_dir, "training_history.csv")
    os.makedirs(out_dir, exist_ok=True)
    keys = list(history[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    print(f"Historial guardado en: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena el clasificador AlphaSound (letra + idioma)"
    )

    # Datos
    p.add_argument("--data_dir",      default="data/processed",
                   help="Directorio raíz con los .npy (contiene english/ y spanish/)")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Dónde guardar los checkpoints")
    p.add_argument("--num_workers",   type=int, default=4,
                   help="Workers para DataLoader (pon 0 en Windows)")
    p.add_argument("--seed",          type=int, default=42)

    # Modelo
    p.add_argument("--n_letters",  type=int,   default=26,
                   help="Número de clases de letras")
    p.add_argument("--n_langs",    type=int,   default=2,
                   help="Número de clases de idioma")
    p.add_argument("--dropout",    type=float, default=0.5,
                   help="Dropout antes de las cabezas de clasificación")

    # Entrenamiento
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3,
                   help="Learning rate inicial para AdamW")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Weight decay (L2) de AdamW")
    p.add_argument("--augment_prob", type=float, default=0.8,
                   help="Probabilidad de aplicar augmentación en training")

    # Pérdida
    p.add_argument("--letter_weight",   type=float, default=0.7,
                   help="Peso de la pérdida de letra en la loss total")
    p.add_argument("--lang_weight",     type=float, default=0.3,
                   help="Peso de la pérdida de idioma en la loss total")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing para CrossEntropyLoss")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
