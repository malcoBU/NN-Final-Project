"""
evaluate.py
-----------
Post-training evaluation on the held-out test set:
  • Overall accuracy and per-class F1-score
  • Confusion matrices (letter + language) saved as PNG
  • Classification report printed to stdout
  • Optional ONNX export for deployment

Usage
-----
    python src/evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/raw
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from dataset import build_dataloaders, IDX_TO_LETTER, IDX_TO_LANG, ALL_LETTERS
from model   import AudioLetterClassifier


# ── Inference pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: AudioLetterClassifier,
    loader,
    device: torch.device,
) -> dict:
    """
    Run the model over the entire loader and collect predictions + ground truth.

    Returns
    -------
    results : dict with keys
        letter_preds, letter_targets  → list[int]
        lang_preds,   lang_targets    → list[int]
    """
    model.eval()

    letter_preds,   letter_targets = [], []
    lang_preds,     lang_targets   = [], []

    for mel, l_true, la_true in loader:
        mel = mel.to(device)
        l_logits, la_logits = model(mel)

        letter_preds.extend(l_logits.argmax(-1).cpu().tolist())
        lang_preds.extend(la_logits.argmax(-1).cpu().tolist())
        letter_targets.extend(l_true.tolist())
        lang_targets.extend(la_true.tolist())

    return {
        "letter_preds":   letter_preds,
        "letter_targets": letter_targets,
        "lang_preds":     lang_preds,
        "lang_targets":   lang_targets,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_classification_report(
    targets: list[int],
    preds: list[int],
    label_map: dict[int, str],
    title: str,
) -> None:
    """Print sklearn's classification report with human-readable class names."""
    unique = sorted(set(targets))
    names  = [label_map[i] for i in unique]
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    print(classification_report(targets, preds, labels=unique, target_names=names))


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    targets: list[int],
    preds: list[int],
    label_map: dict[int, str],
    title: str,
    output_path: str,
    figsize: tuple = (18, 16),
    normalize: bool = True,
) -> None:
    """
    Plot and save a confusion matrix as a PNG file.

    Parameters
    ----------
    targets : list[int]
        Ground-truth class indices.
    preds : list[int]
        Predicted class indices.
    label_map : dict[int, str]
        Mapping from index to human-readable class name.
    title : str
        Figure title.
    output_path : str
        Where to save the PNG file.
    normalize : bool
        If True, each row is normalised to [0, 1] so cells show recall
        per class rather than raw counts. Easier to read when classes
        have different numbers of samples.
    """
    unique = sorted(set(targets) | set(preds))
    names  = [label_map[i] for i in unique]

    cm = confusion_matrix(targets, preds, labels=unique)
    if normalize:
        # Row-normalisation: cm[i, j] = fraction of class i predicted as j
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=vmax,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to: {output_path}")


# ── Per-class F1 bar chart ────────────────────────────────────────────────────

def plot_f1_per_class(
    targets: list[int],
    preds: list[int],
    label_map: dict[int, str],
    title: str,
    output_path: str,
) -> None:
    """
    Bar chart of F1-score per letter class, sorted ascending.
    Useful for quickly spotting which letters the model struggles with.
    """
    unique = sorted(set(targets))
    names  = [label_map[i] for i in unique]
    f1s    = f1_score(targets, preds, labels=unique, average=None, zero_division=0)

    order   = np.argsort(f1s)
    names_s = [names[i] for i in order]
    f1s_s   = f1s[order]

    colors = ["#e74c3c" if f < 0.5 else "#f39c12" if f < 0.8 else "#2ecc71"
              for f in f1s_s]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.4), 5))
    bars = ax.bar(names_s, f1s_s, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, label="F1=0.80")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Letter class")
    ax.set_ylabel("F1-score")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"F1 per-class chart saved to: {output_path}")


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(
    model: AudioLetterClassifier,
    output_path: str = "checkpoints/model.onnx",
    input_shape: tuple = (1, 1, 128, 128),
) -> None:
    """
    Export the model to ONNX format for deployment.

    ONNX (Open Neural Network Exchange) is a standard format supported by:
      • ONNX Runtime (CPU/GPU inference, ~3× faster than PyTorch on CPU)
      • TensorRT (NVIDIA GPU, production speed)
      • CoreML (Apple devices via coremltools)
      • TFLite (Android/iOS via ai-edge-torch)

    Parameters
    ----------
    model : AudioLetterClassifier
    output_path : str
        Path where the .onnx file will be saved.
    input_shape : tuple
        Shape of a single input tensor (batch, channel, height, width).
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["mel_spectrogram"],
        output_names=["letter_logits", "lang_logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size"},
            "letter_logits":   {0: "batch_size"},
            "lang_logits":     {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,  # optimise constant subgraphs
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"ONNX model exported to: {output_path}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    # Load model
    model = AudioLetterClassifier(n_letters=args.n_letters, n_langs=args.n_langs)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"Loaded model from '{args.checkpoint}'")
    print(f"Parameters: {model.count_parameters():,}")

    # Load test split
    _, _, test_loader = build_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_prob=0.0,
    )

    # Collect predictions
    results = collect_predictions(model, test_loader, device)

    # ── Overall accuracy ──────────────────────────────────────────────────────
    letter_acc = accuracy_score(results["letter_targets"], results["letter_preds"])
    lang_acc   = accuracy_score(results["lang_targets"],   results["lang_preds"])
    print(f"\nTest letter accuracy : {letter_acc:.4f}  ({letter_acc*100:.2f}%)")
    print(f"Test language accuracy: {lang_acc:.4f}  ({lang_acc*100:.2f}%)")

    # ── Classification reports ────────────────────────────────────────────────
    print_classification_report(
        results["letter_targets"], results["letter_preds"],
        IDX_TO_LETTER, "Letter classification report"
    )
    print_classification_report(
        results["lang_targets"], results["lang_preds"],
        IDX_TO_LANG, "Language classification report"
    )

    # ── Confusion matrices ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        results["letter_targets"], results["letter_preds"],
        IDX_TO_LETTER,
        title="Letter confusion matrix (normalised recall)",
        output_path=os.path.join(args.output_dir, "confusion_letter.png"),
    )
    plot_confusion_matrix(
        results["lang_targets"], results["lang_preds"],
        IDX_TO_LANG,
        title="Language confusion matrix",
        output_path=os.path.join(args.output_dir, "confusion_lang.png"),
        figsize=(5, 4),
    )

    # ── F1 per class ──────────────────────────────────────────────────────────
    plot_f1_per_class(
        results["letter_targets"], results["letter_preds"],
        IDX_TO_LETTER,
        title="F1-score per letter class",
        output_path=os.path.join(args.output_dir, "f1_per_letter.png"),
    )

    # ── ONNX export ───────────────────────────────────────────────────────────
    if args.export_onnx:
        onnx_path = os.path.join(args.checkpoint_dir, "model.onnx")
        export_onnx(model, output_path=onnx_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AlphaSound classifier")
    p.add_argument("--checkpoint",     required=True,             help="Path to .pt checkpoint")
    p.add_argument("--data_dir",       default="data/raw",        help="Dataset root")
    p.add_argument("--checkpoint_dir", default="checkpoints",     help="Used for ONNX export path")
    p.add_argument("--output_dir",     default="results",         help="Where to save plots")
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--n_letters",      type=int,   default=30)
    p.add_argument("--n_langs",        type=int,   default=2)
    p.add_argument("--export_onnx",    action="store_true",       help="Export model to ONNX")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())