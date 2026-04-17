"""
dataset.py
----------
PyTorch Dataset and DataLoader factory for the AlphaSound project.

Expected directory layout (under data/raw/ or data/processed/)
--------------------------------------------------------------
root/
  english/
    a/  speaker1_a_01.wav  speaker2_a_01.wav  ...
    b/  ...
    ...
  spanish/
    a/  ...
    n/  ...   (includes ñ, ch, ll, rr sub-folders)
    ...

Each .wav file is identified by:
  • letter label  → sub-folder name  (e.g. "b", "ll")
  • language label → parent folder name ("english" / "spanish")
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocess import load_and_preprocess
from augment    import augment
from features   import extract_features, normalize_spectrogram

# ── Label definitions ─────────────────────────────────────────────────────────

LANGUAGE_TO_IDX: dict[str, int] = {
    "english": 0,
    "spanish": 1,
}

# All letters supported — English 26 + Spanish-only: ñ, ch, ll, rr
LETTERS_ENGLISH = list("abcdefghijklmnopqrstuvwxyz")
LETTERS_SPANISH_EXTRA = ["ñ", "ch", "ll", "rr"]
ALL_LETTERS = LETTERS_ENGLISH + LETTERS_SPANISH_EXTRA  # 30 total

LETTER_TO_IDX: dict[str, int] = {l: i for i, l in enumerate(ALL_LETTERS)}
IDX_TO_LETTER: dict[int, str] = {i: l for l, i in LETTER_TO_IDX.items()}
IDX_TO_LANG:   dict[int, str] = {v: k for k, v in LANGUAGE_TO_IDX.items()}


# ── Dataset ───────────────────────────────────────────────────────────────────

class AlphaSoundDataset(Dataset):
    """
    Loads raw .wav files on the fly, preprocesses them, applies optional
    augmentation, extracts Mel-spectrogram features and returns a tuple
    (tensor, letter_label, lang_label).

    Parameters
    ----------
    root_dir : str
        Path to the dataset root (contains 'english/' and 'spanish/' folders).
    split : str
        One of 'train', 'val', 'test'.
    split_ratios : tuple[float, float, float]
        Fractions for train/val/test splitting. Must sum to 1.0.
    augment_prob : float
        Probability of applying augmentation to each training sample.
        Ignored for 'val' and 'test' splits.
    normalize : bool
        If True, standardise each spectrogram to zero mean / unit variance.
    seed : int
        Random seed for reproducible splits.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        split_ratios: tuple = (0.70, 0.15, 0.15),
        augment_prob: float = 0.8,
        normalize: bool = True,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, \
            "split_ratios must sum to 1.0"

        self.root_dir     = Path(root_dir)
        self.split        = split
        self.augment_prob = augment_prob if split == "train" else 0.0
        self.normalize    = normalize

        # Collect all samples → list of dicts {path, letter_idx, lang_idx}
        all_samples = self._scan_directory()

        if not all_samples:
            raise RuntimeError(
                f"No .wav files found under '{root_dir}'. "
                "Check the folder structure: root/language/letter/*.wav"
            )

        # Reproducible shuffle + split
        random.seed(seed)
        random.shuffle(all_samples)
        n = len(all_samples)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])

        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train : n_train + n_val]
        else:  # test
            self.samples = all_samples[n_train + n_val :]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _scan_directory(self) -> list[dict]:
        """
        Walk the dataset root and collect every .wav file with its labels.

        Returns
        -------
        samples : list of {"path": str, "letter_idx": int, "lang_idx": int}
        """
        samples = []
        for lang_dir in sorted(self.root_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang_name = lang_dir.name.lower()
            if lang_name not in LANGUAGE_TO_IDX:
                continue  # skip unexpected folders
            lang_idx = LANGUAGE_TO_IDX[lang_name]

            for letter_dir in sorted(lang_dir.iterdir()):
                if not letter_dir.is_dir():
                    continue
                letter_name = letter_dir.name.lower()
                if letter_name not in LETTER_TO_IDX:
                    continue  # skip unknown letter folders
                letter_idx = LETTER_TO_IDX[letter_name]

                for wav_file in letter_dir.glob("*.wav"):
                    samples.append({
                        "path":       str(wav_file),
                        "letter_idx": letter_idx,
                        "lang_idx":   lang_idx,
                    })
        return samples

    # ── PyTorch interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """
        Returns
        -------
        mel_tensor : torch.Tensor, shape (1, 128, 128), float32
        letter_idx : int
        lang_idx   : int
        """
        sample = self.samples[idx]

        # Load + preprocess waveform
        y = load_and_preprocess(sample["path"])

        # Augment (only when augment_prob > 0, i.e. training split)
        if self.augment_prob > 0:
            y = augment(y, p_apply=self.augment_prob)

        # Extract Mel-spectrogram → numpy (1, 128, 128)
        mel = extract_features(y)

        # Optional per-instance normalisation
        if self.normalize:
            mel = normalize_spectrogram(mel)

        return (
            torch.from_numpy(mel),
            sample["letter_idx"],
            sample["lang_idx"],
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency weights for all letter classes present
        in this split. Pass to WeightedRandomSampler to handle class imbalance.

        Returns
        -------
        weights : torch.Tensor, shape (len(self),)
            Per-sample weight for use with WeightedRandomSampler.
        """
        from collections import Counter

        counts = Counter(s["letter_idx"] for s in self.samples)
        n_classes = len(LETTER_TO_IDX)

        # Weight = 1 / frequency; classes absent in this split get weight 0
        freq = torch.zeros(n_classes)
        for idx, cnt in counts.items():
            freq[idx] = cnt

        class_w = 1.0 / (freq + 1e-8)
        class_w[freq == 0] = 0.0  # absent classes

        # Map per-sample
        sample_weights = torch.tensor(
            [class_w[s["letter_idx"]].item() for s in self.samples]
        )
        return sample_weights

    def summary(self) -> str:
        """Print a short summary of this split."""
        from collections import Counter
        letter_counts = Counter(
            IDX_TO_LETTER[s["letter_idx"]] for s in self.samples
        )
        lang_counts = Counter(
            IDX_TO_LANG[s["lang_idx"]] for s in self.samples
        )
        lines = [
            f"Split : {self.split}",
            f"Total : {len(self.samples)} samples",
            f"Langs : {dict(lang_counts)}",
            f"Letters (top-5): {letter_counts.most_common(5)}",
        ]
        return "\n".join(lines)


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_prob: float = 0.8,
    normalize: bool = True,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders in one call.

    Parameters
    ----------
    root_dir : str
        Root directory of the dataset (contains english/ and spanish/).
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Worker processes for parallel data loading. Set to 0 on Windows.
    augment_prob : float
        Probability of augmenting each training sample.
    normalize : bool
        Per-instance spectrogram standardisation.
    use_weighted_sampler : bool
        If True, use WeightedRandomSampler on the train set to handle
        class imbalance (recommended when some letters are rare).
    seed : int
        Random seed for dataset splitting.

    Returns
    -------
    (train_loader, val_loader, test_loader) : tuple of DataLoader
    """
    train_ds = AlphaSoundDataset(root_dir, "train", augment_prob=augment_prob,
                                 normalize=normalize, seed=seed)
    val_ds   = AlphaSoundDataset(root_dir, "val",   augment_prob=0.0,
                                 normalize=normalize, seed=seed)
    test_ds  = AlphaSoundDataset(root_dir, "test",  augment_prob=0.0,
                                 normalize=normalize, seed=seed)

    # Optional: balance classes in the training set via oversampling
    train_sampler = None
    if use_weighted_sampler:
        weights = train_ds.class_weights()
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # shuffle only if no custom sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # drop incomplete last batch for stable batch norm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(train_ds.summary())
    print(val_ds.summary())
    print(test_ds.summary())

    return train_loader, val_loader, test_loader


# ── Label utilities (used by train.py and evaluate.py) ───────────────────────

def decode_predictions(
    letter_logits: torch.Tensor,
    lang_logits: torch.Tensor,
) -> list[dict]:
    """
    Convert raw model logits to human-readable predictions.

    Parameters
    ----------
    letter_logits : torch.Tensor, shape (B, n_letters)
    lang_logits   : torch.Tensor, shape (B, 2)

    Returns
    -------
    list of {"letter": str, "language": str, "letter_conf": float, "lang_conf": float}
    """
    letter_probs = torch.softmax(letter_logits, dim=-1)
    lang_probs   = torch.softmax(lang_logits,   dim=-1)

    letter_idx = letter_probs.argmax(dim=-1).tolist()
    lang_idx   = lang_probs.argmax(dim=-1).tolist()

    results = []
    for li, la in zip(letter_idx, lang_idx):
        results.append({
            "letter":      IDX_TO_LETTER[li],
            "language":    IDX_TO_LANG[la],
            "letter_conf": round(letter_probs[letter_idx.index(li)][li].item(), 4),
            "lang_conf":   round(lang_probs[lang_idx.index(la)][la].item(), 4),
        })
    return results


def save_label_maps(output_path: str = "data/label_maps.json") -> None:
    """Serialise LETTER_TO_IDX and LANGUAGE_TO_IDX to JSON for inference."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "letter_to_idx": LETTER_TO_IDX,
        "lang_to_idx":   LANGUAGE_TO_IDX,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Label maps saved to {output_path}")