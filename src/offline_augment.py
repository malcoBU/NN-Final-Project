"""
offline_augment.py
------------------
Generates augmented versions of every .npy waveform in data/processed/
and saves them as new files, multiplying the dataset size.

Why offline (pre-generated) instead of online (in the DataLoader)
------------------------------------------------------------------
With only ~540 original samples, even with online augmentation the model
sees few variations per epoch. By pre-generating ×15 augmented copies, the
dataset grows to ~8,100 unique samples that the model can explore freely.

Output structure
----------------
Each file  data/processed/english/a_EN_1.npy
generates  data/processed/english/a_EN_1_aug_01.npy
           data/processed/english/a_EN_1_aug_02.npy
           ...
           data/processed/english/a_EN_1_aug_15.npy

Original files are NOT modified or deleted.
Existing _aug_ files are skipped to avoid re-augmenting.

Usage
-----
    # From the project root:
    python src/offline_augment.py

    # With options:
    python src/offline_augment.py --data_dir data/processed --n_aug 15
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from augment import augment


# ── Main function ─────────────────────────────────────────────────────────────

def generate_augmented_dataset(
    data_dir: str,
    n_aug: int = 15,
    p_apply: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Walk data_dir looking for original .npy files (without _aug_ in the name)
    and generate n_aug augmented versions of each.

    Parameters
    ----------
    data_dir : str
        Root directory containing the .npy files (english/ and spanish/).
    n_aug : int
        Number of augmented copies per original file.
    p_apply : float
        Probability of applying each individual transform.
        With 1.0 all transforms are applied; with 0.8 there is some extra variability.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    stats : dict
        {"original": int, "generated": int, "skipped": int, "failed": int}
    """
    data_dir = Path(data_dir)
    stats = {"original": 0, "generated": 0, "skipped": 0, "failed": 0}

    # Find all original .npy files directly in data_dir
    # (the language is encoded in the filename, not the subfolder)
    originals = [
        f for f in sorted(data_dir.rglob("*.npy"))
        if "_aug_" not in f.stem
    ]

    if not originals:
        print(f"No .npy files found in '{data_dir}'.")
        return stats

    stats["original"] = len(originals)
    total_to_generate = len(originals) * n_aug

    if verbose:
        print(f"Original files found  : {len(originals)}")
        print(f"Copies per file       : {n_aug}")
        print(f"Total to generate     : {total_to_generate}")
        print(f"Estimated final size  : {len(originals) + total_to_generate}\n")

    for i, npy_path in enumerate(originals):
        if verbose:
            # Simple progress bar
            pct = (i + 1) / len(originals) * 100
            print(f"  [{pct:5.1f}%] {npy_path.name}", end="  ")

        try:
            y_original = np.load(str(npy_path))
        except Exception as e:
            if verbose:
                print(f"ERROR loading: {e}")
            stats["failed"] += 1
            continue

        generated_count = 0
        for aug_idx in range(1, n_aug + 1):
            # Augmented filename: a_EN_1_aug_01.npy
            aug_stem = f"{npy_path.stem}_aug_{aug_idx:02d}"
            aug_path = npy_path.parent / f"{aug_stem}.npy"

            # Skip if already exists (allows re-running the script without duplicating)
            if aug_path.exists():
                stats["skipped"] += 1
                continue

            try:
                y_aug = augment(y_original, p_apply=p_apply)
                np.save(str(aug_path), y_aug)
                generated_count += 1
                stats["generated"] += 1
            except Exception as e:
                stats["failed"] += 1
                if verbose:
                    print(f"\n    ERROR on aug {aug_idx}: {e}", end="")

        if verbose:
            print(f"→ +{generated_count} files")

    if verbose:
        print(f"\n{'─' * 50}")
        print(f"Originals     : {stats['original']}")
        print(f"Generated     : {stats['generated']}")
        print(f"Already exist : {stats['skipped']}")
        print(f"Errors        : {stats['failed']}")
        print(f"Total on disk : {stats['original'] + stats['generated']}")
        print(f"{'─' * 50}")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate offline augmented versions of the dataset .npy files"
    )
    p.add_argument(
        "--data_dir", default="data/processed",
        help="Root directory containing the .npy files (default: data/processed)"
    )
    p.add_argument(
        "--n_aug", type=int, default=15,
        help="Number of augmented copies per original file (default: 15)"
    )
    p.add_argument(
        "--p_apply", type=float, default=1.0,
        help="Probability of applying each transform (default: 1.0)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n── Offline augmentation ────────────────────────────────")
    print(f"Directory : {args.data_dir}")
    print(f"Copies    : ×{args.n_aug} per original file\n")

    generate_augmented_dataset(
        data_dir=args.data_dir,
        n_aug=args.n_aug,
        p_apply=args.p_apply,
        verbose=True,
    )
