"""
offline_augment.py
------------------
Genera versiones aumentadas de cada waveform .npy en data/processed/
y las guarda como nuevos ficheros, multiplicando el dataset.

Por qué offline (pre-generado) en lugar de online (en el DataLoader)
---------------------------------------------------------------------
Con solo ~540 muestras originales, incluso con augmentación online el modelo
ve pocas variaciones por época. Generando ×15 copias pre-aumentadas el
dataset pasa a ~8.100 muestras únicas que el modelo puede explorar libremente.

Estructura de salida
--------------------
Cada fichero  data/processed/english/a_EN_1.npy
genera        data/processed/english/a_EN_1_aug_01.npy
              data/processed/english/a_EN_1_aug_02.npy
              ...
              data/processed/english/a_EN_1_aug_15.npy

Los ficheros originales NO se modifican ni eliminan.
Los ficheros _aug_ existentes se saltan para no re-aumentar.

Uso
---
    # Desde la raíz del proyecto:
    python src/offline_augment.py

    # Con opciones:
    python src/offline_augment.py --data_dir data/processed --n_aug 15
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Asegurar que src/ está en el path
sys.path.insert(0, os.path.dirname(__file__))

from augment import augment


# ── Función principal ─────────────────────────────────────────────────────────

def generate_augmented_dataset(
    data_dir: str,
    n_aug: int = 15,
    p_apply: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Recorre data_dir buscando ficheros .npy originales (sin _aug_ en el nombre)
    y genera n_aug versiones aumentadas de cada uno.

    Parameters
    ----------
    data_dir : str
        Directorio raíz con los .npy (contiene english/ y spanish/).
    n_aug : int
        Número de copias aumentadas por fichero original.
    p_apply : float
        Probabilidad de aplicar cada transform individual.
        Con 1.0 se aplican todas; con 0.8 hay algo de variabilidad extra.
    verbose : bool
        Muestra progreso en pantalla.

    Returns
    -------
    stats : dict
        {"original": int, "generated": int, "skipped": int, "failed": int}
    """
    data_dir = Path(data_dir)
    stats = {"original": 0, "generated": 0, "skipped": 0, "failed": 0}

    # Buscar todos los .npy originales directamente en data_dir
    # (sin subcarpetas english/ ni spanish/ — el idioma va en el nombre del fichero)
    originals = [
        f for f in sorted(data_dir.rglob("*.npy"))
        if "_aug_" not in f.stem
    ]

    if not originals:
        print(f"No se encontraron ficheros .npy en '{data_dir}'.")
        return stats

    stats["original"] = len(originals)
    total_to_generate = len(originals) * n_aug

    if verbose:
        print(f"Ficheros originales encontrados : {len(originals)}")
        print(f"Copias por fichero               : {n_aug}")
        print(f"Total a generar                  : {total_to_generate}")
        print(f"Dataset final estimado           : {len(originals) + total_to_generate}\n")

    for i, npy_path in enumerate(originals):
        if verbose:
            # Barra de progreso simple
            pct = (i + 1) / len(originals) * 100
            print(f"  [{pct:5.1f}%] {npy_path.name}", end="  ")

        try:
            y_original = np.load(str(npy_path))
        except Exception as e:
            if verbose:
                print(f"ERROR al cargar: {e}")
            stats["failed"] += 1
            continue

        generated_count = 0
        for aug_idx in range(1, n_aug + 1):
            # Nombre del fichero aumentado: a_EN_1_aug_01.npy
            aug_stem = f"{npy_path.stem}_aug_{aug_idx:02d}"
            aug_path = npy_path.parent / f"{aug_stem}.npy"

            # Saltar si ya existe (para poder relanzar el script sin duplicar)
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
                    print(f"\n    ERROR en aug {aug_idx}: {e}", end="")

        if verbose:
            print(f"→ +{generated_count} ficheros")

    if verbose:
        print(f"\n{'─' * 50}")
        print(f"Originales      : {stats['original']}")
        print(f"Generados       : {stats['generated']}")
        print(f"Ya existían     : {stats['skipped']}")
        print(f"Errores         : {stats['failed']}")
        print(f"Total en disco  : {stats['original'] + stats['generated']}")
        print(f"{'─' * 50}")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera versiones aumentadas offline de los .npy del dataset"
    )
    p.add_argument(
        "--data_dir", default="data/processed",
        help="Directorio raíz con los .npy (default: data/processed)"
    )
    p.add_argument(
        "--n_aug", type=int, default=15,
        help="Número de copias aumentadas por fichero original (default: 15)"
    )
    p.add_argument(
        "--p_apply", type=float, default=1.0,
        help="Probabilidad de aplicar cada transform (default: 1.0)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n── Augmentación offline ────────────────────────────────")
    print(f"Directorio : {args.data_dir}")
    print(f"Copias     : ×{args.n_aug} por fichero original\n")

    generate_augmented_dataset(
        data_dir=args.data_dir,
        n_aug=args.n_aug,
        p_apply=args.p_apply,
        verbose=True,
    )
