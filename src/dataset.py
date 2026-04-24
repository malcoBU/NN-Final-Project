"""
dataset.py
----------
PyTorch Dataset y DataLoader para AlphaSound.

Estructura esperada bajo data/processed/
-----------------------------------------
processed/
  english/
    a_EN_1.npy   a_EN_2.npy   b_EN_1.npy   ...
  spanish/
    a_ES_1.npy   b_ES_1.npy   ...

Cada .npy es una waveform preprocesada (float32, 16 kHz) generada por
preprocess.preprocess_directory(). La letra se extrae del primer carácter
del nombre de fichero; el idioma viene del nombre de la carpeta padre.

Solo se usan las 26 letras del abecedario latino común (a–z).
Data augmentation se aplica sobre la waveform ANTES de extraer features,
por lo que funciona exactamente igual que con .wav.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from augment   import augment
from features  import extract_features, normalize_spectrogram

# ── Etiquetas ─────────────────────────────────────────────────────────────────

LANGUAGE_TO_IDX: dict[str, int] = {
    "english": 0,
    "spanish": 1,
}

# Solo las 26 letras comunes (sin ñ, ll, ch, rr)
ALL_LETTERS: list[str] = list("abcdefghijklmnopqrstuvwxyz")

LETTER_TO_IDX: dict[str, int] = {l: i for i, l in enumerate(ALL_LETTERS)}
IDX_TO_LETTER: dict[int, str] = {i: l for l, i in LETTER_TO_IDX.items()}
IDX_TO_LANG:   dict[int, str] = {v: k for k, v in LANGUAGE_TO_IDX.items()}


# ── Dataset ───────────────────────────────────────────────────────────────────

class AlphaSoundDataset(Dataset):
    """
    Carga waveforms preprocesadas en formato .npy, aplica augmentación
    opcional y extrae features de mel-spectrogram.

    Devuelve tuplas (tensor, letter_idx, lang_idx).

    Parameters
    ----------
    root_dir : str
        Ruta al directorio con los .npy procesados
        (contiene subcarpetas 'english/' y 'spanish/').
    split : str
        'train', 'val' o 'test'.
    split_ratios : tuple[float, float, float]
        Proporciones train/val/test. Deben sumar 1.0.
    augment_prob : float
        Probabilidad de aplicar augmentación a cada muestra de entrenamiento.
        Se ignora en val/test (se fuerza a 0).
    normalize : bool
        Si True, estandariza cada espectrograma a media 0 / varianza 1.
    seed : int
        Semilla para reproducibilidad del split.
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
            f"split debe ser 'train', 'val' o 'test', recibido: '{split}'"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, \
            "split_ratios debe sumar 1.0"

        self.root_dir     = Path(root_dir)
        self.split        = split
        self.augment_prob = augment_prob if split == "train" else 0.0
        self.normalize    = normalize

        all_samples = self._scan_directory()

        if not all_samples:
            raise RuntimeError(
                f"No se encontraron ficheros .npy bajo '{root_dir}'.\n"
                "Comprueba que la estructura es: root/english/*.npy y root/spanish/*.npy"
            )

        # Shuffle reproducible + split
        random.seed(seed)
        random.shuffle(all_samples)
        n       = len(all_samples)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])

        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train : n_train + n_val]
        else:
            self.samples = all_samples[n_train + n_val :]

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _scan_directory(self) -> list[dict]:
        """
        Recorre root_dir buscando ficheros .npy.

        La letra se extrae del primer carácter del nombre de fichero
        (ej: 'a_EN_1.npy' → letra = 'a').
        El idioma se extrae del nombre de la carpeta padre
        (ej: carpeta 'english' → lang_idx = 0).

        Devuelve
        --------
        list de dicts {"path": str, "letter_idx": int, "lang_idx": int}
        """
        samples = []

        for lang_dir in sorted(self.root_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang_name = lang_dir.name.lower()
            if lang_name not in LANGUAGE_TO_IDX:
                continue
            lang_idx = LANGUAGE_TO_IDX[lang_name]

            for npy_file in sorted(lang_dir.rglob("*.npy")):
                # El primer carácter del nombre es la letra
                letter = npy_file.stem[0].lower()
                if letter not in LETTER_TO_IDX:
                    continue  # ignora letras fuera del vocabulario (ñ, etc.)
                letter_idx = LETTER_TO_IDX[letter]

                samples.append({
                    "path":       str(npy_file),
                    "letter_idx": letter_idx,
                    "lang_idx":   lang_idx,
                })

        return samples

    # ── Interfaz PyTorch ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """
        Devuelve
        --------
        mel_tensor : torch.Tensor, shape (1, 128, 128), float32
        letter_idx : int
        lang_idx   : int
        """
        sample = self.samples[idx]

        # Cargar waveform preprocesada desde .npy
        y = np.load(sample["path"])

        # Augmentación sobre la waveform (solo en training)
        if self.augment_prob > 0:
            y = augment(y, p_apply=self.augment_prob)

        # Extraer mel-spectrogram → numpy (1, 128, 128)
        mel = extract_features(y)

        # Normalización por instancia
        if self.normalize:
            mel = normalize_spectrogram(mel)

        return (
            torch.from_numpy(mel),
            sample["letter_idx"],
            sample["lang_idx"],
        )

    # ── Utilidades ────────────────────────────────────────────────────────────

    def class_weights(self) -> torch.Tensor:
        """
        Pesos inversamente proporcionales a la frecuencia de cada clase de letra.
        Útil para WeightedRandomSampler cuando hay desequilibrio de clases.

        Devuelve
        --------
        weights : torch.Tensor, shape (len(self),)
        """
        from collections import Counter

        counts   = Counter(s["letter_idx"] for s in self.samples)
        n_classes = len(LETTER_TO_IDX)

        freq = torch.zeros(n_classes)
        for idx, cnt in counts.items():
            freq[idx] = cnt

        class_w = 1.0 / (freq + 1e-8)
        class_w[freq == 0] = 0.0

        return torch.tensor(
            [class_w[s["letter_idx"]].item() for s in self.samples]
        )

    def summary(self) -> str:
        """Resumen breve del split."""
        from collections import Counter
        letter_counts = Counter(IDX_TO_LETTER[s["letter_idx"]] for s in self.samples)
        lang_counts   = Counter(IDX_TO_LANG[s["lang_idx"]]   for s in self.samples)
        return (
            f"Split  : {self.split}\n"
            f"Total  : {len(self.samples)} muestras\n"
            f"Idiomas: {dict(lang_counts)}\n"
            f"Letras (top-5): {letter_counts.most_common(5)}"
        )


# ── Fábrica de DataLoaders ────────────────────────────────────────────────────

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
    Construye los DataLoaders de train, val y test de una sola vez.

    Parameters
    ----------
    root_dir : str
        Directorio raíz con los .npy (contiene english/ y spanish/).
    batch_size : int
    num_workers : int
        Pon 0 si estás en Windows o tienes problemas de multiprocessing.
    augment_prob : float
        Probabilidad de augmentación en training.
    normalize : bool
        Normalización por instancia del espectrograma.
    use_weighted_sampler : bool
        Si True, usa WeightedRandomSampler en train para equilibrar clases.
    seed : int

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_ds = AlphaSoundDataset(root_dir, "train", augment_prob=augment_prob,
                                 normalize=normalize, seed=seed)
    val_ds   = AlphaSoundDataset(root_dir, "val",   augment_prob=0.0,
                                 normalize=normalize, seed=seed)
    test_ds  = AlphaSoundDataset(root_dir, "test",  augment_prob=0.0,
                                 normalize=normalize, seed=seed)

    print(train_ds.summary())
    print(val_ds.summary())
    print(test_ds.summary())

    train_sampler = None
    if use_weighted_sampler:
        weights       = train_ds.class_weights()
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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

    return train_loader, val_loader, test_loader


# ── Utilidades de etiquetas (usadas por train.py y evaluate.py) ──────────────

def decode_predictions(
    letter_logits: torch.Tensor,
    lang_logits: torch.Tensor,
) -> list[dict]:
    """
    Convierte logits crudos del modelo en predicciones legibles.

    Parameters
    ----------
    letter_logits : torch.Tensor, shape (B, 26)
    lang_logits   : torch.Tensor, shape (B, 2)

    Returns
    -------
    list de dicts {"letter": str, "language": str,
                   "letter_conf": float, "lang_conf": float}
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
    """Guarda LETTER_TO_IDX y LANGUAGE_TO_IDX en JSON para inferencia."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "letter_to_idx": LETTER_TO_IDX,
        "lang_to_idx":   LANGUAGE_TO_IDX,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Label maps guardados en {output_path}")
