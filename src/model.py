"""
model.py
--------
EfficientNet-B0 pre-entrenado en ImageNet como backbone compartido,
con dos cabezas de clasificación independientes:
  • letter_head  → predice la letra pronunciada  (27 clases: a–z + ñ)
  • lang_head    → predice el idioma (English=0, Spanish=1)

Transfer learning:
  El backbone (EfficientNet-B0) viene con pesos pre-entrenados en ImageNet.
  Aunque ImageNet es visión natural, sus filtros reconocen bordes, texturas
  y patrones locales que son igualmente útiles en espectrogramas.
  Se modifica solo el primer conv para aceptar 1 canal (escala de grises)
  en lugar de 3, promediando los pesos pre-entrenados.

  Estrategia de fine-tuning recomendada:
  • Fase 1: congelar todo el backbone, entrenar solo las cabezas (5-10 épocas)
  • Fase 2: descongelar y entrenar todo con LR muy bajo (CosineAnnealingLR)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import EfficientNet_B0_Weights


# ── Modelo principal ──────────────────────────────────────────────────────────

class AudioLetterClassifier(nn.Module):
    """
    EfficientNet-B0 con cabezas duales para clasificación de letra e idioma.

    Arquitectura
    ------------
    Input : (B, 1, 128, 128)  — log-Mel spectrogram, 1 canal

    Backbone EfficientNet-B0 (pre-entrenado ImageNet):
      • Primer Conv2d modificado: 3 canales → 1 canal
        (pesos inicializados como media de los 3 canales originales)
      • Extrae un mapa de features de 1280 dimensiones
      • AdaptiveAvgPool2d → (B, 1280)

    Cabezas:
      letter_head : Dropout → Linear(1280 → n_letters)
      lang_head   : Dropout → Linear(1280 → n_langs)

    Parameters
    ----------
    n_letters : int
        Número de clases de letras (default 27: a–z + ñ).
    n_langs : int
        Número de clases de idioma (default 2: EN + ES).
    dropout : float
        Dropout antes de las cabezas de clasificación.
    freeze_backbone : bool
        Si True, congela el backbone para la Fase 1 del fine-tuning.
        Llama a unfreeze_backbone() para descongelar en Fase 2.
    """

    def __init__(
        self,
        n_letters: int = 27,
        n_langs: int = 2,
        dropout: float = 0.4,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ── Cargar EfficientNet-B0 pre-entrenado ──────────────────────────────
        efficientnet = tv_models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # ── Adaptar primer Conv2d: 3 canales → 1 canal ────────────────────────
        # EfficientNet-B0: features[0] es ConvNormActivation,
        # y features[0][0] es el Conv2d de entrada.
        first_conv = efficientnet.features[0][0]  # Conv2d(3, 32, ...)

        new_first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        # Inicializar con la media de los 3 canales originales:
        # preserva la mayoría del conocimiento pre-entrenado
        new_first_conv.weight.data = first_conv.weight.data.mean(
            dim=1, keepdim=True
        )
        efficientnet.features[0][0] = new_first_conv

        # ── Extraer solo la parte de features (sin el classifier original) ────
        self.backbone = efficientnet.features   # → (B, 1280, H', W')
        self.pool     = nn.AdaptiveAvgPool2d(1) # → (B, 1280, 1, 1)
        self.dropout  = nn.Dropout(p=dropout)

        # ── Cabezas de clasificación ──────────────────────────────────────────
        # 1280 = número de features de salida de EfficientNet-B0
        self.letter_head = nn.Linear(1280, n_letters)
        self.lang_head   = nn.Linear(1280, n_langs)

        # Inicialización de las cabezas
        nn.init.xavier_uniform_(self.letter_head.weight)
        nn.init.zeros_(self.letter_head.bias)
        nn.init.xavier_uniform_(self.lang_head.weight)
        nn.init.zeros_(self.lang_head.bias)

        if freeze_backbone:
            self.freeze_backbone()

    # ── Fine-tuning helpers ───────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Fase 1: congela el backbone para entrenar solo las cabezas.
        Útil cuando el dataset es muy pequeño — evita destruir los pesos
        pre-entrenados antes de que las cabezas aprendan algo útil.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone congelado. Solo se entrenan las cabezas.")

    def unfreeze_backbone(self) -> None:
        """
        Fase 2: descongela el backbone para fine-tuning completo.
        Llamar después de que las cabezas hayan convergido (Fase 1).
        Usar con un LR mucho más bajo (e.g., 1e-4 o menos).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone descongelado. Fine-tuning completo activado.")

    def unfreeze_last_n_blocks(self, n: int = 3) -> None:
        """
        Alternativa: descongela solo los últimos n bloques del backbone.
        Compromiso entre Fase 1 y Fase 2 completo.
        """
        blocks = list(self.backbone.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Últimos {n} bloques del backbone descongelados.")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 128, 128)

        Returns
        -------
        letter_logits : (B, n_letters)
        lang_logits   : (B, n_langs)
        """
        features = self.backbone(x)          # (B, 1280, H', W')
        features = self.pool(features)       # (B, 1280, 1, 1)
        features = features.flatten(1)       # (B, 1280)
        features = self.dropout(features)

        return self.letter_head(features), self.lang_head(features)

    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inferencia: devuelve índices de clase directamente."""
        self.eval()
        with torch.no_grad():
            letter_logits, lang_logits = self.forward(x)
        return letter_logits.argmax(dim=-1), lang_logits.argmax(dim=-1)

    def count_parameters(self) -> int:
        """Parámetros entrenables totales."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_trainable_vs_total(self) -> tuple[int, int]:
        """Devuelve (entrenables, total) — útil para ver qué está congelado."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


# ── Loss function (sin cambios) ───────────────────────────────────────────────

class DualTaskLoss(nn.Module):
    """
    Suma ponderada de dos CrossEntropy (letra + idioma).

    Total Loss = letter_weight × CE(letra) + lang_weight × CE(idioma)

    Parameters
    ----------
    letter_weight : float
        Peso de la loss de letra (tarea principal). Default 0.7.
    lang_weight : float
        Peso de la loss de idioma (tarea auxiliar). Default 0.3.
    label_smoothing : float
        Suavizado de etiquetas para evitar predicciones sobreconfiadas.
    """

    def __init__(
        self,
        letter_weight: float = 0.7,
        lang_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.letter_weight = letter_weight
        self.lang_weight   = lang_weight

        self.letter_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.lang_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        letter_logits: torch.Tensor,
        lang_logits: torch.Tensor,
        letter_targets: torch.Tensor,
        lang_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss, letter_loss, lang_loss  — todos escalares
        """
        l_loss  = self.letter_criterion(letter_logits, letter_targets)
        la_loss = self.lang_criterion(lang_logits,     lang_targets)
        total   = self.letter_weight * l_loss + self.lang_weight * la_loss
        return total, l_loss, la_loss
