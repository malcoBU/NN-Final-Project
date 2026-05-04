"""
model.py
--------
Pre-trained EfficientNet-B0 (ImageNet) as a shared backbone,
with two independent classification heads:
  • letter_head  → predicts the spoken letter  (27 classes: a–z + ñ)
  • lang_head    → predicts the language (English=0, Spanish=1)

Transfer learning:
  The backbone (EfficientNet-B0) comes with weights pre-trained on ImageNet.
  Although ImageNet is natural vision, its filters recognise edges, textures,
  and local patterns that are equally useful in spectrograms.
  Only the first conv is modified to accept 1 channel (greyscale) instead of 3,
  by averaging the pre-trained weights across the three original channels.

  Recommended fine-tuning strategy:
  • Phase 1: freeze the entire backbone, train only the heads (5–10 epochs)
  • Phase 2: unfreeze and train everything with a very low LR (CosineAnnealingLR)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import EfficientNet_B0_Weights


# ── Main model ────────────────────────────────────────────────────────────────

class AudioLetterClassifier(nn.Module):
    """
    EfficientNet-B0 with dual heads for letter and language classification.

    Architecture
    ------------
    Input : (B, 1, 128, 128)  — log-Mel spectrogram, 1 channel

    EfficientNet-B0 backbone (pre-trained on ImageNet):
      • First Conv2d modified: 3 channels → 1 channel
        (weights initialised as the mean of the 3 original channels)
      • Extracts a 1280-dimensional feature map
      • AdaptiveAvgPool2d → (B, 1280)

    Heads:
      letter_head : Dropout → Linear(1280 → n_letters)
      lang_head   : Dropout → Linear(1280 → n_langs)

    Parameters
    ----------
    n_letters : int
        Number of letter classes (default 27: a–z + ñ).
    n_langs : int
        Number of language classes (default 2: EN + ES).
    dropout : float
        Dropout applied before the classification heads.
    freeze_backbone : bool
        If True, freezes the backbone for Phase 1 fine-tuning.
        Call unfreeze_backbone() to unfreeze for Phase 2.
    """

    def __init__(
        self,
        n_letters: int = 27,
        n_langs: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ── Load pre-trained EfficientNet-B0 ──────────────────────────────────
        efficientnet = tv_models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # ── Adapt first Conv2d: 3 channels → 1 channel ────────────────────────
        # EfficientNet-B0: features[0] is ConvNormActivation,
        # and features[0][0] is the input Conv2d.
        first_conv = efficientnet.features[0][0]  # Conv2d(3, 32, ...)

        new_first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        # Initialise with the mean of the 3 original channels:
        # preserves most of the pre-trained knowledge
        new_first_conv.weight.data = first_conv.weight.data.mean(
            dim=1, keepdim=True
        )
        efficientnet.features[0][0] = new_first_conv

        # ── Extract only the feature part (without the original classifier) ───
        self.backbone = efficientnet.features   # → (B, 1280, H', W')
        self.pool     = nn.AdaptiveAvgPool2d(1) # → (B, 1280, 1, 1)
        self.dropout  = nn.Dropout(p=dropout)

        # ── Classification heads ──────────────────────────────────────────────
        # 1280 = number of output features from EfficientNet-B0
        self.letter_head = nn.Linear(1280, n_letters)
        self.lang_head   = nn.Linear(1280, n_langs)

        # Head initialisation
        nn.init.xavier_uniform_(self.letter_head.weight)
        nn.init.zeros_(self.letter_head.bias)
        nn.init.xavier_uniform_(self.lang_head.weight)
        nn.init.zeros_(self.lang_head.bias)

        if freeze_backbone:
            self.freeze_backbone()

    # ── Fine-tuning helpers ───────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Phase 1: freeze the backbone to train only the heads.
        Useful when the dataset is very small — prevents destroying the
        pre-trained weights before the heads have learned anything useful.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only the heads will be trained.")

    def unfreeze_backbone(self) -> None:
        """
        Phase 2: unfreeze the backbone for full fine-tuning.
        Call after the heads have converged (Phase 1).
        Use with a much lower LR (e.g. 1e-4 or less).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. Full fine-tuning enabled.")

    def unfreeze_last_n_blocks(self, n: int = 3) -> None:
        """
        Alternative: unfreeze only the last n blocks of the backbone.
        A compromise between Phase 1 and full Phase 2 fine-tuning.
        """
        blocks = list(self.backbone.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Last {n} backbone blocks unfrozen.")

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
        """Inference: returns class indices directly."""
        self.eval()
        with torch.no_grad():
            letter_logits, lang_logits = self.forward(x)
        return letter_logits.argmax(dim=-1), lang_logits.argmax(dim=-1)

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_trainable_vs_total(self) -> tuple[int, int]:
        """Returns (trainable, total) — useful for inspecting what is frozen."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


# ── Loss function ─────────────────────────────────────────────────────────────

class DualTaskLoss(nn.Module):
    """
    Weighted sum of two CrossEntropy losses (letter + language).

    Total Loss = letter_weight × CE(letter) + lang_weight × CE(language)

    Parameters
    ----------
    letter_weight : float
        Weight for the letter loss (primary task). Default 0.7.
    lang_weight : float
        Weight for the language loss (auxiliary task). Default 0.3.
    label_smoothing : float
        Label smoothing to prevent overconfident predictions.
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
        total_loss, letter_loss, lang_loss  — all scalars
        """
        l_loss  = self.letter_criterion(letter_logits, letter_targets)
        la_loss = self.lang_criterion(lang_logits,     lang_targets)
        total   = self.letter_weight * l_loss + self.lang_weight * la_loss
        return total, l_loss, la_loss
