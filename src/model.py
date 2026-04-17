"""
model.py
--------
CNN backbone with two independent classification heads:
  • letter_head  → predicts which letter was spoken  (up to 30 classes)
  • lang_head    → predicts the language (English = 0, Spanish = 1)

The backbone is shared: both tasks learn from the same feature extractor,
which forces the network to build representations useful for both.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2d → BatchNorm2d → ReLU  (optionally followed by MaxPool2d).

    BatchNorm normalises the activations of each mini-batch, which:
      • speeds up training (allows higher learning rates)
      • acts as a mild regulariser (reduces need for large Dropout)
      • makes the network less sensitive to weight initialisation

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of filters (output feature maps) to learn.
    kernel_size : int
        Side length of the convolution window (3 = 3×3).
    pool : bool
        Whether to halve the spatial dimensions with MaxPool2d(2) after ReLU.
    dropout : float
        Spatial dropout probability (0 = disabled).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2,  # 'same' padding: output H,W unchanged
                bias=False,                # bias redundant when followed by BN
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0:
            # Dropout2d zeroes entire feature maps (channels), more effective
            # for CNNs than element-wise dropout
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionPool(nn.Module):
    """
    Soft attention pooling over the spatial dimensions.

    Instead of simply averaging or taking the max across (H, W), this module
    learns a scalar attention weight per spatial location so the network can
    focus on the most discriminative time-frequency regions.

    Input  : (B, C, H, W)
    Output : (B, C)   — channel-wise weighted sum over H×W
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # 1×1 conv reduces C channels to 1 scalar per spatial location
        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weights: (B, 1, H, W) → softmax over H*W positions
        w = self.attn(x)
        w = w.view(w.size(0), 1, -1)          # (B, 1, H*W)
        w = F.softmax(w, dim=-1)
        w = w.view(w.size(0), 1, x.size(2), x.size(3))  # (B, 1, H, W)

        # Weighted sum: (B, C, H, W) × (B, 1, H, W) → (B, C)
        pooled = (x * w).sum(dim=[2, 3])
        return pooled


# ── Main model ────────────────────────────────────────────────────────────────

class AudioLetterClassifier(nn.Module):
    """
    CNN with a shared backbone and two linear classification heads.

    Architecture
    ------------
    Input: (B, 1, 128, 128)  — batch of log-Mel spectrograms

    Backbone (3 conv blocks):
      ConvBlock(1→32,   pool=True)   → (B, 32,  64, 64)
      ConvBlock(32→64,  pool=True)   → (B, 64,  32, 32)
      ConvBlock(64→128, pool=True)   → (B, 128, 16, 16)
      ConvBlock(128→256, pool=False) → (B, 256, 16, 16)
      AttentionPool                  → (B, 256)
      Dropout(0.5)

    Heads:
      letter_head: Linear(256 → n_letters)
      lang_head:   Linear(256 → n_langs)

    Parameters
    ----------
    n_letters : int
        Number of letter classes (default 30: 26 EN + ñ/ch/ll/rr).
    n_langs : int
        Number of language classes (default 2: EN + ES).
    dropout : float
        Dropout probability before the classification heads.
    """

    def __init__(
        self,
        n_letters: int = 30,
        n_langs: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            ConvBlock(1,    32,  pool=True,  dropout=0.1),   # 128→64
            ConvBlock(32,   64,  pool=True,  dropout=0.1),   # 64→32
            ConvBlock(64,  128,  pool=True,  dropout=0.2),   # 32→16
            ConvBlock(128, 256,  pool=False, dropout=0.0),   # 16→16 (no pool)
        )

        # Attention pooling replaces the naive Flatten+AdaptiveAvgPool combo
        self.pool    = AttentionPool(in_channels=256)
        self.dropout = nn.Dropout(p=dropout)

        # ── Heads ─────────────────────────────────────────────────────────────
        # Both heads share the same 256-dim feature vector from the backbone.
        # Each head is a single Linear layer — enough for the final mapping
        # because the backbone already learned a rich representation.
        self.letter_head = nn.Linear(256, n_letters)
        self.lang_head   = nn.Linear(256, n_langs)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He initialisation for Conv layers; Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 128, 128)

        Returns
        -------
        letter_logits : torch.Tensor, shape (B, n_letters)
            Raw scores for each letter class (no softmax applied).
        lang_logits   : torch.Tensor, shape (B, n_langs)
            Raw scores for each language class.
        """
        features = self.backbone(x)   # (B, 256, 16, 16)
        features = self.pool(features) # (B, 256)
        features = self.dropout(features)

        return self.letter_head(features), self.lang_head(features)

    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference: returns class indices directly.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 128, 128)

        Returns
        -------
        letter_preds : torch.Tensor, shape (B,) — letter class indices
        lang_preds   : torch.Tensor, shape (B,) — language class indices
        """
        self.eval()
        with torch.no_grad():
            letter_logits, lang_logits = self.forward(x)
        return letter_logits.argmax(dim=-1), lang_logits.argmax(dim=-1)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss function ─────────────────────────────────────────────────────────────

class DualTaskLoss(nn.Module):
    """
    Weighted sum of two CrossEntropy losses (letter + language).

    Total Loss = letter_weight × CE(letter) + lang_weight × CE(language)

    Parameters
    ----------
    letter_weight : float
        Weight for the letter prediction loss. Higher = more gradient
        towards letter accuracy. Default 0.7 (primary task).
    lang_weight : float
        Weight for the language prediction loss. Default 0.3 (auxiliary task).
    label_smoothing : float
        Fraction of probability mass to spread across wrong classes.
        Prevents overconfident predictions; 0.1 is a good default.
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

        self.letter_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lang_criterion   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        letter_logits: torch.Tensor,
        lang_logits: torch.Tensor,
        letter_targets: torch.Tensor,
        lang_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        letter_logits   : (B, n_letters)
        lang_logits     : (B, n_langs)
        letter_targets  : (B,) int64
        lang_targets    : (B,) int64

        Returns
        -------
        total_loss   : scalar tensor
        letter_loss  : scalar tensor  (for logging)
        lang_loss    : scalar tensor  (for logging)
        """
        l_loss = self.letter_criterion(letter_logits, letter_targets)
        la_loss = self.lang_criterion(lang_logits,   lang_targets)
        total   = self.letter_weight * l_loss + self.lang_weight * la_loss
        return total, l_loss, la_loss