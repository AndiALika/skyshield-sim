"""Small CNN that detects drone RF signatures in spectrograms (binary: drone present or not).

Requires the ``ml`` extra (PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_SCALE = 10.0  # spectrograms are in dB; divide so typical values are O(1)


class SpectrogramCNN(nn.Module):
    """Four conv blocks, then global average + global max pooling.

    Global max pooling matters here: a drone control burst can occupy a few percent of the
    time-frequency plane, and averaging alone would wash it out.

    GroupNorm rather than BatchNorm: BatchNorm's running statistics lagged the weights on short
    training runs (eval-mode accuracy 0.69 vs 1.0 with batch statistics, same weights), and
    GroupNorm behaves identically in training and inference.
    """

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for i, c in enumerate(channels):
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.GroupNorm(4, c), nn.ReLU(inplace=True)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x / INPUT_SCALE)
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.head(pooled).squeeze(1)  # logits


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = -1


def train_detector(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig | None = None,
) -> tuple[SpectrogramCNN, TrainHistory]:
    """Train with Adam + cosine LR schedule; keep the weights with the lowest validation loss."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = SpectrogramCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(X) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).float()
    hist = TrainHistory()
    best, best_state = np.inf, None
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        hist.train_loss.append(total / len(Xt))
        p = predict_proba(model, X_val)
        eps = 1e-7
        vl = float(-np.mean(y_val * np.log(p + eps) + (1 - y_val) * np.log(1 - p + eps)))
        hist.val_loss.append(vl)
        hist.val_acc.append(float(np.mean((p > 0.5) == y_val)))
        if vl < best:
            best, hist.best_epoch = vl, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """P(drone present) for each spectrogram."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i : i + batch_size]))).numpy())
    model.train(was_training)
    return np.concatenate(out) if out else np.empty(0)


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path) -> SpectrogramCNN:
    model = SpectrogramCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
