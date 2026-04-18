"""
backend/core/model_loader.py

Singleton module to cache the YOLOv8 and MobileNetV2 models.
Thread-safe load via _lock. Both models are loaded once on startup
and reused across every inference request.

Architecture (v2):
  Stage 1 — YOLOv8s (yolov8_v2.pt):   detects sign regions via tiling
  Stage 2 — MobileNetV2 (best_mobilenet_v2.pth):  classifies crops
             8 output logits: index 0 = background, indices 1–7 = Zalo classes
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

from backend.config.settings import Settings


class _ModelStore:
    """Internal container holding the loaded model instances."""

    def __init__(self) -> None:
        self.yolo: Optional[YOLO] = None
        self.cnn: Optional[nn.Module] = None   # MobileNetV2 — 8-class Zalo classifier


_store = _ModelStore()
_lock = threading.Lock()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_mobilenetv2(num_classes: int) -> nn.Module:
    """Build a MobileNetV2 with a custom classification head.

    Args:
        num_classes: Total output units (8 for Zalo: 1 background + 7 classes).
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


# ---------------------------------------------------------------------------
# Public load function
# ---------------------------------------------------------------------------

def load_models(settings: Settings) -> None:
    """Load YOLOv8 and MobileNetV2 into the module-level cache.

    Uses a threading lock so concurrent startup calls are safe.
    """
    global _store

    with _lock:
        if _store.yolo is not None and _store.cnn is not None:
            return  # Already loaded

        yolo_path: Path = settings.yolo_model_path
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

        cnn_path: Path = settings.cnn_model_path
        if not cnn_path.exists():
            raise FileNotFoundError(f"CNN model not found: {cnn_path}")

        print(f"[ModelLoader] Loading YOLOv8 from: {yolo_path}")
        _store.yolo = YOLO(str(yolo_path))

        print(
            f"[ModelLoader] Loading MobileNetV2 "
            f"({settings.cnn_num_classes} classes) from: {cnn_path}"
        )
        cnn = _build_mobilenetv2(settings.cnn_num_classes)
        state_dict = torch.load(cnn_path, map_location="cpu", weights_only=True)
        cnn.load_state_dict(state_dict)
        cnn.to(get_device())
        cnn.eval()
        _store.cnn = cnn

        print("[ModelLoader] YOLOv8 + MobileNetV2 loaded successfully.")


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_yolo() -> YOLO:
    if _store.yolo is None:
        raise RuntimeError("YOLO model not loaded. Call load_models() first.")
    return _store.yolo


def get_cnn() -> nn.Module:
    if _store.cnn is None:
        raise RuntimeError("CNN model not loaded. Call load_models() first.")
    return _store.cnn
