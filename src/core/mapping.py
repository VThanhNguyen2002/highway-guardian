"""
src/core/mapping.py

Zalo AI 2020 traffic sign class mapping — 7 classes + 1 background.

MobileNetV2 (best_mobilenet_v2.pth) outputs 8 logits:
  - Index 0: Background (ignored in results display)
  - Indices 1–7: The 7 Zalo AI 2020 traffic sign categories

Source: configs/zalo_classes.json
"""

from __future__ import annotations

SIGN_NAMES: dict[int, str] = {
    0: "Background",
    1: "Cấm ngược chiều",
    2: "Cấm dừng và đỗ",
    3: "Cấm rẽ",
    4: "Giới hạn tốc độ",
    5: "Cấm ô tô",
    6: "Cấm đỗ",
    7: "Cấm các phương tiện khác",
}

# Valid foreground class IDs (exclude background at index 0)
VALID_CLASS_IDS: frozenset[int] = frozenset(range(1, 8))


def get_sign_name(class_id: int) -> str:
    """Return the Vietnamese category name for a given class index.

    Args:
        class_id: Integer in range [0, 7] produced by the MobileNetV2 classifier.

    Returns:
        Vietnamese category string, or a fallback label for unknown indices.
    """
    return SIGN_NAMES.get(class_id, f"Không xác định (ID={class_id})")


def is_valid_class(class_id: int) -> bool:
    """Return True if class_id is a foreground Zalo traffic sign class.

    Args:
        class_id: Integer class index from the CNN output.

    Returns:
        True if the sign is one of the 7 Zalo classes; False for background or unknown.
    """
    return class_id in VALID_CLASS_IDS
