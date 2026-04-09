"""
src/core/mapping.py

Canonical traffic sign category mapping.
Source: Zalo AI 2020 traffic sign dataset (IDs 1–7) + Custom 2026 standards (IDs 100–102).

CNN model outputs class indices in range [0, 102] (103 total).
Index 0 is reserved/background; valid indices are 1–7 and 100–102.
"""

from __future__ import annotations

SIGN_NAMES: dict[int, str] = {
    1: "Cấm ngược chiều",
    2: "Cấm dừng và đỗ",
    3: "Cấm rẽ",
    4: "Giới hạn tốc độ",
    5: "Cấm còn lại",
    6: "Nguy hiểm",
    7: "Hiệu lệnh",
    100: "LED",
    101: "Metro",
    102: "TramSac",
}

# IDs that are newly proposed 2026 standards — always valid by definition.
NEW_2026_SIGN_IDS: frozenset[int] = frozenset({100, 101, 102})


def get_sign_name(class_id: int) -> str:
    """Return the Vietnamese category name for a given CNN class index.

    Args:
        class_id: Integer in range [0, 102] produced by the MobileNetV2 classifier.

    Returns:
        Vietnamese category string, or a fallback label for unknown indices.
    """
    return SIGN_NAMES.get(class_id, f"Không xác định (ID={class_id})")


def is_new_2026_sign(class_id: int) -> bool:
    """Return True if the sign belongs to the newly proposed 2026 standard set.

    These signs are not covered by QCVN 41:2019 and are always considered valid.

    Args:
        class_id: Integer class index from the CNN output.

    Returns:
        True if the sign is a 2026 standard sign; False otherwise.
    """
    return class_id in NEW_2026_SIGN_IDS
