"""
backend/db/schemas.py

SQLModel ORM table definition for detection history.
Each row records one sign detected in a single inference request.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class DetectionRecord(SQLModel, table=True):
    """Persistent record of a single traffic sign detection event.

    Attributes:
        id: Auto-incremented primary key.
        timestamp: UTC datetime of the detection request.
        image_path: Relative path to the saved upload (under uploads_dir).
        class_id: CNN output class index in range [1–7, 100–102].
        class_name: Vietnamese category name resolved from SIGN_NAMES.
        confidence: CNN softmax probability for the predicted class.
        is_valid: True if the sign is compliant with QCVN 41:2019 or is a 2026 standard.
    """

    __tablename__ = "detection_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        nullable=False,
        index=True,
    )
    image_path: str = Field(nullable=False)
    class_id: int = Field(nullable=False, index=True)
    class_name: str = Field(nullable=False)
    confidence: float = Field(nullable=False)
    is_valid: bool = Field(nullable=False, index=True)


# ---------------------------------------------------------------------------
# Pydantic response models (not table-backed)
# ---------------------------------------------------------------------------


class DetectionRecordRead(SQLModel):
    """Read-only projection of DetectionRecord for API responses."""

    id: int
    timestamp: datetime
    image_path: str
    class_id: int
    class_name: str
    confidence: float
    is_valid: bool
