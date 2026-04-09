"""
backend/config/settings.py

Environment-driven configuration using pydantic-settings.
Values are loaded from environment variables (or .env file).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API metadata
    app_title: str = "Highway Guardian API"
    app_version: str = "2.0.0"
    api_prefix: str = "/api/v1"

    # CORS
    cors_origins: list[str] = [
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:8501",
        "http://localhost:3000",
    ]

    # Database
    database_url: str = "sqlite+aiosqlite:///./highway_guardian.db"

    # Base Directory (Khai báo là ClassVar để Pydantic không coi nó là một trường cần validate)
    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent

    # Model & Doc Paths (Giá trị mặc định sẽ bị ghi đè bởi file .env)
    yolo_model_path: Path = Path("models/yolo/yolov8.pt")
    cnn_model_path: Path = Path("models/cnn/best_mobilenet_v2.pth")
    pdf_path: Path = Path("docs/quy-chuan-ky-thuat-qcvn-41-2019-bgtvt-bao-hieu-duong-bo.pdf")
    
    cnn_num_classes: int = 103
    cnn_input_size: tuple[int, int] = (224, 224)

    # Detection defaults
    default_confidence_threshold: float = 0.25
    default_iou_threshold: float = 0.45

    # Storage
    uploads_dir: Path = Path("uploads")

    # Security
    max_upload_size_mb: int = 10

    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance."""
    return Settings()