"""
streamlit_app/api_client.py

HTTP client for consuming the Highway Guardian FastAPI backend.
All methods are synchronous (Streamlit runs in a synchronous context).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000")
_API_PREFIX: str = "/api/v1"
_TIMEOUT: float = 60.0  # seconds — inference can take up to ~30s on CPU


def _base_url() -> str:
    return f"{BACKEND_URL}{_API_PREFIX}"


def submit_detect_task(
    file_bytes: bytes,
    filename: str,
    confidence_threshold: float = 0.25,
    mode: str = "yolo",
) -> dict[str, Any]:
    """Send an image to the /detect endpoint to initiate a background task.

    Args:
        file_bytes: Raw bytes of the image file.
        filename: Original filename (used to set the correct MIME type).
        confidence_threshold: Minimum YOLO confidence.
        mode: Run mode ('yolo', 'cnn', 'ensemble').

    Returns:
        JSON with `task_id` and `status` ('PENDING').
    """
    content_type = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else "image/png"
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(
                f"{_base_url()}/detect",
                files={"file": (filename, file_bytes, content_type)},
                data={
                    "confidence_threshold": str(confidence_threshold),
                    "mode": mode,
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        return {"success": False, "error": exc.response.text, "task_id": None}
    except Exception as exc:
        return {"success": False, "error": str(exc), "task_id": None}


def get_detect_task_status(task_id: str) -> dict[str, Any]:
    """Poll the backend for task status.
    
    Returns:
        Parsed JSON dict with `status`, `result` and `error`.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{_base_url()}/detect/{task_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        return {"success": False, "error": exc.response.text}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def get_history(
    limit: int = 20,
    offset: int = 0,
    is_valid: Optional[bool] = None,
) -> dict[str, Any]:
    """Fetch paginated detection history from the backend.

    Args:
        limit: Max records to retrieve.
        offset: Pagination offset.
        is_valid: Optional filter for valid/invalid records.

    Returns:
        Parsed JSON dict with ``records``, ``total_returned``, ``limit``, ``offset``.
    """
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if is_valid is not None:
        params["is_valid"] = str(is_valid).lower()

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{_base_url()}/history", params=params)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        return {"records": [], "error": str(exc)}
