"""
src/utils/firebase_sync.py

Firebase Admin SDK integration for Highway Guardian.

Provides a lazy-initialized singleton Firestore client and a
``sync_detection`` function that pushes detection results to the
``detections`` collection every time MobileNetV2 successfully classifies
a traffic sign crop.

Usage
-----
    from src.utils.firebase_sync import sync_detection

    sync_detection(
        data={"label": "Cấm rẽ", "confidence": 0.93},
        display_name="admin"
    )

The ``display_name`` value should come from the authenticated user's
``displayName`` field in the ``users`` Firestore collection.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Firebase Admin SDK — lazy singleton
# ---------------------------------------------------------------------------

_db: Any = None          # google.cloud.firestore.Client
_init_lock = threading.Lock()

# Path to the service-account key file (relative to project root)
_KEY_PATH = Path(__file__).resolve().parent.parent.parent / "keys" / "firebase_key.json"


def _get_db() -> Any:
    """Return the Firestore client, initialising it on the first call."""
    global _db

    if _db is not None:
        return _db

    with _init_lock:
        # Double-checked locking: another thread may have initialised while we waited.
        if _db is not None:
            return _db

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                # Only initialise once per process
                cred = credentials.Certificate(str(_KEY_PATH))
                firebase_admin.initialize_app(cred)

            _db = firestore.client()
            print(f"[FirebaseSync] Firestore client initialised (key: {_KEY_PATH.name})")

        except Exception as exc:
            print(f"[FirebaseSync] WARNING — Could not initialise Firebase Admin SDK: {exc}")
            _db = None
            raise

    return _db


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sync_detection(data: dict[str, Any], display_name: str = "admin") -> bool:
    """Push a detection result to the Firestore ``detections`` collection.

    Args:
        data: Dictionary with detection metadata.  Expected keys:
              ``label`` (str), ``confidence`` (float), ``class_id`` (int),
              ``box_coordinates`` (list[int]), ``is_valid`` (bool),
              ``model_used`` (str), ``image_path`` (str, optional).
        display_name: The ``displayName`` of the authenticated user.

    Returns:
        True on success, False if the operation failed (non-blocking).
    """
    try:
        from google.cloud.firestore import SERVER_TIMESTAMP  # type: ignore

        db = _get_db()
        if db is None:
            return False

        doc = {
            # ── Required fields ────────────────────────────────────────────
            "label":           data.get("label", "Unknown"),
            "confidence":      round(float(data.get("confidence", 0.0)), 4),
            "timestamp":       SERVER_TIMESTAMP,
            "performed_by":    display_name,
            # ── Model provenance ───────────────────────────────────────────
            "model_used":      data.get("model_used", "YOLOv8+MobileNetV2"),
            # ── Detection metadata ─────────────────────────────────────────
            "class_id":        int(data.get("class_id", -1)),
            "box_coordinates": data.get("box_coordinates", []),
            "is_valid":        bool(data.get("is_valid", False)),
            # ── Optional thumbnail path ────────────────────────────────────
            "image_path":      data.get("image_path", ""),
        }

        db.collection("detections").add(doc)
        print(f"[FirebaseSync] Synced: {doc['label']} ({doc['confidence']:.2%}) via {doc['model_used']}")
        return True

    except Exception as exc:
        print(f"[FirebaseSync] ERROR syncing detection: {exc}")
        return False

