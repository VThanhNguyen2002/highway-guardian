"""
backend/api/v1/routes/detect.py

POST /api/v1/detect — accepts an image file and returns detection results.
Each detected sign is persisted to the database as a DetectionRecord.

Security hardening:
  - File size cap enforced before disk write (prevents DoS via large uploads).
  - Extension whitelist enforced independently of user-supplied filename.
  - Filename is always a UUID — user-supplied filename is discarded entirely.
  - Inference runs in a thread pool worker to avoid blocking the async event loop.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import io
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.config.settings import Settings, get_settings
from backend.core.inference_pipeline import DetectionResult, TrafficSignDetector
from backend.core.task_manager import create_task, update_task_status, get_task_status, TaskStatus
from backend.db.database import get_session, engine
from backend.db.schemas import DetectionRecord

router = APIRouter(prefix="/detect", tags=["Detection"])

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

# Exhaustive whitelist of accepted MIME types.
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/jpg",
    "image/png",
})

# Extension whitelist derived independently of user-supplied filename.
# Only these extensions are written to disk, regardless of what content_type claims.
_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DetectionResultResponse(BaseModel):
    """API response shape for a single detected sign."""

    box_coordinates: Optional[list[int]] = None
    confidence: float
    class_id: int
    class_name: str
    is_valid: bool


class DetectTaskResponse(BaseModel):
    """Initial response returning the background task info."""
    task_id: str
    status: str

class DetectTaskStatusResponse(DetectTaskResponse):
    """Response when polling for task completion."""
    result: Optional[list[DetectionResultResponse]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------

def get_detector() -> TrafficSignDetector:
    """FastAPI dependency: return the application-scoped TrafficSignDetector.

    Returns:
        The singleton TrafficSignDetector loaded during lifespan startup.
    """
    from backend.main import get_app_state
    return get_app_state().detector


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=DetectTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run two-stage traffic sign detection on an uploaded image.",
)
async def detect(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG or PNG image to analyse.")],
    background_tasks: BackgroundTasks,
    confidence_threshold: Annotated[
        float,
        Form(ge=0.0, le=1.0, description="Minimum YOLO detection confidence."),
    ] = 0.25,
    mode: Annotated[str, Form(description="Detection mode: 'yolo', 'cnn', or 'ensemble'")] = "yolo",
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    detector: TrafficSignDetector = Depends(get_detector),
) -> DetectTaskResponse:
    """Detect and classify traffic signs in the uploaded image.

    Security measures applied in this order:
    1. MIME type check against ``_ALLOWED_CONTENT_TYPES`` whitelist.
    2. File size check: rejects uploads exceeding ``settings.max_upload_size_mb``.
       Size is read from the ``Content-Length`` header first; if absent the body
       is streamed and counted, aborting early on overflow.
    3. Extension is resolved from the validated MIME type — the user-supplied
       filename is completely discarded. The saved filename is always a UUID.

    Performance:
    - The synchronous CPU-bound inference call is delegated to a thread pool
      via ``run_in_threadpool`` to prevent blocking the async event loop.

    Args:
        request: Raw Starlette request (used for Content-Length header).
        file: Multipart image upload (JPEG/PNG).
        confidence_threshold: Minimum YOLO confidence for a detection to be kept.
        settings: Injected application settings.
        session: Injected async database session.
        detector: Injected TrafficSignDetector singleton.

    Returns:
        ``DetectResponse`` containing predictions and persisted image path.

    Raises:
        HTTPException 413: If the upload exceeds the configured size limit.
        HTTPException 415: If the MIME type is not in the allowed whitelist.
        HTTPException 500: If inference or file I/O fails unexpectedly.
    """
    # ── 1. MIME type whitelist check ─────────────────────────────────────────
    content_type: str = file.content_type or ""
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{content_type}'. "
                "Accepted: image/jpeg, image/png."
            ),
        )

    # ── 2. File size enforcement ──────────────────────────────────────────────
    max_bytes: int = settings.max_upload_size_mb * 1024 * 1024

    # Check Content-Length header first (fast path — no body read needed).
    content_length_header: str | None = request.headers.get("content-length")
    if content_length_header is not None:
        try:
            declared_size = int(content_length_header)
        except ValueError:
            declared_size = 0
        if declared_size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Upload size {declared_size // (1024 * 1024)} MiB exceeds "
                    f"the {settings.max_upload_size_mb} MiB limit."
                ),
            )

    # Stream-read the body, enforcing the cap regardless of Content-Length.
    chunks: list[bytes] = []
    bytes_read: int = 0
    chunk_size: int = 1024 * 64  # 64 KiB per chunk

    while True:
        chunk: bytes = await file.read(chunk_size)
        if not chunk:
            break
        bytes_read += len(chunk)
        if bytes_read > max_bytes:
            await file.close()
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Upload body exceeds the {settings.max_upload_size_mb} MiB limit."
                ),
            )
        chunks.append(chunk)

    file_bytes: bytes = b"".join(chunks)

    # ── 3. Persist upload with a UUID filename (user filename discarded) ──────
    # Extension is derived from the validated MIME type — never from user input.
    safe_extension: str = _CONTENT_TYPE_TO_EXT[content_type]
    filename: str = f"{uuid.uuid4().hex}{safe_extension}"
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    save_path: Path = settings.uploads_dir / filename

    background_tasks.add_task(save_path.write_bytes, file_bytes)
    await file.close()

    relative_image_path: str = str(save_path.relative_to(settings.uploads_dir.parent))
    
    # Generate task ID and enqueue task
    task_id = str(uuid.uuid4())
    create_task(task_id)

    background_tasks.add_task(
        _background_inference_task,
        task_id,
        file_bytes,
        relative_image_path,
        mode,
        confidence_threshold,
        detector
    )

    return DetectTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING
    )


async def _background_inference_task(
    task_id: str,
    file_bytes: bytes,
    relative_image_path: str,
    mode: str,
    confidence_threshold: float,
    detector: TrafficSignDetector,
) -> None:
    update_task_status(task_id, TaskStatus.PROCESSING)
    from PIL import Image as PILImage

    try:
        def _run_inference() -> list[DetectionResult]:
            with PILImage.open(io.BytesIO(file_bytes)) as img:
                if mode == "cnn":
                    return [detector.classify_cnn(img)]
                elif mode == "ensemble":
                    return detector.detect_ensemble(img)
                else:
                    return detector.detect_yolo(img, confidence_threshold=confidence_threshold)

        results: list[DetectionResult] = await run_in_threadpool(_run_inference)

        # ── Persist detection records ──────────────────────────────────────────
        async with AsyncSession(engine, expire_on_commit=False) as session:
            now: datetime = datetime.utcnow()
            for result in results:
                record = DetectionRecord(
                    timestamp=now,
                    image_path=relative_image_path,
                    class_id=result.class_id,
                    class_name=result.class_name,
                    confidence=result.confidence,
                    is_valid=result.is_valid,
                )
                session.add(record)
            await session.commit()

        update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            result=[
                {
                    "box_coordinates": r.box_coordinates,
                    "confidence": r.confidence,
                    "class_id": r.class_id,
                    "class_name": r.class_name,
                    "is_valid": r.is_valid,
                }
                for r in results
            ],
        )

    except Exception as exc:
        update_task_status(task_id, TaskStatus.FAILED, error=str(exc))


@router.get(
    "/{task_id}",
    response_model=DetectTaskStatusResponse,
    summary="Poll for detection task status.",
)
async def get_detect_task_status(task_id: str) -> DetectTaskStatusResponse:
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )
    return DetectTaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error"),
    )
