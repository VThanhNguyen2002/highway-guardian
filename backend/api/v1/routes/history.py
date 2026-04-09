"""
backend/api/v1/routes/history.py

GET /api/v1/history — paginated query over DetectionRecord table.
Supports optional date-range and is_valid filtering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.db.database import get_session
from backend.db.schemas import DetectionRecord, DetectionRecordRead

router = APIRouter(prefix="/history", tags=["History"])


class HistoryResponse(BaseModel):
    """Paginated history response envelope."""

    total_returned: int
    limit: int
    offset: int
    records: list[DetectionRecordRead]


@router.get(
    "",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve paginated detection history.",
)
async def get_history(
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return."),
    offset: int = Query(default=0, ge=0, description="Number of records to skip."),
    start_date: Optional[datetime] = Query(
        default=None, description="Filter records on or after this UTC datetime."
    ),
    end_date: Optional[datetime] = Query(
        default=None, description="Filter records on or before this UTC datetime."
    ),
    is_valid: Optional[bool] = Query(
        default=None, description="Filter by validity status."
    ),
    session: AsyncSession = Depends(get_session),
) -> HistoryResponse:
    """Return a paginated list of detection history records.

    Args:
        limit: Maximum number of records per page.
        offset: Record offset for pagination.
        start_date: Optional UTC lower-bound for ``timestamp`` filter.
        end_date: Optional UTC upper-bound for ``timestamp`` filter.
        is_valid: Optional boolean filter on the ``is_valid`` column.
        session: Injected async database session.

    Returns:
        ``HistoryResponse`` with the matching records and pagination metadata.
    """
    query = select(DetectionRecord).order_by(DetectionRecord.timestamp.desc())  # type: ignore[arg-type]

    if start_date is not None:
        query = query.where(DetectionRecord.timestamp >= start_date)
    if end_date is not None:
        query = query.where(DetectionRecord.timestamp <= end_date)
    if is_valid is not None:
        query = query.where(DetectionRecord.is_valid == is_valid)

    query = query.offset(offset).limit(limit)
    result = await session.exec(query)
    records: list[DetectionRecord] = result.all()

    return HistoryResponse(
        total_returned=len(records),
        limit=limit,
        offset=offset,
        records=[DetectionRecordRead.model_validate(r) for r in records],
    )
