"""
backend/api/v1/routes/analytics.py

Analytics endpoints returning aggregated statistics over DetectionRecord data:
- GET /api/v1/analytics/trend    — daily/monthly detection counts
- GET /api/v1/analytics/validity — valid vs invalid ratio
- GET /api/v1/analytics/frequency — top-N sign class frequencies
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.db.database import get_session
from backend.db.schemas import DetectionRecord

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TrendPoint(BaseModel):
    """Single data point in a trend series."""

    period: str   # ISO date string: "2026-04-09" (daily) or "2026-04" (monthly)
    count: int


class TrendResponse(BaseModel):
    """Trend analytics response."""

    granularity: str
    data: list[TrendPoint]


class ValidityResponse(BaseModel):
    """Valid/invalid count and ratio response."""

    valid_count: int
    invalid_count: int
    total: int
    valid_ratio: float


class FrequencyPoint(BaseModel):
    """Frequency entry for a single sign class."""

    class_id: int
    class_name: str
    count: int


class FrequencyResponse(BaseModel):
    """Sign class frequency analytics response."""

    top_n: int
    data: list[FrequencyPoint]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.get(
    "/trend",
    response_model=TrendResponse,
    status_code=status.HTTP_200_OK,
    summary="Detection counts aggregated by day or month.",
)
async def get_trend(
    granularity: Literal["daily", "monthly"] = Query(
        default="daily", description="Aggregation granularity."
    ),
    days: int = Query(
        default=30, ge=1, le=365, description="Lookback window in days."
    ),
    session: AsyncSession = Depends(get_session),
) -> TrendResponse:
    """Return detection counts grouped by day or month within the lookback window.

    Args:
        granularity: ``"daily"`` or ``"monthly"`` aggregation.
        days: Number of past days to include.
        session: Injected async database session.

    Returns:
        ``TrendResponse`` with a list of (period, count) data points.
    """
    cutoff: datetime = datetime.utcnow() - timedelta(days=days)
    result = await session.exec(
        select(DetectionRecord.timestamp).where(DetectionRecord.timestamp >= cutoff)
    )
    timestamps: list[datetime] = result.all()

    counter: Counter[str] = Counter()
    for ts in timestamps:
        if granularity == "daily":
            key = ts.strftime("%Y-%m-%d")
        else:
            key = ts.strftime("%Y-%m")
        counter[key] += 1

    sorted_points = [
        TrendPoint(period=period, count=count)
        for period, count in sorted(counter.items())
    ]

    return TrendResponse(granularity=granularity, data=sorted_points)


@router.get(
    "/validity",
    response_model=ValidityResponse,
    status_code=status.HTTP_200_OK,
    summary="Ratio of valid vs invalid detections.",
)
async def get_validity(
    days: Optional[int] = Query(
        default=None, ge=1, description="Lookback window in days (omit for all-time)."
    ),
    session: AsyncSession = Depends(get_session),
) -> ValidityResponse:
    """Return valid and invalid detection counts and their ratio.

    Args:
        days: Optional lookback window. If None, queries all records.
        session: Injected async database session.

    Returns:
        ``ValidityResponse`` with counts and the valid ratio.
    """
    query = select(DetectionRecord.is_valid)
    if days is not None:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.where(DetectionRecord.timestamp >= cutoff)

    result = await session.exec(query)
    values: list[bool] = result.all()

    valid_count = sum(1 for v in values if v)
    invalid_count = len(values) - valid_count
    total = len(values)
    valid_ratio = round(valid_count / total, 4) if total > 0 else 0.0

    return ValidityResponse(
        valid_count=valid_count,
        invalid_count=invalid_count,
        total=total,
        valid_ratio=valid_ratio,
    )


@router.get(
    "/frequency",
    response_model=FrequencyResponse,
    status_code=status.HTTP_200_OK,
    summary="Top-N most frequently detected sign classes.",
)
async def get_frequency(
    top_n: int = Query(default=10, ge=1, le=50, description="Number of top classes."),
    days: Optional[int] = Query(
        default=None, ge=1, description="Lookback window in days (omit for all-time)."
    ),
    session: AsyncSession = Depends(get_session),
) -> FrequencyResponse:
    """Return the most frequently detected sign categories.

    Args:
        top_n: Max number of classes to return.
        days: Optional lookback window.
        session: Injected async database session.

    Returns:
        ``FrequencyResponse`` with class_id, class_name, and count.
    """
    query = select(DetectionRecord.class_id, DetectionRecord.class_name)
    if days is not None:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.where(DetectionRecord.timestamp >= cutoff)

    result = await session.exec(query)
    rows: list[tuple[int, str]] = result.all()

    counter: Counter[tuple[int, str]] = Counter(rows)
    top = counter.most_common(top_n)

    return FrequencyResponse(
        top_n=top_n,
        data=[
            FrequencyPoint(class_id=cid, class_name=cname, count=cnt)
            for (cid, cname), cnt in top
        ],
    )
