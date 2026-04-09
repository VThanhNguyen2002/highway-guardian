"""
backend/api/v1/router.py

Aggregates all v1 route modules under a single APIRouter.
Mounted in main.py under the settings.api_prefix (/api/v1).
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.v1.routes import analytics, detect, history

api_router = APIRouter()

api_router.include_router(detect.router)
api_router.include_router(history.router)
api_router.include_router(analytics.router)
