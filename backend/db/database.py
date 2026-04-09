"""
backend/db/database.py

Async SQLModel engine and session factory.
Supports SQLite (development) and PostgreSQL (production) via DATABASE_URL.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.config.settings import get_settings

_settings = get_settings()

# Single engine instance for the process lifetime.
engine = create_async_engine(
    str(_settings.database_url),
    echo=_settings.debug,
    # SQLite-specific: required for async writes from multiple coroutines.
    connect_args={"check_same_thread": False}
    if "sqlite" in str(_settings.database_url)
    else {},
)


async def create_db_and_tables() -> None:
    """Create all SQLModel tables if they do not already exist.

    Called once during application lifespan startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for use as a FastAPI dependency.

    Yields:
        An open AsyncSession that is automatically committed and closed.
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
