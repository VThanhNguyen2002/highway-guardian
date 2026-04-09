"""
backend/main.py

FastAPI application factory and lifespan manager.

Startup sequence:
1. Initialise uploads directory.
2. Load YOLO and CNN models into the module-level cache.
3. Parse QCVN 41:2019 PDF and build the rule validator.
4. Create database tables if they do not exist.

All heavy resources are stored in ``app.state`` for dependency injection.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1.router import api_router
from backend.config.settings import Settings, get_settings
from backend.core.inference_pipeline import TrafficSignDetector
from backend.core.model_loader import load_models
from backend.core.pdf_parser import PDFRuleParser
from backend.db.database import create_db_and_tables


@dataclass
class AppState:
    """Container for application-scoped singleton resources.

    Attributes:
        detector: Fully initialised two-stage inference pipeline.
        pdf_parser: Pre-parsed QCVN 41:2019 PDF rule validator.
    """

    detector: TrafficSignDetector
    pdf_parser: PDFRuleParser


# Module-level reference populated during lifespan startup.
_app_state: AppState | None = None


def get_app_state() -> AppState:
    """Return the application-scoped state container.

    Returns:
        Populated ``AppState`` instance.

    Raises:
        RuntimeError: If called before the lifespan startup has completed.
    """
    if _app_state is None:
        raise RuntimeError("AppState not initialised. Application is not running.")
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Startup:
        - Initialises uploads directory.
        - Loads ML models (YOLO + CNN) into cache.
        - Parses PDF rule document.
        - Creates SQLModel database tables.

    Shutdown:
        - Emits a shutdown log message (no stateful teardown required).

    Args:
        app: The FastAPI application instance.
    """
    global _app_state

    settings: Settings = get_settings()

    print("[Startup] Initialising Highway Guardian API v2...")

    # Ensure uploads directory exists.
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    # Load ML models.
    print("[Startup] Loading ML models...")
    load_models(settings)

    # Parse PDF rule document.
    print("[Startup] Parsing QCVN 41:2019 PDF...")
    pdf_parser = PDFRuleParser(settings.pdf_path)

    # Build detector.
    detector = TrafficSignDetector(settings=settings, pdf_parser=pdf_parser)

    # Create database tables.
    print("[Startup] Initialising database...")
    await create_db_and_tables()

    _app_state = AppState(detector=detector, pdf_parser=pdf_parser)
    print("[Startup] All systems ready.")

    yield

    print("[Shutdown] Highway Guardian API shutting down.")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Configured ``FastAPI`` instance with CORS middleware and API routes.
    """
    settings: Settings = get_settings()

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        """Return a simple liveness probe response."""
        return {"status": "ok", "version": settings.app_version}

    return app


app: FastAPI = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=_settings.debug,
    )
