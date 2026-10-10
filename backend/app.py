import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI

from backend.config.settings import AppSettings, get_settings
from backend.api.routes import router as api_router
from backend.api.conversation_routes import router as conversation_router
from backend.api.middleware import RequestCorrelationMiddleware
from backend.services.vision_service import get_vision_service
from backend.services.conversation_service import get_conversation_service

def setup_logging(log_level: str = "INFO") -> None:
    """Configures structured application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Predictable application lifecycle manager.
    Initializes software pipelines at startup and performs clean teardown at shutdown.
    Requires no physical hardware (camera/mic/GPU) to start.
    """
    settings: AppSettings = app.state.settings
    setup_logging(settings.log_level)
    logger = logging.getLogger("nextsight.lifecycle")
    
    logger.info(f"Starting {settings.app_name} v{settings.version} in '{settings.environment}' mode")
    logger.info("Initializing software vision service and AI engine registry...")
    # Pre-initialize vision service
    service = getattr(app.state, "vision_service", None) or get_vision_service(settings=settings)
    readiness = service.check_readiness()
    logger.info(f"Subsystem readiness check: {readiness}")
    logger.info("NextSight AI Backend started successfully on software runtime.")

    yield # Server runs here

    logger.info("Shutting down NextSight AI Backend...")
    logger.info("Teardown complete.")


def create_app(settings: Optional[AppSettings] = None) -> FastAPI:
    """
    Application factory creating configured FastAPI app.
    Side-effect free on import; instantiates cleanly in test harnesses.
    """
    cfg = settings or get_settings()
    app = FastAPI(
        title=cfg.app_name,
        version=cfg.version,
        description="NextSight Smart Glasses Python AI Backend Foundation (Phase 5)",
        lifespan=lifespan
    )
    app.state.settings = cfg
    from backend.power.policy import get_power_policy_manager
    app.state.power_manager = get_power_policy_manager(require_verified_telemetry=cfg.is_strict_telemetry_required)
    app.state.vision_service = get_vision_service(settings=cfg, power_manager=app.state.power_manager)

    # Add middleware
    app.add_middleware(RequestCorrelationMiddleware)

    # Include routes
    app.include_router(api_router)
    app.include_router(conversation_router)

    return app

# Default application instance for Uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "backend.app:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower()
    )
