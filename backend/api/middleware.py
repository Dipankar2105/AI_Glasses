import time
import uuid
import logging
import threading
from collections import defaultdict
from typing import Callable, Dict, List, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("nextsight.api")

# Paths that are public and never require authentication
PUBLIC_PATH_PREFIXES = (
    "/health",
    "/ready",
    "/api/v1/capabilities",
    "/docs",
    "/redoc",
    "/openapi.json"
)


class InMemoryRateLimiter:
    """Thread-safe sliding-window rate limiter per client IP."""
    def __init__(self, limit_per_minute: int = 300):
        self.limit_per_minute = limit_per_minute
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        window_start = now - 60.0
        with self._lock:
            timestamps = self._history[client_id]
            # Prune timestamps older than 60 seconds
            valid_timestamps = [t for t in timestamps if t > window_start]
            if len(valid_timestamps) >= self.limit_per_minute:
                self._history[client_id] = valid_timestamps
                return False
            valid_timestamps.append(now)
            self._history[client_id] = valid_timestamps
            return True

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware attaching a unique X-Request-ID correlation header to every request,
    measuring response duration, and logging structured request metadata without
    exposing raw image or credentials data.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id

        start_time = time.time()
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000.0
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
            
            logger.info(
                f"[{req_id}] {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration_ms:.2f}ms"
            )
            return response
        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000.0
            logger.error(
                f"[{req_id}] {request.method} {request.url.path} "
                f"FAILED error={str(exc)} duration={duration_ms:.2f}ms"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_SERVER_ERROR",
                    "detail": "An unexpected error occurred processing your request",
                    "request_id": req_id
                },
                headers={"X-Request-ID": req_id}
            )


class SecurityAndRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Security gate enforcing:
    1. Maximum request payload size bounds (Content-Length header check).
    2. Client API Key authentication when configured (X-API-Key or Bearer token).
    3. In-memory sliding-window rate limiting.
    """
    def __init__(self, app, rate_limiter: Optional[InMemoryRateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or InMemoryRateLimiter(limit_per_minute=300)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = getattr(request.app.state, "settings", None)
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        client_ip = request.client.host if request.client else "unknown"

        # 1. Payload size check via Content-Length header
        max_bytes = settings.max_request_body_bytes if settings else 15 * 1024 * 1024
        content_length_str = request.headers.get("content-length")
        if content_length_str:
            try:
                content_length = int(content_length_str)
                if content_length > max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "PAYLOAD_TOO_LARGE",
                            "detail": f"Request size ({content_length} bytes) exceeds limit of {max_bytes} bytes",
                            "request_id": req_id
                        },
                        headers={"X-Request-ID": req_id}
                    )
            except ValueError:
                pass

        # 2. Rate limiting check
        if settings and settings.rate_limit_per_minute:
            self.rate_limiter.limit_per_minute = settings.rate_limit_per_minute

        if not self.rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "detail": "Too many requests. Please retry in a few seconds.",
                    "request_id": req_id
                },
                headers={"X-Request-ID": req_id, "Retry-After": "10"}
            )

        # 3. Authentication check (if API key is configured on server)
        configured_key = settings.api_key if settings else None
        if configured_key and configured_key.strip():
            path = request.url.path
            is_public = any(path.startswith(p) for p in PUBLIC_PATH_PREFIXES)
            if not is_public:
                provided_key = request.headers.get("X-API-Key")
                if not provided_key:
                    auth_hdr = request.headers.get("Authorization", "")
                    if auth_hdr.startswith("Bearer "):
                        provided_key = auth_hdr[7:].strip()

                if not provided_key or provided_key != configured_key:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": "UNAUTHORIZED",
                            "detail": "Invalid or missing API key",
                            "request_id": req_id
                        },
                        headers={"X-Request-ID": req_id}
                    )

        return await call_next(request)
