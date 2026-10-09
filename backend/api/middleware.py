import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("nextsight.api")

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
