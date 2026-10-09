"""Token bucket rate limiter for throttling sensor events and API bursts."""

import time
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter with floating point tolerance."""

    def __init__(self, rate_per_second: float = 10.0, burst_capacity: int = 20) -> None:
        self.rate_per_second = rate_per_second
        self.burst_capacity = burst_capacity
        self.tokens = float(burst_capacity)
        self.last_update = time.time()

    def allow_request(self, tokens_requested: float = 1.0, current_time: Optional[float] = None) -> bool:
        """Check if request can proceed under current token allocation."""
        now = current_time if current_time is not None else time.time()
        elapsed = max(0.0, now - self.last_update)
        self.last_update = now

        # Replenish tokens
        self.tokens = min(float(self.burst_capacity), self.tokens + elapsed * self.rate_per_second)

        # Allow with 1e-7 float epsilon margin
        if self.tokens >= (tokens_requested - 1e-7):
            self.tokens = max(0.0, self.tokens - tokens_requested)
            return True
        return False
