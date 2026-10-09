"""Circuit breaker pattern implementation for fault-tolerant service interactions."""

import time
from enum import Enum
from typing import Callable, TypeVar, Optional

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Service failing, requests blocked
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """Protects downstream services from cascading overload during outages."""

    def __init__(
        self,
        name: str = "service",
        failure_threshold: int = 3,
        recovery_timeout_s: float = 5.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.time()

    def can_execute(self, current_time: Optional[float] = None) -> bool:
        """Check if circuit permits request execution."""
        now = current_time if current_time is not None else time.time()

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if (now - self.last_state_change) >= self.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self, current_time: Optional[float] = None) -> None:
        """Record successful execution, resetting failure counts."""
        now = current_time if current_time is not None else time.time()
        self.failure_count = 0
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.last_state_change = now

    def record_failure(self, current_time: Optional[float] = None) -> None:
        """Record execution failure, potentially tripping the circuit."""
        now = current_time if current_time is not None else time.time()
        self.failure_count += 1

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = now
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = now

    def execute(self, func: Callable[[], T], current_time: Optional[float] = None) -> T:
        """Wrap callable execution in circuit breaker logic."""
        if not self.can_execute(current_time):
            raise RuntimeError(f"CircuitBreaker '{self.name}' is OPEN; execution blocked")

        try:
            result = func()
            self.record_success(current_time)
            return result
        except Exception:
            self.record_failure(current_time)
            raise
