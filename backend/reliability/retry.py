"""Reliability utilities: Exponential backoff retry policies and error filtering."""

import time
import random
from typing import Callable, TypeVar, Optional, Tuple, Type

T = TypeVar("T")


class RetryPolicy:
    """Configurable retry execution policy with exponential backoff and jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_s: float = 0.05,
        backoff_multiplier: float = 2.0,
        max_backoff_s: float = 1.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ) -> None:
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_s = max_backoff_s
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def execute(self, func: Callable[[], T], retries: Optional[int] = None) -> T:
        """Execute callable with retries upon specified exceptions."""
        attempts = retries if retries is not None else self.max_retries
        current_backoff = self.initial_backoff_s

        for attempt in range(1, attempts + 1):
            try:
                return func()
            except self.retryable_exceptions as e:
                if attempt >= attempts:
                    raise RuntimeError(f"Max retries ({attempts}) exceeded: {e}") from e

                sleep_time = current_backoff
                if self.jitter:
                    sleep_time *= random.uniform(0.8, 1.2)

                time.sleep(min(sleep_time, self.max_backoff_s))
                current_backoff *= self.backoff_multiplier

        raise RuntimeError("Max retries exceeded")
