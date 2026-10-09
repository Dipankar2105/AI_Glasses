"""Firmware reliability interface for smart glasses."""

from backend.reliability.retry import RetryPolicy as BackendRetryPolicy


class RetryPolicy:
    """Firmware retry policy wrapper."""

    def __init__(self, max_retries: int = 3, initial_backoff_s: float = 0.01):
        self._policy = BackendRetryPolicy(
            max_retries=max_retries,
            initial_backoff_s=initial_backoff_s,
            jitter=False,
        )

    def execute(self, func, retries: int = None):
        return self._policy.execute(func, retries=retries)
