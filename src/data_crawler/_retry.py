import functools
import logging
import time

logger = logging.getLogger(__name__)


def with_retry(max_attempts: int = 5, backoff_base: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator: retry on exception with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts - 1:
                        raise
                    wait = backoff_base ** attempt
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.0fs...",
                        func.__name__, attempt + 1, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
        return wrapper
    return decorator
