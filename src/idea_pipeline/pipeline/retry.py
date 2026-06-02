import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import httpx
from google.genai import errors as genai_errors
from pydantic_ai.exceptions import ModelHTTPError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TIMEOUT_SECONDS = 300  # 5 minutes per attempt
_MAX_RETRIES = 4
_BASE_DELAY = 5.0
_MAX_DELAY = 120.0
_JITTER_FACTOR = 0.25


def _status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from the exception types
    our SDKs actually raise.

    Each layer raises its own exception type rather than a raw
    ``httpx.HTTPStatusError``: Pydantic AI raises ``ModelHTTPError`` and the
    google-genai SDK (used directly for TTS) raises ``APIError`` subclasses.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, ModelHTTPError):
        return exc.status_code
    if isinstance(exc, genai_errors.APIError):
        return exc.code
    return None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True
    status_code = _status_code(exc)
    if status_code is not None:
        return status_code == 429 or status_code >= 500
    return False


async def run_with_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    context: str = "agent call",
) -> T:
    last_exc: BaseException | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await asyncio.wait_for(coro_fn(), timeout=_TIMEOUT_SECONDS)
        except BaseException as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
            jitter = delay * _JITTER_FACTOR * random.uniform(-1, 1)
            sleep_time = delay + jitter
            logger.warning(
                f"Retryable error on {context} (attempt {attempt + 1}/{_MAX_RETRIES + 1}): "
                f"{type(exc).__name__}: {exc}. Retrying in {sleep_time:.1f}s"
            )
            await asyncio.sleep(sleep_time)

    raise RuntimeError(
        f"{context} failed after {_MAX_RETRIES + 1} attempts"
    ) from last_exc
