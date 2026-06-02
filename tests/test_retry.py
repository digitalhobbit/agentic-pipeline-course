import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic_ai.exceptions import ModelHTTPError

from idea_pipeline.pipeline.retry import is_retryable, run_with_retry


# --- is_retryable: raw httpx errors ---


def test_is_retryable_timeout():
    assert is_retryable(asyncio.TimeoutError()) is True


def test_is_retryable_http_429():
    response = MagicMock()
    response.status_code = 429
    assert is_retryable(httpx.HTTPStatusError("rate limited", request=MagicMock(), response=response)) is True


def test_is_retryable_http_503():
    response = MagicMock()
    response.status_code = 503
    assert is_retryable(httpx.HTTPStatusError("server error", request=MagicMock(), response=response)) is True


def test_is_retryable_http_500():
    response = MagicMock()
    response.status_code = 500
    assert is_retryable(httpx.HTTPStatusError("server error", request=MagicMock(), response=response)) is True


def test_is_retryable_http_400():
    response = MagicMock()
    response.status_code = 400
    assert is_retryable(httpx.HTTPStatusError("bad request", request=MagicMock(), response=response)) is False


def test_is_retryable_value_error():
    assert is_retryable(ValueError("some error")) is False


# --- is_retryable: Pydantic AI errors (what agent calls actually raise) ---


def _model_http_error(status_code: int) -> ModelHTTPError:
    return ModelHTTPError(status_code=status_code, model_name="gemini-2.5-pro", body=None)


def test_is_retryable_model_http_error_429():
    assert is_retryable(_model_http_error(429)) is True


def test_is_retryable_model_http_error_500():
    assert is_retryable(_model_http_error(500)) is True


def test_is_retryable_model_http_error_503():
    assert is_retryable(_model_http_error(503)) is True


def test_is_retryable_model_http_error_400():
    assert is_retryable(_model_http_error(400)) is False


# --- is_retryable: google-genai errors (what the direct TTS call raises) ---


def test_is_retryable_genai_client_error_429():
    err = genai_errors.ClientError(429, {"error": {"message": "rate limited", "code": 429}})
    assert is_retryable(err) is True


def test_is_retryable_genai_server_error_503():
    err = genai_errors.ServerError(503, {"error": {"message": "unavailable", "code": 503}})
    assert is_retryable(err) is True


def test_is_retryable_genai_client_error_400():
    err = genai_errors.ClientError(400, {"error": {"message": "bad request", "code": 400}})
    assert is_retryable(err) is False


# --- run_with_retry ---


@pytest.mark.anyio
async def test_run_with_retry_success_first_attempt():
    coro_fn = AsyncMock(return_value="ok")
    result = await run_with_retry(coro_fn)
    assert result == "ok"
    coro_fn.assert_called_once()


@pytest.mark.anyio
async def test_run_with_retry_non_retryable_raises_immediately():
    coro_fn = AsyncMock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError, match="bad input"):
        await run_with_retry(coro_fn)
    coro_fn.assert_called_once()


@pytest.mark.anyio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_run_with_retry_succeeds_on_second_attempt(mock_sleep):
    coro_fn = AsyncMock(side_effect=[asyncio.TimeoutError(), "ok"])
    result = await run_with_retry(coro_fn)
    assert result == "ok"
    assert coro_fn.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.anyio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_run_with_retry_exhausts_all_retries(mock_sleep):
    coro_fn = AsyncMock(side_effect=asyncio.TimeoutError())
    with pytest.raises(RuntimeError, match="failed after 5 attempts"):
        await run_with_retry(coro_fn, context="test call")
    assert coro_fn.call_count == 5  # 1 initial + 4 retries
    assert mock_sleep.call_count == 4


@pytest.mark.anyio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_run_with_retry_http_429_is_retried(mock_sleep):
    response = MagicMock()
    response.status_code = 429
    err = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=response)
    coro_fn = AsyncMock(side_effect=[err, "ok"])
    result = await run_with_retry(coro_fn)
    assert result == "ok"
    assert coro_fn.call_count == 2


@pytest.mark.anyio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_run_with_retry_model_http_error_429_is_retried(mock_sleep):
    err = _model_http_error(429)
    coro_fn = AsyncMock(side_effect=[err, "ok"])
    result = await run_with_retry(coro_fn)
    assert result == "ok"
    assert coro_fn.call_count == 2


