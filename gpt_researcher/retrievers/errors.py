"""Error classification and retry policy for research retrieval."""

import asyncio
import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
import requests


_MESSAGES = {
    "authentication": "Search credentials were rejected or are missing.",
    "invalid_request": "The search provider rejected the request.",
    "invalid_response": "The search provider returned an invalid response.",
    "rate_limited": "The search provider rate-limited the request.",
    "timeout": "The search request timed out.",
    "connection": "The search provider could not be reached.",
    "unavailable": "The search provider was temporarily unavailable.",
    "configuration": "The search provider is not configured correctly.",
    "tool_error": "The MCP tool reported a failure.",
    "unexpected": "The retrieval step failed.",
}
_TRANSIENT = {"timeout", "connection", "unavailable", "rate_limited"}
_MAX_RETRY_DELAY = 10
logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    def __init__(self, provider: str, code: str, *, retry_after: float | None = None):
        self.provider = provider
        self.code = code
        self.retry_after = retry_after
        self.attempts = 1
        super().__init__(_MESSAGES[code])


class InsufficientSearchEvidence(RuntimeError):
    def __init__(self, search_errors):
        self.search_errors = search_errors
        super().__init__("Insufficient evidence. The research retrieved no usable material.")


def _retry_after_seconds(value):
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


def classify_search_error(provider: str, error: Exception) -> SearchError:
    if isinstance(error, SearchError):
        return error
    if isinstance(error, ExceptionGroup):
        errors = [classify_search_error(provider, child) for child in error.exceptions]
        if all(item.code in _TRANSIENT for item in errors):
            rate_limits = [item for item in errors if item.code == "rate_limited"]
            if rate_limits:
                delay = max((item.retry_after or 0 for item in rate_limits), default=0)
                return SearchError(provider, "rate_limited", retry_after=delay)
            return errors[0]
        return next(item for item in errors if item.code not in _TRANSIENT)
    if isinstance(error, (TimeoutError, requests.Timeout, httpx.TimeoutException)):
        return SearchError(provider, "timeout")
    if isinstance(error, (ConnectionError, requests.ConnectionError, httpx.NetworkError)):
        return SearchError(provider, "connection")
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status in (401, 403):
        return SearchError(provider, "authentication")
    if status == 429:
        delay = _retry_after_seconds(response.headers.get("Retry-After"))
        return SearchError(provider, "rate_limited", retry_after=delay)
    if status == 408:
        return SearchError(provider, "timeout")
    if isinstance(status, int) and 500 <= status < 600:
        return SearchError(provider, "unavailable")
    if isinstance(status, int) and 400 <= status < 500:
        return SearchError(provider, "invalid_request")
    return SearchError(provider, "unexpected")


async def run_search_call(provider, invoke, *, max_attempts=3):
    """Retry a read operation within a bounded number of attempts and wait."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await invoke()
        except Exception as exc:
            error = classify_search_error(provider, exc)
            error.attempts = max(error.attempts, attempt)
            delay = max(2 ** (attempt - 1), error.retry_after or 0)
            if attempt == max_attempts or error.code not in _TRANSIENT or delay > _MAX_RETRY_DELAY:
                if error is exc:
                    raise
                raise error from exc
            logger.info("Retrying %s search after %s (%s/%s)", provider, error.code, attempt, max_attempts)
            await asyncio.sleep(delay)


def format_search_limitations(errors):
    if not errors:
        return ""
    unavailable = dict.fromkeys((error["provider"], error["message"]) for error in errors)
    lines = [f"- **{provider}** — {message}" for provider, message in unavailable]
    return "\n\n## Search coverage limitations\n\n" + "\n".join(lines) + "\n"
