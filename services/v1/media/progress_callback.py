"""Progress callback helper: POST phase and optional progress to transcribe-ui."""
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_LAST_CALL: dict = {"time": 0.0, "phase": None}
_THROTTLE_SECONDS = 1.0


def report_progress(
    callback_url: Optional[str],
    phase: str,
    current_seconds: Optional[float] = None,
    total_seconds: Optional[float] = None,
    progress_pct: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """POST progress to callback_url. No-op if callback_url is empty. Throttled to ~1s per phase."""
    if not callback_url or not callback_url.strip():
        return
    now = time.monotonic()
    last_time = _LAST_CALL.get("time", 0)
    last_phase = _LAST_CALL.get("phase")
    if last_phase == phase and (now - last_time) < _THROTTLE_SECONDS:
        return
    _LAST_CALL["time"] = now
    _LAST_CALL["phase"] = phase

    body = {"phase": phase}
    if current_seconds is not None:
        body["current_seconds"] = current_seconds
    if total_seconds is not None:
        body["total_seconds"] = total_seconds
    if progress_pct is not None:
        body["progress_pct"] = progress_pct
    if message is not None:
        body["message"] = message

    try:
        resp = requests.post(callback_url, json=body, timeout=5)
        if resp.status_code >= 400:
            logger.warning("Progress callback %s returned %s", callback_url, resp.status_code)
    except Exception as e:
        logger.warning("Progress callback failed: %s", e)
