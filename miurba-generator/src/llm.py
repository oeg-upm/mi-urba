"""
LLM API client with retry logic.

Supports any OpenAI-compatible endpoint.
Automatically normalizes base_url to avoid double /v1 paths.
Does NOT impose artificial token limits — the model decides when to stop.
"""

import json
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "kimi-k2.6:cloud",
        temperature: float = 0.85,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        api_key: str | None = None,
        model_load_wait: int | None = None,
        connection_backoff: bool = False,
        extra_params: dict | None = None,
    ):
        # Normalize: strip trailing slash and /v1 suffix
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self.base_url = base
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key
        # Optional patience for cold models: 5xx within this window (seconds)
        # are treated as "model loading" and do NOT consume retry attempts.
        # None = disabled (any 5xx counts as a normal retry).
        self.model_load_wait = model_load_wait
        # Optional generous backoff for server crashes: connection errors
        # (no HTTP response) wait max(retry_delay, 20s) * attempt instead of
        # the standard retry_delay * attempt.
        self.connection_backoff = connection_backoff
        # Provider-specific payload keys passed verbatim (enable_thinking,
        # chat_template_kwargs, ...). Merged into every request payload.
        self.extra_params = extra_params or {}
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if model_load_wait is not None and model_load_wait < 0:
            raise ValueError(f"model_load_wait must be >= 0 or None, got {model_load_wait}")
        self._stats = {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                       "errors": 0, "load_waits": 0}

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        No max_tokens limit is set — the model generates until natural stop.
        This avoids truncation issues with thinking/reasoning models.
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        # Extra provider-specific params (verbatim merge)
        payload.update(self.extra_params)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        attempt = 0
        load_deadline = time.time() + self.model_load_wait if self.model_load_wait else None
        while attempt < self.max_retries:
            try:
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                self._stats["calls"] += 1
                usage = data.get("usage", {})
                self._stats["tokens_in"] += usage.get("prompt_tokens", 0)
                self._stats["tokens_out"] += usage.get("completion_tokens", 0)

                return data

            except (requests.RequestException, json.JSONDecodeError) as e:
                # 5xx within the model-load window: the swapping gateway is
                # still loading the model — poll without consuming retries.
                is_load_error = (
                    load_deadline is not None
                    and isinstance(e, requests.HTTPError)
                    and e.response is not None
                    and e.response.status_code >= 500
                    and time.time() < load_deadline
                )
                if is_load_error:
                    self._stats["load_waits"] += 1
                    logger.info("Model loading (5xx within model_load_wait window) — waiting 15s...")
                    time.sleep(15)
                    continue

                attempt += 1
                last_error = e
                self._stats["errors"] += 1
                logger.warning(
                    f"API call failed (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    is_connection_error = isinstance(
                        e, (requests.ConnectionError, requests.Timeout)
                    )
                    if self.connection_backoff and is_connection_error:
                        wait = max(self.retry_delay, 20.0) * attempt
                    else:
                        wait = self.retry_delay * attempt
                    logger.info(f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)

        raise RuntimeError(
            f"API call failed after {attempt or self.max_retries} attempts: {last_error}"
        )

    def extract_content(self, response: dict) -> str:
        """Extract the assistant's text content from a chat response.

        Returns "" on any unexpected structure (missing choices, null
        content, reasoning-model payloads) instead of raising: callers
        already treat empty content as a failed record.
        """
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Unexpected response structure: {str(response)[:200]}")
            return ""
        return (content or "").strip()

    def extract_json(self, response: dict) -> dict:
        """Extract and parse JSON from the assistant's response content.

        strict=False tolerates raw control characters (e.g. newlines)
        inside JSON strings, which small models emit sometimes.
        """
        content = self.extract_content(response)
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)
        return json.loads(content, strict=False)