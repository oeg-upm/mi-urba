"""
LLM API client with retry logic.

Supports any OpenAI-compatible endpoint (Ollama cloud, llama-swap, etc.).
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
        max_retries: int = 8,
        retry_delay: float = 5.0,
        api_key: str | None = None,
        extra_params: dict | None = None,
        json_schema: dict | None = None,
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
        self.extra_params = extra_params or {}
        self.json_schema = json_schema
        self._stats = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0}

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
            "temperature": temperature or self.temperature,
        }
        # Merge extra params (e.g. enable_thinking=false, chat_template_kwargs)
        payload.update(self.extra_params)

        # Constrained decoding: force output to conform to a JSON schema
        if self.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": self.json_schema},
            }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
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
                last_error = e
                self._stats["errors"] += 1
                # Connection errors (server down/unreachable) get generous backoff
                # to survive inference-server crashes that take minutes to recover.
                is_conn_error = isinstance(e, (
                    requests.ConnectionError,
                    requests.Timeout,
                ))
                logger.warning(
                    f"API call failed (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    if is_conn_error:
                        # Generous exponential backoff for server crashes:
                        # 5s, 10s, 20s, 40s, 80s, 120s, 240s  (~10 min total)
                        wait = min(self.retry_delay * (2 ** (attempt - 1)), 240)
                    else:
                        # Linear backoff for HTTP errors / bad JSON
                        wait = self.retry_delay * attempt
                    logger.info(f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)

        raise RuntimeError(
            f"API call failed after {self.max_retries} attempts: {last_error}"
        )

    def extract_content(self, response: dict) -> str:
        """Extract the assistant's text content from a chat response."""
        return response["choices"][0]["message"]["content"].strip()

    def extract_json(self, response: dict) -> dict:
        """Extract and parse JSON from the assistant's response content."""
        content = self.extract_content(response)
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)
        return json.loads(content)