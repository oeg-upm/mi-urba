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

# Structured-output modes a client may declare (models.<name>.structured_output):
#   None          — free generation, no response_format sent
#   "json_object" — response_format json_object: valid JSON, no schema
#   "json_schema" — response_format json_schema: server enforces the schema
VALID_STRUCTURED_MODES = (None, "json_object", "json_schema")


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
        structured_output: str | None = None,
        json_schema: dict | None = None,
    ):
        if structured_output not in VALID_STRUCTURED_MODES:
            raise ValueError(
                f"structured_output must be one of {VALID_STRUCTURED_MODES}, "
                f"got {structured_output!r}"
            )
        if structured_output == "json_schema" and json_schema is None:
            raise ValueError("structured_output='json_schema' requires json_schema")
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
        self.structured_output = structured_output
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

        # Structured output (mode declared per model in config)
        if self.structured_output == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.structured_output == "json_schema":
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
                logger.warning(
                    f"API call failed (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
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