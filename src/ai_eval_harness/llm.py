"""The smallest interface a judge needs from a model, plus a Gemini implementation. The harness itself needs no model."""

from __future__ import annotations

import os
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-2.5-flash"


class JudgeError(RuntimeError):
    pass


class JudgeClient(Protocol):
    name: str

    def complete(self, *, system: str, prompt: str) -> str:
        """Return the model's text answer (expected to be JSON)."""
        ...


class GeminiJudge:
    name = "gemini"

    def __init__(self, *, model: str | None = None, api_key: str | None = None, temperature: float = 0.0, client: Any = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        if client is not None:          # injected in tests
            self._client, self._types = client, None
            return
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise JudgeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini judge.")
        try:
            from google import genai
        except ImportError:
            raise JudgeError("The Gemini judge needs the extra: pip install 'ai-eval-harness[gemini]'") from None
        self._client, self._types = genai.Client(api_key=key), genai.types

    def complete(self, *, system: str, prompt: str) -> str:
        config: Any = {"system_instruction": system, "temperature": self.temperature, "response_mime_type": "application/json"}
        if self._types is not None:
            config = self._types.GenerateContentConfig(**config)
        return self._client.models.generate_content(model=self.model, contents=prompt, config=config).text or ""


def make_judge(kind: str, model: str | None = None) -> JudgeClient | None:
    if kind == "none":
        return None
    if kind == "gemini":
        return GeminiJudge(model=model)
    raise ValueError(f"Unknown --judge value '{kind}'. Use none or gemini.")
