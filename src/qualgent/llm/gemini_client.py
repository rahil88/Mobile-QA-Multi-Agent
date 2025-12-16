"""Gemini client for vision + text prompts."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

__all__ = ["GeminiClient", "GeminiError"]

# Load .env from project root
load_dotenv()


class GeminiError(Exception):
    """Raised when Gemini API call fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GeminiClient:
    """Client for Google Gemini API with vision support.

    Parameters
    ----------
    api_key
        Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
    model
        Model name to use. Defaults to "gemini-2.5-flash".
    timeout_s
        Request timeout in seconds.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "gemini-2.5-flash",
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise GeminiError(
                "GEMINI_API_KEY not found. Set it in .env or pass api_key param."
            )
        self._model = model
        self._timeout = timeout_s
        self._client = httpx.Client(timeout=self._timeout)

    def _encode_image(self, image_path: Path) -> dict[str, Any]:
        """Encode an image file as base64 inline data for Gemini."""
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": b64,
            }
        }

    def generate(
        self,
        prompt: str,
        *,
        images: list[Path] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text from prompt and optional images.

        Parameters
        ----------
        prompt
            Text prompt to send.
        images
            Optional list of image paths to include for vision tasks.
        temperature
            Sampling temperature (0-1). Lower = more deterministic.
        max_tokens
            Maximum tokens to generate.

        Returns
        -------
        str
            Generated text response.

        Raises
        ------
        GeminiError
            If the API call fails.
        """
        # Build content parts
        parts: list[dict[str, Any]] = []

        # Add images first (if any)
        if images:
            for img_path in images:
                parts.append(self._encode_image(img_path))

        # Add text prompt
        parts.append({"text": prompt})

        # Build request payload
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        url = f"{self.BASE_URL}/{self._model}:generateContent?key={self._api_key}"

        try:
            response = self._client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except httpx.RequestError as exc:
            raise GeminiError(f"Request failed: {exc}") from exc

        if response.status_code != 200:
            raise GeminiError(
                f"Gemini API returned {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

        data = response.json()

        # Extract text from response
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise GeminiError(f"No candidates in response: {data}")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise GeminiError(f"No parts in response: {data}")
            return parts[0].get("text", "")
        except (KeyError, IndexError) as exc:
            raise GeminiError(f"Failed to parse response: {data}") from exc

    def generate_json(
        self,
        prompt: str,
        *,
        images: list[Path] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        retry_on_parse_error: bool = True,
    ) -> dict[str, Any]:
        """Generate and parse JSON from prompt.

        The prompt should instruct the model to output valid JSON.
        This method will attempt to parse the response and optionally
        retry once if parsing fails.

        Parameters
        ----------
        prompt
            Text prompt (should request JSON output).
        images
            Optional images for vision tasks.
        temperature
            Sampling temperature.
        max_tokens
            Maximum tokens to generate.
        retry_on_parse_error
            If True, retry once with a stricter prompt on parse failure.

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        GeminiError
            If JSON parsing fails after retries.
        """
        response_text = self.generate(
            prompt, images=images, temperature=temperature, max_tokens=max_tokens
        )

        # Try to extract JSON from response
        parsed = self._try_parse_json(response_text)
        if parsed is not None:
            return parsed

        if not retry_on_parse_error:
            raise GeminiError(f"Failed to parse JSON from response: {response_text}")

        # Retry with stricter prompt
        retry_prompt = (
            f"Your previous response was not valid JSON. "
            f"Please respond with ONLY valid JSON, no markdown code blocks or extra text.\n\n"
            f"Original request:\n{prompt}\n\n"
            f"Your previous (invalid) response:\n{response_text}"
        )

        response_text = self.generate(
            retry_prompt, images=images, temperature=0.0, max_tokens=max_tokens
        )

        parsed = self._try_parse_json(response_text)
        if parsed is not None:
            return parsed

        raise GeminiError(f"Failed to parse JSON after retry: {response_text}")

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Attempt to parse JSON from text, handling common formatting issues."""
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "GeminiClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

