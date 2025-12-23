"""OpenAI client for vision + text prompts."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

__all__ = ["OpenAIClient", "OpenAIError"]

# Load .env from project root
load_dotenv()


class OpenAIError(Exception):
    """Raised when OpenAI API call fails."""

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


class OpenAIClient:
    """Client for OpenAI API with vision support.

    Parameters
    ----------
    api_key
        OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
    model
        Model name to use. Defaults to "gpt-5-mini".
    timeout_s
        Request timeout in seconds.
    """

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "gpt-5-mini",
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise OpenAIError(
                "OPENAI_API_KEY not found. Set it in .env or pass api_key param."
            )
        self._model = model
        self._timeout = timeout_s
        self._client = httpx.Client(timeout=self._timeout)

    def _encode_image(self, image_path: Path) -> dict[str, Any]:
        """Encode an image file as base64 data URL for OpenAI."""
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
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64}",
                "detail": "high",
            },
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
        OpenAIError
            If the API call fails.
        """
        # Build content array
        content: list[dict[str, Any]] = []

        # Add images first (if any)
        if images:
            for img_path in images:
                content.append(self._encode_image(img_path))

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Build request payload
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_completion_tokens": max_tokens,
        }
        
        # Only include temperature if not using a model that restricts it
        # gpt-5-mini and similar models only support the default temperature (1)
        if not self._model.startswith("gpt-5"):
            payload["temperature"] = temperature

        # Retry logic for rate limits (429)
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self._client.post(
                    self.BASE_URL,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                )
            except httpx.RequestError as exc:
                raise OpenAIError(f"Request failed: {exc}") from exc

            if response.status_code == 200:
                break

            if response.status_code == 429 and attempt < max_retries - 1:
                # Parse retry delay from error message if available
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                try:
                    # Try to extract delay from message like "try again in 413ms"
                    match = re.search(r"try again in (\d+)ms", response.text)
                    if match:
                        delay = max(delay, int(match.group(1)) / 1000 + 0.5)
                except Exception:
                    pass
                print(f"    [Rate limit] Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(delay)
                continue

            raise OpenAIError(
                f"OpenAI API returned {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
                response_body=response.text,
            )

        data = response.json()

        # Extract text from response
        try:
            choices = data.get("choices", [])
            if not choices:
                raise OpenAIError(f"No choices in response: {data}")
            message = choices[0].get("message", {})
            return message.get("content", "")
        except (KeyError, IndexError) as exc:
            raise OpenAIError(f"Failed to parse response: {data}") from exc

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
        OpenAIError
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
            raise OpenAIError(f"Failed to parse JSON from response: {response_text}")

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

        raise OpenAIError(f"Failed to parse JSON after retry: {response_text}")

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
            json_text = text[start:end]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

            # Fix trailing commas (common LLM mistake)
            # Remove trailing commas before } or ]
            fixed = re.sub(r',(\s*[}\]])', r'\1', json_text)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            # Fix missing commas between fields (another common LLM mistake)
            # Pattern: "value"\n    "key" should become "value",\n    "key"
            fixed = re.sub(r'(")\s*\n(\s*")', r'\1,\n\2', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "OpenAIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

