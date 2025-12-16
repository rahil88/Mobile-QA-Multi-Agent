"""Supervisor module - verifies test outcomes using Gemini vision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qualgent.agent.types import SupervisorVerdict, TestStatus
from qualgent.llm.gemini_client import GeminiClient, GeminiError

__all__ = ["Supervisor", "SupervisorError"]


class SupervisorError(Exception):
    """Raised when verification fails."""


SUPERVISOR_SYSTEM_PROMPT = """You are a QA test supervisor. Your job is to verify whether a test step or test case has passed or failed by analyzing screenshots.

IMPORTANT RULES:
1. Output ONLY valid JSON, no markdown code blocks or explanations.
2. Be precise and objective - describe exactly what you see.
3. For color checks, describe the actual colors you observe.
4. For element existence checks, confirm whether elements are present or absent.
5. Match your observations against the expected result to determine pass/fail.

OUTPUT FORMAT (JSON):
{
    "status": "PASSED" or "FAILED",
    "evidence": "Detailed description of what you observed in the screenshot(s)",
    "expected_vs_actual": "Comparison: Expected X, but observed Y",
    "confidence": 0.95  // 0.0 to 1.0, how confident you are in this verdict
}

VERIFICATION GUIDELINES:
- PASSED: The expected state is clearly visible in the screenshot.
- FAILED: The expected state is NOT present, or a different state is observed.
- When checking for text, look for exact or near-exact matches.
- When checking for colors, describe the actual color (e.g., "gray", "white", "blue").
- When checking for missing elements, confirm you've looked in the expected location.
- Be honest about uncertainty - use lower confidence if the evidence is ambiguous."""


class Supervisor:
    """Verifies test outcomes by analyzing screenshots.

    Parameters
    ----------
    gemini_client
        Gemini client for LLM calls.
    """

    def __init__(self, gemini_client: GeminiClient) -> None:
        self._client = gemini_client

    def verify_step(
        self,
        expected_result: str,
        screenshot_path: Path,
        *,
        before_screenshot: Path | None = None,
        additional_context: str = "",
    ) -> SupervisorVerdict:
        """Verify a test step by analyzing screenshot(s).

        Parameters
        ----------
        expected_result
            Description of what should be visible/true.
        screenshot_path
            Path to current (after) screenshot.
        before_screenshot
            Optional path to before screenshot for comparison.
        additional_context
            Any additional context about what was being tested.

        Returns
        -------
        SupervisorVerdict
            Verification result with evidence.

        Raises
        ------
        SupervisorError
            If verification fails.
        """
        images = [screenshot_path]
        image_desc = "the screenshot"

        if before_screenshot:
            images = [before_screenshot, screenshot_path]
            image_desc = "the BEFORE and AFTER screenshots (in order)"

        context = ""
        if additional_context:
            context = f"\n\nAdditional context: {additional_context}"

        prompt = f"""{SUPERVISOR_SYSTEM_PROMPT}

EXPECTED RESULT: {expected_result}
{context}

Analyze {image_desc} and verify whether the expected result is achieved.
Output your verdict as JSON."""

        try:
            response = self._client.generate_json(
                prompt,
                images=images,
                temperature=0.1,
            )
        except GeminiError as exc:
            raise SupervisorError(f"Gemini API error: {exc}") from exc

        return self._parse_response(response)

    def verify_test_completion(
        self,
        test_goal: str,
        expected_result: str,
        final_screenshot: Path,
        *,
        action_history: list[str] | None = None,
    ) -> SupervisorVerdict:
        """Verify overall test completion.

        Parameters
        ----------
        test_goal
            The goal of the test case.
        expected_result
            What success should look like.
        final_screenshot
            Screenshot of final state.
        action_history
            List of actions that were taken.

        Returns
        -------
        SupervisorVerdict
            Final test verdict.
        """
        history = ""
        if action_history:
            history = "\n\nActions taken during test:\n" + "\n".join(
                f"- {a}" for a in action_history
            )

        prompt = f"""{SUPERVISOR_SYSTEM_PROMPT}

TEST GOAL: {test_goal}

EXPECTED RESULT: {expected_result}
{history}

This is the FINAL verification for the complete test case.
Analyze the screenshot and determine if the test PASSED or FAILED.
Output your verdict as JSON."""

        try:
            response = self._client.generate_json(
                prompt,
                images=[final_screenshot],
                temperature=0.1,
            )
        except GeminiError as exc:
            raise SupervisorError(f"Gemini API error: {exc}") from exc

        return self._parse_response(response)

    def _parse_response(self, data: dict[str, Any]) -> SupervisorVerdict:
        """Parse and validate the Gemini response."""
        status_str = data.get("status", "").upper()

        if status_str == "PASSED":
            status = TestStatus.PASSED
        elif status_str == "FAILED":
            status = TestStatus.FAILED
        else:
            # Default to failed if unclear
            status = TestStatus.FAILED

        return SupervisorVerdict(
            status=status,
            evidence=data.get("evidence", "No evidence provided"),
            expected_vs_actual=data.get("expected_vs_actual", ""),
            confidence=float(data.get("confidence", 0.5)),
        )

