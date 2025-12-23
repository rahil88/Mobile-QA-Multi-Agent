"""Supervisor module - verifies test outcomes using LLM vision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from qualgent.agent.types import SupervisorVerdict, TestStatus

__all__ = ["Supervisor", "SupervisorError"]


class LLMClient(Protocol):
    """Protocol for LLM clients (Gemini or OpenAI)."""

    def generate_json(
        self,
        prompt: str,
        *,
        images: list[Path] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...


class SupervisorError(Exception):
    """Raised when verification fails."""


SUPERVISOR_SYSTEM_PROMPT = """You are a QA test supervisor. Your job is to verify whether a test step or test case has passed or failed by analyzing screenshots and UI context.

IMPORTANT RULES:
1. Output ONLY valid JSON, no markdown code blocks or explanations.
2. Be precise and objective - describe exactly what you see.
3. Use the VISIBLE_UI_TEXTS list to verify text presence (this is ground truth from the device).
4. For color checks, describe the actual colors you observe in the screenshot.
5. For element existence checks, check both screenshot AND VISIBLE_UI_TEXTS.
6. Match your observations against the expected result to determine pass/fail.

OUTPUT FORMAT (JSON):
{
    "status": "PASSED" or "FAILED",
    "evidence": "Detailed description of what you observed in the screenshot(s) and UI texts",
    "expected_vs_actual": "Comparison: Expected X, but observed Y",
    "confidence": 0.95  // 0.0 to 1.0, how confident you are in this verdict
}

VERIFICATION GUIDELINES:
- PASSED: The expected state is clearly visible in the screenshot AND/OR confirmed in VISIBLE_UI_TEXTS.
- FAILED: The expected state is NOT present, or a different state is observed.
- When checking for text, first check VISIBLE_UI_TEXTS (ground truth), then screenshot.
- When checking for colors, describe the actual color (e.g., "gray", "white", "blue").
- When checking for missing elements, confirm you've looked in the expected location.
- Be honest about uncertainty - use lower confidence if the evidence is ambiguous.

APP VS SYSTEM UI DISTINCTION (CRITICAL):
- If you see "sdk_gphone64_arm64", "Documents", "Files in X", "USE THIS FOLDER" - this is the Android FILE PICKER, NOT the app!
- For "inside vault" tests: You must see the APP UI (e.g., "Obsidian", "Untitled - Obsidian", "Create new note", note editing UI)
- Just seeing the vault name in a file picker does NOT mean you're inside the vault - mark as FAILED!
- The app title bar usually shows "[AppName]" or "[NoteName] - [AppName]" when you're actually inside the app."""


class Supervisor:
    """Verifies test outcomes by analyzing screenshots and UI context.

    Parameters
    ----------
    llm_client
        LLM client for API calls (GeminiClient or OpenAIClient).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    def verify_step(
        self,
        expected_result: str,
        screenshot_path: Path,
        *,
        before_screenshot: Path | None = None,
        ui_texts: list[str] | None = None,
        additional_context: str = "",
    ) -> SupervisorVerdict:
        """Verify a test step by analyzing screenshot(s) and UI context.

        Parameters
        ----------
        expected_result
            Description of what should be visible/true.
        screenshot_path
            Path to current (after) screenshot.
        before_screenshot
            Optional path to before screenshot for comparison.
        ui_texts
            List of visible text labels from UI dump (ground truth).
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

        # Build UI texts section
        ui_texts_str = ""
        if ui_texts:
            ui_texts_str = "\n\nVISIBLE_UI_TEXTS (ground truth from device):\n" + "\n".join(
                f"- {t}" for t in ui_texts[:30]
            )

        context = ""
        if additional_context:
            context = f"\n\nAdditional context: {additional_context}"

        prompt = f"""{SUPERVISOR_SYSTEM_PROMPT}

EXPECTED RESULT: {expected_result}
{ui_texts_str}
{context}

Analyze {image_desc} and verify whether the expected result is achieved.
Output your verdict as JSON."""

        try:
            response = self._client.generate_json(
                prompt,
                images=images,
                temperature=0.1,
            )
        except Exception as exc:
            raise SupervisorError(f"LLM API error: {exc}") from exc

        return self._parse_response(response)

    def verify_test_completion(
        self,
        test_goal: str,
        expected_result: str,
        final_screenshot: Path,
        *,
        action_history: list[str] | None = None,
        ui_texts: list[str] | None = None,
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
        ui_texts
            List of visible text labels from UI dump (ground truth).

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

        # Build UI texts section
        ui_texts_str = ""
        if ui_texts:
            ui_texts_str = "\n\nVISIBLE_UI_TEXTS (ground truth from device):\n" + "\n".join(
                f"- {t}" for t in ui_texts[:30]
            )

        prompt = f"""{SUPERVISOR_SYSTEM_PROMPT}

TEST GOAL: {test_goal}

EXPECTED RESULT: {expected_result}
{ui_texts_str}
{history}

This is the FINAL verification for the complete test case.
Analyze the screenshot and VISIBLE_UI_TEXTS to determine if the test PASSED or FAILED.
Output your verdict as JSON."""

        try:
            response = self._client.generate_json(
                prompt,
                images=[final_screenshot],
                temperature=0.1,
            )
        except Exception as exc:
            raise SupervisorError(f"LLM API error: {exc}") from exc

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

