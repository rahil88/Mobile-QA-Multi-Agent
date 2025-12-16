"""Planner module - uses Gemini to decide next actions based on screenshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qualgent.agent.types import Action, ActionType, PlannerResponse
from qualgent.llm.gemini_client import GeminiClient, GeminiError

__all__ = ["Planner", "PlannerError"]


class PlannerError(Exception):
    """Raised when planning fails."""


PLANNER_SYSTEM_PROMPT = """You are a mobile app QA automation planner. Your job is to analyze screenshots and decide what actions to take to accomplish the given test goal.

IMPORTANT RULES:
1. Output ONLY valid JSON, no markdown code blocks or explanations.
2. Coordinates are NORMALIZED (0.0 to 1.0), where (0,0) is top-left and (1,1) is bottom-right.
3. Plan ONE or TWO actions at a time - don't try to do everything at once.
4. After each action, you'll get a new screenshot to verify the result.

ACTION TYPES:
- tap: Tap at coordinates. Params: {"x": float, "y": float}
- swipe: Swipe gesture. Params: {"x1": float, "y1": float, "x2": float, "y2": float, "duration_ms": int}
- type_text: Type text (assumes a text field is focused). Params: {"text": string}
- key_event: Send key event. Params: {"key_code": int} (4=BACK, 66=ENTER, 3=HOME)
- wait: Wait for UI to settle. Params: {"seconds": float}
- launch_app: Launch app. Params: {"package": string}
- force_stop: Force stop app. Params: {"package": string}

OUTPUT FORMAT (JSON):
{
    "actions": [
        {
            "action_type": "tap|swipe|type_text|key_event|wait|launch_app|force_stop",
            "params": { ... },
            "description": "Human-readable description of what this does"
        }
    ],
    "stop_condition": "Description of what state indicates this step is complete",
    "notes": "Any observations or reasoning",
    "is_complete": false  // true if the test goal has been fully achieved
}

COORDINATE TIPS:
- Look for buttons, text fields, menu items in the screenshot
- Estimate coordinates based on visual position
- Center of screen is approximately (0.5, 0.5)
- Top of screen is y=0.0, bottom is y=1.0
- For text input, first tap the field, then use type_text"""


class Planner:
    """Plans next actions based on current screen state and test goal.

    Parameters
    ----------
    gemini_client
        Gemini client for LLM calls.
    """

    def __init__(self, gemini_client: GeminiClient) -> None:
        self._client = gemini_client

    def plan_next_actions(
        self,
        test_goal: str,
        screenshot_path: Path,
        *,
        previous_actions: list[str] | None = None,
        step_context: str = "",
    ) -> PlannerResponse:
        """Analyze screenshot and plan next actions.

        Parameters
        ----------
        test_goal
            The overall goal of this test case.
        screenshot_path
            Path to current screenshot.
        previous_actions
            List of actions already taken (for context).
        step_context
            Additional context about the current step.

        Returns
        -------
        PlannerResponse
            Planned actions and metadata.

        Raises
        ------
        PlannerError
            If planning fails.
        """
        # Build the user prompt
        history = ""
        if previous_actions:
            history = "\n\nPrevious actions taken:\n" + "\n".join(
                f"- {a}" for a in previous_actions[-5:]  # Last 5 actions
            )

        context = ""
        if step_context:
            context = f"\n\nCurrent step context: {step_context}"

        prompt = f"""{PLANNER_SYSTEM_PROMPT}

TEST GOAL: {test_goal}
{context}
{history}

Analyze the screenshot and output JSON for the next action(s) to take toward the goal.
If the goal appears to be achieved, set is_complete to true."""

        try:
            response = self._client.generate_json(
                prompt,
                images=[screenshot_path],
                temperature=0.1,
            )
        except GeminiError as exc:
            raise PlannerError(f"Gemini API error: {exc}") from exc

        return self._parse_response(response)

    def _parse_response(self, data: dict[str, Any]) -> PlannerResponse:
        """Parse and validate the Gemini response."""
        actions: list[Action] = []

        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list):
            raise PlannerError(f"Expected 'actions' to be a list, got: {type(raw_actions)}")

        for raw_action in raw_actions:
            try:
                action_type = ActionType(raw_action.get("action_type", ""))
            except ValueError:
                raise PlannerError(
                    f"Invalid action_type: {raw_action.get('action_type')}"
                )

            actions.append(
                Action(
                    action_type=action_type,
                    params=raw_action.get("params", {}),
                    description=raw_action.get("description", ""),
                )
            )

        return PlannerResponse(
            actions=actions,
            stop_condition=data.get("stop_condition", ""),
            notes=data.get("notes", ""),
            is_complete=bool(data.get("is_complete", False)),
        )

