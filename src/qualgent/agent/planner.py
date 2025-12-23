"""Planner module - uses LLM to decide next actions based on observation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from qualgent.agent.types import Action, ActionType, Observation, PlannerResponse

__all__ = ["Planner", "PlannerError"]


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


class PlannerError(Exception):
    """Raised when planning fails."""


PLANNER_SYSTEM_PROMPT = """You are a mobile app QA automation planner. Your job is to analyze the current screen state and decide what actions to take.

CRITICAL GROUNDING RULE:
You will be given a list of VISIBLE_UI_TEXTS - these are the actual text labels currently on screen.
When using tap_text, you MUST use text that exists in VISIBLE_UI_TEXTS (exact match or partial if justified).
DO NOT guess or assume text exists - if it's not in VISIBLE_UI_TEXTS, it's not on screen!

IMPORTANT RULES:
1. Output ONLY valid JSON, no markdown code blocks or explanations. NO trailing commas!
2. Plan ONE action at a time - you'll get fresh observation after each action.
3. **FILE PICKER CHECK**: If VISIBLE_UI_TEXTS contains "sdk_gphone64_arm64" or "Files in" - you're in Android's file picker!
   → IMMEDIATELY tap "USE THIS FOLDER" to confirm and exit the file picker. Don't tap folder names!
4. If the previous action failed, analyze why and try a different approach.
5. If your target text is not in VISIBLE_UI_TEXTS, consider: scroll, back, or wait.
6. NAVIGATE FORWARD: Look for "Continue", "Skip", "Next", "ALLOW" buttons to proceed.
7. SET is_complete=true IMMEDIATELY when the expected result is achieved!

ACTION TYPES (in order of preference):
- tap_text: Tap element by visible text. Params: {"text": string, "partial": bool}
  ONLY use text from VISIBLE_UI_TEXTS! Set partial=true for substring match.
- tap_and_type: Tap input field and type text. Params: {"target_text": string, "input_text": string, "partial": bool}
  USE THIS FOR INPUT FIELDS! 
  CRITICAL: target_text must be the PLACEHOLDER TEXT inside the input field, NOT the label above it!
  - Labels like "Vault name", "Password", "Email" are NOT tappable input fields - they are just text labels!
  - Placeholder text like "My vault", "Enter password", "email@example.com" ARE the actual input fields!
  - Look for grayed-out/lighter text in VISIBLE_UI_TEXTS - that's usually the placeholder.
  Example: If you see "Vault name" (label) and "My vault" (placeholder), use {"target_text": "My vault", "input_text": "InternVault"}
- scroll_until_text: Scroll until text appears. Params: {"text": string, "direction": "up"|"down"|"left"|"right", "max_swipes": int}
  Use when target might be off-screen.
- back: Press back button. Params: {}
  Use to go back or dismiss dialogs.
- type_text: Type text into ALREADY focused input field. Params: {"text": string}
  Only use if field is already focused (e.g., keyboard visible). Prefer tap_and_type instead!
- tap: Tap at coordinates. Params: {"x": float, "y": float}
  ONLY when no text label available (icons, specific areas). Coords are 0.0-1.0 normalized.
- swipe: Swipe gesture. Params: {"x1": float, "y1": float, "x2": float, "y2": float, "duration_ms": int}
- key_event: Send key. Params: {"key_code": int} (4=BACK, 66=ENTER, 3=HOME)
- wait: Wait for UI. Params: {"seconds": float}
- relaunch_app: Force stop and relaunch. Params: {"package": string}
  Use as last resort when stuck.

OUTPUT FORMAT (JSON):
{
    "action": {
        "action_type": "tap_text|tap_and_type|scroll_until_text|back|type_text|tap|swipe|key_event|wait|relaunch_app",
        "params": { ... },
        "description": "What this action does and why"
    },
    "reasoning": "Why this action based on VISIBLE_UI_TEXTS and goal",
    "is_complete": false  // true ONLY if the test goal has been fully achieved
}

*** CRITICAL: GOAL COMPLETION CHECK ***
BEFORE setting is_complete=true, verify ALL parts of the goal are achieved:

For "create vault" tests - SET is_complete=true IF you see ANY of these in VISIBLE_UI_TEXTS:
  - "Untitled - Obsidian" or "X - Obsidian" (you're viewing a note INSIDE the vault!)
  - "New tab - Obsidian" (you're in the vault's new tab screen!)
  - "Create new note" (vault UI is showing!)
  - The vault name (e.g., "InternVault") appears in breadcrumb
  >>> IF ANY OF THESE ARE TRUE: SET is_complete=true AND STOP! <<<

For "create note" tests - CHECK ALL REQUIREMENTS before setting is_complete=true:
  1. The note title from the goal appears in VISIBLE_UI_TEXTS (e.g., "Meeting Notes - Obsidian")
  2. IF the goal mentions typing content into the body (e.g., "Daily Standup"), that text MUST ALSO
     appear in VISIBLE_UI_TEXTS. If you don't see the body content yet, you haven't typed it!
  3. DO NOT assume goal is complete just because you see a note title - check if body content was required!
  >>> Only set is_complete=true when ALL required elements are visible <<<

IMPORTANT: Read the TEST GOAL carefully! If it says "type X into the body", you must:
  1. First create/open the note with the correct title
  2. Then type the body content
  3. Only set is_complete=true when BOTH title AND body content appear in VISIBLE_UI_TEXTS

STOP REPEATING THE SAME ACTION! If you've already achieved the goal, set is_complete=true.
Don't keep tapping the same element hoping something changes!

RECOVERY STRATEGIES:
- If tap_text failed (element not found): check if text is in VISIBLE_UI_TEXTS, if not try scroll_until_text or back
- If screen looks wrong: use back to return to previous screen
- If completely stuck after multiple attempts: use relaunch_app
- If type_text didn't work (text not appearing in field): You forgot to focus the input field first. Use tap_text or tap to click on the input field, then retry type_text.
- If you see an intermediate/setup screen: Look for "Continue", "Skip", "Next", or similar buttons in VISIBLE_UI_TEXTS to proceed forward.

FILE PICKER HANDLING:
- If you see "sdk_gphone64_arm64", "Documents", "Files in X" - you're in the Android FILE PICKER, not the app!
- In a file picker, DON'T tap on folder names repeatedly - look for CONFIRMATION buttons:
  * "USE THIS FOLDER" - tap this to confirm folder selection
  * "SELECT", "OK", "ALLOW", "OPEN" - tap these to proceed
- The file picker is a SYSTEM UI, not the app. You must confirm your selection to return to the app.

NOTE EDITING PATTERNS (for note-taking apps like Obsidian):
- To rename a note: Use tap_and_type on the title (e.g., "Untitled") - this will REPLACE the existing title
- After typing a TITLE, press ENTER to move to the body: use key_event with {"key_code": 66}
- To type in the BODY: First ensure cursor is in body (press Enter after title), then use type_text
- Sequence for creating a note with title and body:
  1. tap_and_type on existing title (e.g., "Untitled") with your new title (e.g., "Meeting Notes")
  2. key_event {"key_code": 66} to press Enter and move to body
  3. type_text to add body content (e.g., "Daily Standup")

VISUAL VERIFICATION TASKS:
When the goal asks you to VERIFY something visual (like a color, icon, or appearance):
1. Navigate to the screen where the element is visible
2. CAREFULLY EXAMINE THE SCREENSHOT - look at the actual colors, icons, shapes
3. Make your determination based on what you SEE in the image, not what you expect
4. If verifying a COLOR: Look at the actual pixel colors in the screenshot
   - If the goal says "verify X is RED" but you see it's gray/monochrome/another color → FAIL the test
   - Set is_complete=true and explain in reasoning that the verification FAILED because the actual color doesn't match
5. DO NOT keep navigating after you can see the element - just verify and report!
6. For verification failures, output:
   {"action": {"action_type": "wait", "params": {"seconds": 0}, "description": "Verification complete - TEST FAILED: [explain what you observed vs expected]"}, "reasoning": "...", "is_complete": true}"""


class Planner:
    """Plans next actions based on current screen state and test goal.

    Parameters
    ----------
    llm_client
        LLM client for API calls (GeminiClient or OpenAIClient).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    def plan_next_action(
        self,
        test_goal: str,
        observation: Observation,
        *,
        previous_actions: list[str] | None = None,
        step_context: str = "",
    ) -> PlannerResponse:
        """Analyze observation and plan the next action.

        Parameters
        ----------
        test_goal
            The overall goal of this test case.
        observation
            Current screen observation (screenshot + ui_texts + context).
        previous_actions
            List of actions already taken (for context).
        step_context
            Additional context about the current step.

        Returns
        -------
        PlannerResponse
            Planned action and metadata.

        Raises
        ------
        PlannerError
            If planning fails.
        """
        # Build context from observation
        ui_texts_str = "\n".join(f"- {t}" for t in observation.ui_texts[:50])  # Limit to 50
        if not ui_texts_str:
            ui_texts_str = "(no text detected on screen)"

        # Previous action result context
        prev_result_str = ""
        if observation.previous_result and not observation.previous_result.success:
            prev_result_str = f"\n\nPREVIOUS ACTION FAILED:\n- Action: {observation.previous_action.description if observation.previous_action else 'unknown'}\n- Error: {observation.previous_result.error_message}\nYou must try a DIFFERENT approach!"

        # Action history
        history = ""
        if previous_actions:
            history = "\n\nAction history (recent):\n" + "\n".join(
                f"- {a}" for a in previous_actions[-5:]
            )

        # Attempted actions this step (to avoid repeats)
        attempted = ""
        if observation.attempted_actions:
            attempted = "\n\nAlready attempted (don't repeat):\n" + "\n".join(
                f"- {a}" for a in observation.attempted_actions
            )

        context = ""
        if step_context:
            context = f"\n\nStep context: {step_context}"

        prompt = f"""{PLANNER_SYSTEM_PROMPT}

TEST GOAL: {test_goal}
{context}

VISIBLE_UI_TEXTS (what's actually on screen):
{ui_texts_str}
{prev_result_str}
{history}
{attempted}

Based on the screenshot and VISIBLE_UI_TEXTS, output JSON for the next action.
Remember: only use tap_text with text from VISIBLE_UI_TEXTS!"""

        try:
            response = self._client.generate_json(
                prompt,
                images=[observation.screenshot_path],
                temperature=0.1,
            )
        except Exception as exc:
            raise PlannerError(f"LLM API error: {exc}") from exc

        return self._parse_response(response)

    # Keep old method for backward compatibility during transition
    def plan_next_actions(
        self,
        test_goal: str,
        screenshot_path: Path,
        *,
        previous_actions: list[str] | None = None,
        step_context: str = "",
    ) -> PlannerResponse:
        """Legacy method - creates a minimal Observation and calls plan_next_action."""
        observation = Observation(
            screenshot_path=screenshot_path,
            ui_texts=[],  # No UI texts in legacy mode
        )
        return self.plan_next_action(
            test_goal, observation,
            previous_actions=previous_actions,
            step_context=step_context,
        )

    def _parse_response(self, data: dict[str, Any]) -> PlannerResponse:
        """Parse and validate the LLM response."""
        actions: list[Action] = []

        # New format: single "action" key
        if "action" in data:
            raw_action = data["action"]
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
        # Legacy format: "actions" list
        elif "actions" in data:
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
            notes=data.get("reasoning", data.get("notes", "")),
            is_complete=bool(data.get("is_complete", False)),
        )

