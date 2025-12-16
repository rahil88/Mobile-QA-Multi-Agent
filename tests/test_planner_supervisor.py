"""Unit tests for Planner and Supervisor JSON parsing (no network required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qualgent.agent.planner import Planner, PlannerError
from qualgent.agent.supervisor import Supervisor, SupervisorError
from qualgent.agent.types import ActionType, TestStatus


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


class TestPlannerParsing:
    """Test Planner response parsing."""

    @pytest.fixture
    def mock_gemini(self) -> MagicMock:
        """Create a mock GeminiClient."""
        return MagicMock()

    @pytest.fixture
    def planner(self, mock_gemini: MagicMock) -> Planner:
        """Create a Planner with mocked Gemini client."""
        return Planner(mock_gemini)

    def test_parse_tap_action(self, planner: Planner, mock_gemini: MagicMock) -> None:
        """Planner parses tap action correctly."""
        mock_gemini.generate_json.return_value = {
            "actions": [
                {
                    "action_type": "tap",
                    "params": {"x": 0.5, "y": 0.3},
                    "description": "Tap the create button",
                }
            ],
            "stop_condition": "Vault creation dialog appears",
            "notes": "Found create vault button",
            "is_complete": False,
        }

        result = planner.plan_next_actions(
            test_goal="Create a vault",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert len(result.actions) == 1
        assert result.actions[0].action_type == ActionType.TAP
        assert result.actions[0].params["x"] == 0.5
        assert result.actions[0].params["y"] == 0.3
        assert result.stop_condition == "Vault creation dialog appears"
        assert result.is_complete is False

    def test_parse_type_text_action(self, planner: Planner, mock_gemini: MagicMock) -> None:
        """Planner parses type_text action correctly."""
        mock_gemini.generate_json.return_value = {
            "actions": [
                {
                    "action_type": "type_text",
                    "params": {"text": "InternVault"},
                    "description": "Type vault name",
                }
            ],
            "stop_condition": "Text appears in field",
            "notes": "",
            "is_complete": False,
        }

        result = planner.plan_next_actions(
            test_goal="Name the vault",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert len(result.actions) == 1
        assert result.actions[0].action_type == ActionType.TYPE_TEXT
        assert result.actions[0].params["text"] == "InternVault"

    def test_parse_multiple_actions(self, planner: Planner, mock_gemini: MagicMock) -> None:
        """Planner parses multiple actions in sequence."""
        mock_gemini.generate_json.return_value = {
            "actions": [
                {"action_type": "tap", "params": {"x": 0.5, "y": 0.2}, "description": "Tap field"},
                {"action_type": "type_text", "params": {"text": "Hello"}, "description": "Type text"},
                {"action_type": "key_event", "params": {"key_code": 66}, "description": "Press enter"},
            ],
            "stop_condition": "Form submitted",
            "notes": "",
            "is_complete": False,
        }

        result = planner.plan_next_actions(
            test_goal="Fill form",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert len(result.actions) == 3
        assert result.actions[0].action_type == ActionType.TAP
        assert result.actions[1].action_type == ActionType.TYPE_TEXT
        assert result.actions[2].action_type == ActionType.KEY_EVENT

    def test_parse_complete_flag(self, planner: Planner, mock_gemini: MagicMock) -> None:
        """Planner recognizes is_complete flag."""
        mock_gemini.generate_json.return_value = {
            "actions": [],
            "stop_condition": "",
            "notes": "Goal achieved - vault is open",
            "is_complete": True,
        }

        result = planner.plan_next_actions(
            test_goal="Enter vault",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert result.is_complete is True
        assert len(result.actions) == 0

    def test_invalid_action_type_raises_error(self, planner: Planner, mock_gemini: MagicMock) -> None:
        """Planner raises error for invalid action type."""
        mock_gemini.generate_json.return_value = {
            "actions": [
                {"action_type": "invalid_action", "params": {}, "description": ""},
            ],
            "stop_condition": "",
            "notes": "",
            "is_complete": False,
        }

        with pytest.raises(PlannerError) as exc_info:
            planner.plan_next_actions(
                test_goal="Do something",
                screenshot_path=Path("/fake/screenshot.png"),
            )

        assert "invalid_action" in str(exc_info.value).lower()

    def test_empty_actions_list(self, planner: Planner, mock_gemini: MagicMock) -> None:
        """Planner handles empty actions list."""
        mock_gemini.generate_json.return_value = {
            "actions": [],
            "stop_condition": "Waiting for UI",
            "notes": "No action needed yet",
            "is_complete": False,
        }

        result = planner.plan_next_actions(
            test_goal="Wait",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert len(result.actions) == 0
        assert result.is_complete is False


# ---------------------------------------------------------------------------
# Supervisor tests
# ---------------------------------------------------------------------------


class TestSupervisorParsing:
    """Test Supervisor response parsing."""

    @pytest.fixture
    def mock_gemini(self) -> MagicMock:
        """Create a mock GeminiClient."""
        return MagicMock()

    @pytest.fixture
    def supervisor(self, mock_gemini: MagicMock) -> Supervisor:
        """Create a Supervisor with mocked Gemini client."""
        return Supervisor(mock_gemini)

    def test_parse_passed_verdict(self, supervisor: Supervisor, mock_gemini: MagicMock) -> None:
        """Supervisor parses PASSED verdict correctly."""
        mock_gemini.generate_json.return_value = {
            "status": "PASSED",
            "evidence": "The vault 'InternVault' is visible in the sidebar",
            "expected_vs_actual": "Expected: InternVault visible. Actual: InternVault visible.",
            "confidence": 0.95,
        }

        result = supervisor.verify_step(
            expected_result="Vault InternVault is visible",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert result.status == TestStatus.PASSED
        assert "InternVault" in result.evidence
        assert result.confidence == 0.95

    def test_parse_failed_verdict(self, supervisor: Supervisor, mock_gemini: MagicMock) -> None:
        """Supervisor parses FAILED verdict correctly."""
        mock_gemini.generate_json.return_value = {
            "status": "FAILED",
            "evidence": "The Appearance icon is gray/monochrome, not red",
            "expected_vs_actual": "Expected: Red icon. Actual: Gray icon.",
            "confidence": 0.9,
        }

        result = supervisor.verify_step(
            expected_result="Appearance icon is red",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert result.status == TestStatus.FAILED
        assert "gray" in result.evidence.lower() or "monochrome" in result.evidence.lower()

    def test_parse_lowercase_status(self, supervisor: Supervisor, mock_gemini: MagicMock) -> None:
        """Supervisor handles lowercase status strings."""
        mock_gemini.generate_json.return_value = {
            "status": "passed",
            "evidence": "Test passed",
            "expected_vs_actual": "",
            "confidence": 1.0,
        }

        result = supervisor.verify_step(
            expected_result="Something",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert result.status == TestStatus.PASSED

    def test_unknown_status_defaults_to_failed(self, supervisor: Supervisor, mock_gemini: MagicMock) -> None:
        """Supervisor defaults to FAILED for unknown status."""
        mock_gemini.generate_json.return_value = {
            "status": "UNKNOWN",
            "evidence": "Cannot determine",
            "expected_vs_actual": "",
            "confidence": 0.3,
        }

        result = supervisor.verify_step(
            expected_result="Something",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert result.status == TestStatus.FAILED

    def test_missing_confidence_defaults(self, supervisor: Supervisor, mock_gemini: MagicMock) -> None:
        """Supervisor uses default confidence if not provided."""
        mock_gemini.generate_json.return_value = {
            "status": "PASSED",
            "evidence": "Looks good",
        }

        result = supervisor.verify_step(
            expected_result="Something",
            screenshot_path=Path("/fake/screenshot.png"),
        )

        assert result.confidence == 0.5  # Default

    def test_verify_test_completion(self, supervisor: Supervisor, mock_gemini: MagicMock) -> None:
        """Supervisor verifies complete test with action history."""
        mock_gemini.generate_json.return_value = {
            "status": "PASSED",
            "evidence": "Note 'Meeting Notes' is open with 'Daily Standup' content",
            "expected_vs_actual": "All conditions met",
            "confidence": 0.98,
        }

        result = supervisor.verify_test_completion(
            test_goal="Create a note with content",
            expected_result="Note 'Meeting Notes' contains 'Daily Standup'",
            final_screenshot=Path("/fake/final.png"),
            action_history=["tap: Create note", "type_text: Meeting Notes", "type_text: Daily Standup"],
        )

        assert result.status == TestStatus.PASSED
        assert result.confidence == 0.98


# ---------------------------------------------------------------------------
# Integration-style tests (still mocked)
# ---------------------------------------------------------------------------


class TestPlannerSupervisorIntegration:
    """Test Planner and Supervisor working together."""

    def test_plan_then_verify_flow(self) -> None:
        """Simulate a plan-execute-verify flow."""
        mock_gemini = MagicMock()

        # First call: Planner plans an action
        mock_gemini.generate_json.side_effect = [
            {
                "actions": [
                    {"action_type": "tap", "params": {"x": 0.5, "y": 0.5}, "description": "Tap button"},
                ],
                "stop_condition": "Button pressed",
                "notes": "",
                "is_complete": False,
            },
            # Second call: Supervisor verifies
            {
                "status": "PASSED",
                "evidence": "Button was successfully pressed, new screen visible",
                "expected_vs_actual": "Expected: new screen. Actual: new screen.",
                "confidence": 0.9,
            },
        ]

        planner = Planner(mock_gemini)
        supervisor = Supervisor(mock_gemini)

        # Plan
        plan = planner.plan_next_actions(
            test_goal="Press the button",
            screenshot_path=Path("/fake/before.png"),
        )
        assert len(plan.actions) == 1

        # Verify (after execution would happen)
        verdict = supervisor.verify_step(
            expected_result="Button press leads to new screen",
            screenshot_path=Path("/fake/after.png"),
        )
        assert verdict.status == TestStatus.PASSED

