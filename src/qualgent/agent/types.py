"""Type definitions for the Supervisor-Planner-Executor QA agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ActionType(str, Enum):
    """Types of actions the executor can perform."""

    TAP = "tap"
    SWIPE = "swipe"
    TYPE_TEXT = "type_text"
    KEY_EVENT = "key_event"
    LAUNCH_APP = "launch_app"
    FORCE_STOP = "force_stop"
    CLEAR_DATA = "clear_data"
    WAIT = "wait"
    SCREENSHOT = "screenshot"


@dataclass
class Action:
    """A single action to be executed by the Executor.

    Attributes
    ----------
    action_type
        The type of action to perform.
    params
        Parameters for the action. Structure depends on action_type:
        - tap: {"x": float, "y": float} (normalized 0-1 coords)
        - swipe: {"x1": float, "y1": float, "x2": float, "y2": float, "duration_ms": int}
        - type_text: {"text": str}
        - key_event: {"key_code": int}
        - launch_app: {"package": str}
        - force_stop: {"package": str}
        - clear_data: {"package": str}
        - wait: {"seconds": float}
        - screenshot: {"path": str} (optional)
    description
        Human-readable description of what this action does.
    """

    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class PlannerResponse:
    """Response from the Planner LLM.

    Attributes
    ----------
    actions
        List of actions to execute in order.
    stop_condition
        Description of what state indicates the step is complete.
    notes
        Any additional observations or reasoning from the planner.
    is_complete
        Whether the planner believes the test goal has been achieved.
    """

    actions: list[Action]
    stop_condition: str = ""
    notes: str = ""
    is_complete: bool = False


@dataclass
class StepResult:
    """Result of executing a single action.

    Attributes
    ----------
    action
        The action that was executed.
    success
        Whether the action executed without errors.
    error_message
        Error message if success is False.
    screenshot_path
        Path to screenshot taken after this action (if any).
    """

    action: Action
    success: bool
    error_message: str = ""
    screenshot_path: Path | None = None


class TestStatus(str, Enum):
    """Status of a test case."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class SupervisorVerdict:
    """Verdict from the Supervisor after verifying a test step or test completion.

    Attributes
    ----------
    status
        PASSED or FAILED.
    evidence
        What the supervisor observed in the screenshot(s).
    expected_vs_actual
        Comparison of expected vs actual state.
    confidence
        Confidence level (0-1) in the verdict.
    """

    status: TestStatus
    evidence: str
    expected_vs_actual: str = ""
    confidence: float = 1.0


@dataclass
class TestCase:
    """A QA test case to execute.

    Attributes
    ----------
    id
        Unique identifier for the test.
    name
        Human-readable name.
    description
        Natural language description of what to test.
    expected_result
        Description of what success looks like.
    should_pass
        Whether this test is expected to pass (for reporting purposes).
    setup_actions
        Optional actions to run before the test (e.g., launch app).
    teardown_actions
        Optional actions to run after the test (e.g., force stop).
    """

    id: str
    name: str
    description: str
    expected_result: str
    should_pass: bool = True
    setup_actions: list[Action] = field(default_factory=list)
    teardown_actions: list[Action] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of running a complete test case.

    Attributes
    ----------
    test_case
        The test case that was run.
    status
        Final status (passed/failed/error).
    verdict
        Supervisor's final verdict.
    steps
        List of all step results during the test.
    screenshots
        Paths to all screenshots taken during the test.
    duration_seconds
        How long the test took to run.
    error_message
        Error message if status is ERROR.
    """

    test_case: TestCase
    status: TestStatus
    verdict: SupervisorVerdict | None = None
    steps: list[StepResult] = field(default_factory=list)
    screenshots: list[Path] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: str = ""

