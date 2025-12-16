"""Executor module - translates planned actions to ADB commands."""

from __future__ import annotations

from pathlib import Path

from qualgent.agent.types import Action, ActionType, StepResult
from qualgent.tools.adb_controller import AdbController, AdbError

__all__ = ["Executor", "ExecutorError"]


class ExecutorError(Exception):
    """Raised when action execution fails."""


class Executor:
    """Executes planned actions via ADB.

    Parameters
    ----------
    adb
        AdbController instance.
    screen_width
        Screen width in pixels (for coordinate conversion).
    screen_height
        Screen height in pixels (for coordinate conversion).
    """

    def __init__(
        self,
        adb: AdbController,
        screen_width: int,
        screen_height: int,
    ) -> None:
        self._adb = adb
        self._screen_width = screen_width
        self._screen_height = screen_height

    @classmethod
    def from_adb(cls, adb: AdbController) -> "Executor":
        """Create executor, automatically fetching screen dimensions.

        Parameters
        ----------
        adb
            AdbController instance.

        Returns
        -------
        Executor
            Configured executor.
        """
        width, height = adb.get_screen_size()
        return cls(adb, width, height)

    def _normalized_to_pixels(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalized (0-1) coordinates to pixel coordinates."""
        px = int(x * self._screen_width)
        py = int(y * self._screen_height)
        return px, py

    def execute(self, action: Action) -> StepResult:
        """Execute a single action.

        Parameters
        ----------
        action
            The action to execute.

        Returns
        -------
        StepResult
            Result of the action execution.
        """
        try:
            self._execute_action(action)
            return StepResult(action=action, success=True)
        except (AdbError, ExecutorError) as exc:
            return StepResult(
                action=action,
                success=False,
                error_message=str(exc),
            )

    def execute_all(
        self,
        actions: list[Action],
        *,
        screenshot_dir: Path | None = None,
    ) -> list[StepResult]:
        """Execute a list of actions in order.

        Parameters
        ----------
        actions
            Actions to execute.
        screenshot_dir
            If provided, take a screenshot after each action.

        Returns
        -------
        list[StepResult]
            Results for each action.
        """
        results: list[StepResult] = []

        for i, action in enumerate(actions):
            result = self.execute(action)

            # Take screenshot after action if requested
            if screenshot_dir and result.success:
                screenshot_path = screenshot_dir / f"step_{i:03d}.png"
                try:
                    self._adb.take_screenshot(screenshot_path)
                    result.screenshot_path = screenshot_path
                except AdbError:
                    pass  # Non-fatal if screenshot fails

            results.append(result)

            # Stop on first failure
            if not result.success:
                break

        return results

    def _execute_action(self, action: Action) -> None:
        """Internal action dispatch."""
        match action.action_type:
            case ActionType.TAP:
                self._do_tap(action.params)

            case ActionType.SWIPE:
                self._do_swipe(action.params)

            case ActionType.TYPE_TEXT:
                text = action.params.get("text", "")
                if not text:
                    raise ExecutorError("type_text requires 'text' param")
                self._adb.type_text(text)

            case ActionType.KEY_EVENT:
                key_code = action.params.get("key_code")
                if key_code is None:
                    raise ExecutorError("key_event requires 'key_code' param")
                self._adb.send_key_event(int(key_code))

            case ActionType.LAUNCH_APP:
                package = action.params.get("package", "")
                if not package:
                    raise ExecutorError("launch_app requires 'package' param")
                self._adb.launch_app(package)

            case ActionType.FORCE_STOP:
                package = action.params.get("package", "")
                if not package:
                    raise ExecutorError("force_stop requires 'package' param")
                self._adb.force_stop(package)

            case ActionType.CLEAR_DATA:
                package = action.params.get("package", "")
                if not package:
                    raise ExecutorError("clear_data requires 'package' param")
                self._adb.clear_app_data(package)

            case ActionType.WAIT:
                seconds = action.params.get("seconds", 1.0)
                AdbController.wait(float(seconds))

            case ActionType.SCREENSHOT:
                path = action.params.get("path", "screenshot.png")
                self._adb.take_screenshot(path)

            case _:
                raise ExecutorError(f"Unknown action type: {action.action_type}")

    def _do_tap(self, params: dict) -> None:
        """Execute a tap action."""
        x = params.get("x")
        y = params.get("y")
        if x is None or y is None:
            raise ExecutorError("tap requires 'x' and 'y' params")
        px, py = self._normalized_to_pixels(float(x), float(y))
        self._adb.tap_coordinates(px, py)

    def _do_swipe(self, params: dict) -> None:
        """Execute a swipe action."""
        x1 = params.get("x1")
        y1 = params.get("y1")
        x2 = params.get("x2")
        y2 = params.get("y2")
        if None in (x1, y1, x2, y2):
            raise ExecutorError("swipe requires 'x1', 'y1', 'x2', 'y2' params")
        px1, py1 = self._normalized_to_pixels(float(x1), float(y1))
        px2, py2 = self._normalized_to_pixels(float(x2), float(y2))
        duration = int(params.get("duration_ms", 300))
        self._adb.swipe(px1, py1, px2, py2, duration)

