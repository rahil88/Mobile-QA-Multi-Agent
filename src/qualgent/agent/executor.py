"""Executor module - translates planned actions to ADB commands."""

from __future__ import annotations

from pathlib import Path

from qualgent.agent.types import Action, ActionType, ErrorType, StepResult
from qualgent.tools.adb_controller import AdbController, AdbError

__all__ = ["Executor", "ExecutorError"]


class ExecutorError(Exception):
    """Raised when action execution fails."""

    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN) -> None:
        super().__init__(message)
        self.error_type = error_type


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
            Result of the action execution with structured error info.
        """
        try:
            self._execute_action(action)
            return StepResult(action=action, success=True, error_type=ErrorType.NONE)
        except ExecutorError as exc:
            return StepResult(
                action=action,
                success=False,
                error_type=exc.error_type,
                error_message=str(exc),
            )
        except AdbError as exc:
            # Determine error type from ADB error
            error_type = ErrorType.ADB_FAILURE
            msg = str(exc).lower()
            if "not found" in msg:
                error_type = ErrorType.ELEMENT_NOT_FOUND
            elif "timed out" in msg or "timeout" in msg:
                error_type = ErrorType.TIMEOUT
            return StepResult(
                action=action,
                success=False,
                error_type=error_type,
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

            case ActionType.TAP_TEXT:
                self._do_tap_text(action.params)

            case ActionType.SWIPE:
                self._do_swipe(action.params)

            case ActionType.TYPE_TEXT:
                text = action.params.get("text", "")
                if not text:
                    raise ExecutorError("type_text requires 'text' param", ErrorType.INVALID_PARAMS)
                print(f"      [TypeText] Typing: '{text}'")
                self._adb.type_text(text)

            case ActionType.TAP_AND_TYPE:
                target_text = action.params.get("target_text", "")
                input_text = action.params.get("input_text", "")
                partial = action.params.get("partial", False)
                if not target_text:
                    raise ExecutorError("tap_and_type requires 'target_text' param", ErrorType.INVALID_PARAMS)
                if not input_text:
                    raise ExecutorError("tap_and_type requires 'input_text' param", ErrorType.INVALID_PARAMS)
                print(f"      [TapAndType] Tapping '{target_text}' then typing: '{input_text}'")
                self._adb.tap_and_type(target_text, input_text, partial=partial)

            case ActionType.KEY_EVENT:
                key_code = action.params.get("key_code")
                if key_code is None:
                    raise ExecutorError("key_event requires 'key_code' param", ErrorType.INVALID_PARAMS)
                print(f"      [KeyEvent] Sending key: {key_code}")
                self._adb.send_key_event(int(key_code))

            case ActionType.BACK:
                print("      [Back] Pressing back button")
                self._adb.back()

            case ActionType.HOME:
                print("      [Home] Pressing home button")
                self._adb.home()

            case ActionType.LAUNCH_APP:
                package = action.params.get("package", "")
                if not package:
                    raise ExecutorError("launch_app requires 'package' param", ErrorType.INVALID_PARAMS)
                print(f"      [LaunchApp] Launching: {package}")
                self._adb.launch_app(package)

            case ActionType.FORCE_STOP:
                package = action.params.get("package", "")
                if not package:
                    raise ExecutorError("force_stop requires 'package' param", ErrorType.INVALID_PARAMS)
                print(f"      [ForceStop] Stopping: {package}")
                self._adb.force_stop(package)

            case ActionType.CLEAR_DATA:
                package = action.params.get("package", "")
                if not package:
                    raise ExecutorError("clear_data requires 'package' param", ErrorType.INVALID_PARAMS)
                print(f"      [ClearData] Clearing: {package}")
                self._adb.clear_app_data(package)

            case ActionType.RELAUNCH_APP:
                package = action.params.get("package", "")
                if not package:
                    raise ExecutorError("relaunch_app requires 'package' param", ErrorType.INVALID_PARAMS)
                print(f"      [RelaunchApp] Relaunching: {package}")
                self._adb.relaunch_app(package)

            case ActionType.SCROLL_UNTIL_TEXT:
                self._do_scroll_until_text(action.params)

            case ActionType.WAIT:
                seconds = action.params.get("seconds", 1.0)
                print(f"      [Wait] Waiting {seconds}s")
                AdbController.wait(float(seconds))

            case ActionType.SCREENSHOT:
                path = action.params.get("path", "screenshot.png")
                print(f"      [Screenshot] Saving to: {path}")
                self._adb.take_screenshot(path)

            case _:
                raise ExecutorError(f"Unknown action type: {action.action_type}", ErrorType.UNKNOWN)

    def _do_tap(self, params: dict) -> None:
        """Execute a tap action."""
        x = params.get("x")
        y = params.get("y")
        if x is None or y is None:
            raise ExecutorError("tap requires 'x' and 'y' params")
        px, py = self._normalized_to_pixels(float(x), float(y))
        print(f"      [Tap] normalized=({x:.2f}, {y:.2f}) -> pixels=({px}, {py})")
        self._adb.tap_coordinates(px, py)

    def _do_tap_text(self, params: dict) -> None:
        """Execute a tap_text action - find element by text and tap it."""
        text = params.get("text", "")
        if not text:
            raise ExecutorError("tap_text requires 'text' param")
        partial = params.get("partial", False)
        print(f"      [TapText] Looking for element with text: '{text}'")
        result = self._adb.tap_text(text, partial=partial)
        print(f"      {result}")

    def _do_swipe(self, params: dict) -> None:
        """Execute a swipe action."""
        x1 = params.get("x1")
        y1 = params.get("y1")
        x2 = params.get("x2")
        y2 = params.get("y2")
        if None in (x1, y1, x2, y2):
            raise ExecutorError("swipe requires 'x1', 'y1', 'x2', 'y2' params", ErrorType.INVALID_PARAMS)
        px1, py1 = self._normalized_to_pixels(float(x1), float(y1))
        px2, py2 = self._normalized_to_pixels(float(x2), float(y2))
        duration = int(params.get("duration_ms", 300))
        print(f"      [Swipe] ({px1}, {py1}) -> ({px2}, {py2}), duration={duration}ms")
        self._adb.swipe(px1, py1, px2, py2, duration)

    def _do_scroll_until_text(self, params: dict) -> None:
        """Execute a scroll_until_text action."""
        text = params.get("text", "")
        if not text:
            raise ExecutorError("scroll_until_text requires 'text' param", ErrorType.INVALID_PARAMS)
        direction = params.get("direction", "down")
        max_swipes = params.get("max_swipes", 5)
        partial = params.get("partial", False)
        print(f"      [ScrollUntilText] Looking for '{text}' by scrolling {direction} (max {max_swipes})")
        result = self._adb.scroll_until_text(text, direction=direction, max_swipes=max_swipes, partial=partial)
        print(f"      {result}")

