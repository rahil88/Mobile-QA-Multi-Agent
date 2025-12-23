"""ADB controller module for wrapping adb subprocess calls."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

__all__ = ["AdbController", "AdbError"]


class AdbError(Exception):
    """Raised when an ADB command fails."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class AdbController:
    """Wrapper around ADB (Android Debug Bridge) subprocess calls.

    Parameters
    ----------
    device_serial
        Optional device/emulator serial. When provided, all commands
        are prefixed with ``adb -s <serial>``.
    adb_path
        Path or name of the adb executable. Defaults to ``"adb"``.
    timeout_s
        Timeout in seconds for each subprocess call. ``None`` means no timeout.
    cwd
        Working directory for subprocess calls. ``None`` uses the current directory.
    """

    def __init__(
        self,
        device_serial: str | None = None,
        *,
        adb_path: str = "adb",
        timeout_s: float | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._device_serial = device_serial
        self._adb_path = adb_path
        self._timeout_s = timeout_s
        self._cwd = cwd

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _base_cmd(self) -> list[str]:
        """Return the base command prefix (adb or adb -s <serial>)."""
        if self._device_serial:
            return [self._adb_path, "-s", self._device_serial]
        return [self._adb_path]

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        capture_output: bool = True,
        text: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run an ADB command via subprocess.

        Raises
        ------
        AdbError
            If check is True and the command returns a non-zero exit code.
        """
        cmd = [*self._base_cmd(), *args]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=capture_output,
                text=text,
                timeout=self._timeout_s,
                cwd=self._cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(
                f"Command timed out after {self._timeout_s}s: {' '.join(cmd)}"
            ) from exc

        if check and result.returncode != 0:
            raise AdbError(
                f"ADB command failed (exit {result.returncode}): {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    def _run_bytes(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run an ADB command and capture binary output."""
        cmd = [*self._base_cmd(), *args]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=False,
                timeout=self._timeout_s,
                cwd=self._cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(
                f"Command timed out after {self._timeout_s}s: {' '.join(cmd)}"
            ) from exc

        if check and result.returncode != 0:
            raise AdbError(
                f"ADB command failed (exit {result.returncode}): {' '.join(cmd)}\n"
                f"stderr: {result.stderr.decode(errors='replace')}",
                returncode=result.returncode,
                stdout=None,
                stderr=result.stderr.decode(errors="replace"),
            )
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def take_screenshot(self, output_path: str | Path = "screenshot.png") -> str:
        """Capture the device screen and save it to *output_path*.

        Parameters
        ----------
        output_path
            File path where the PNG screenshot will be saved.
            Defaults to ``"screenshot.png"``.

        Returns
        -------
        str
            Confirmation message including the saved path.
        """
        result = self._run_bytes(["exec-out", "screencap", "-p"])
        path = Path(output_path)
        path.write_bytes(result.stdout)
        return f"Saved screenshot to {path}"

    def tap_coordinates(self, x: int, y: int) -> str:
        """Tap the screen at the given coordinates.

        Parameters
        ----------
        x
            Horizontal pixel coordinate.
        y
            Vertical pixel coordinate.

        Returns
        -------
        str
            Confirmation message.
        """
        self._run(["shell", "input", "tap", str(x), str(y)])
        return f"Tapped at ({x}, {y})"

    def type_text(self, text: str) -> str:
        """Type text on the device.

        Spaces are encoded as ``%s`` per ADB input text convention.
        Other shell-unsafe characters are escaped as needed.

        Parameters
        ----------
        text
            The string to type.

        Returns
        -------
        str
            Confirmation message.
        """
        # ADB `input text` requires spaces to be encoded as %s
        encoded = text.replace(" ", "%s")
        # Escape shell metacharacters that could cause issues.
        # Backslashes must be escaped first to avoid double-escaping.
        encoded = encoded.replace("\\", "\\\\")
        for char in ("'", '"', "`", "$", "(", ")", "&", "|", ";", "<", ">"):
            encoded = encoded.replace(char, f"\\{char}")
        self._run(["shell", "input", "text", encoded])
        return f"Typed text: {text!r}"

    def send_key_event(self, key_code: int) -> str:
        """Send a key event to the device.

        Parameters
        ----------
        key_code
            Android KeyEvent code (e.g., 3 for HOME, 4 for BACK, 66 for ENTER).

        Returns
        -------
        str
            Confirmation message.
        """
        self._run(["shell", "input", "keyevent", str(key_code)])
        return f"Sent key event: {key_code}"

    # ------------------------------------------------------------------ #
    # App lifecycle controls
    # ------------------------------------------------------------------ #

    def launch_app(self, package: str) -> str:
        """Launch an app using monkey (works without knowing the activity name).

        Parameters
        ----------
        package
            The package name (e.g., ``"md.obsidian"``).

        Returns
        -------
        str
            Confirmation message.
        """
        self._run([
            "shell", "monkey",
            "-p", package,
            "-c", "android.intent.category.LAUNCHER",
            "1",
        ])
        return f"Launched app: {package}"

    def force_stop(self, package: str) -> str:
        """Force stop an app.

        Parameters
        ----------
        package
            The package name to stop.

        Returns
        -------
        str
            Confirmation message.
        """
        self._run(["shell", "am", "force-stop", package])
        return f"Force stopped: {package}"

    def clear_app_data(self, package: str) -> str:
        """Clear all app data (reset to fresh install state).

        Parameters
        ----------
        package
            The package name whose data should be cleared.

        Returns
        -------
        str
            Confirmation message.
        """
        self._run(["shell", "pm", "clear", package])
        return f"Cleared data for: {package}"

    def is_package_installed(self, package: str) -> bool:
        """Check if a package is installed on the device.

        Parameters
        ----------
        package
            The package name to check.

        Returns
        -------
        bool
            True if installed, False otherwise.
        """
        result = self._run(["shell", "pm", "list", "packages", package], check=False)
        return f"package:{package}" in result.stdout

    # ------------------------------------------------------------------ #
    # Screen / input helpers
    # ------------------------------------------------------------------ #

    def get_screen_size(self) -> tuple[int, int]:
        """Get the device screen resolution.

        Returns
        -------
        tuple[int, int]
            (width, height) in pixels.
        """
        result = self._run(["shell", "wm", "size"])
        # Output format: "Physical size: 1080x1920"
        for line in result.stdout.strip().splitlines():
            if ":" in line:
                size_str = line.split(":")[-1].strip()
                if "x" in size_str:
                    w, h = size_str.split("x")
                    return int(w), int(h)
        raise AdbError(f"Could not parse screen size from: {result.stdout}")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> str:
        """Perform a swipe gesture.

        Parameters
        ----------
        x1, y1
            Starting coordinates.
        x2, y2
            Ending coordinates.
        duration_ms
            Duration of the swipe in milliseconds.

        Returns
        -------
        str
            Confirmation message.
        """
        self._run([
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        ])
        return f"Swiped from ({x1}, {y1}) to ({x2}, {y2}) in {duration_ms}ms"

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> str:
        """Perform a long press at the given coordinates.

        Parameters
        ----------
        x, y
            Coordinates to press.
        duration_ms
            Duration of the press in milliseconds.

        Returns
        -------
        str
            Confirmation message.
        """
        # Long press is a swipe from (x,y) to (x,y) with a duration
        self._run([
            "shell", "input", "swipe",
            str(x), str(y), str(x), str(y), str(duration_ms),
        ])
        return f"Long pressed at ({x}, {y}) for {duration_ms}ms"

    @staticmethod
    def wait(seconds: float) -> str:
        """Wait for a specified number of seconds.

        This is a simple sleep, useful for waiting for UI transitions.

        Parameters
        ----------
        seconds
            Number of seconds to wait.

        Returns
        -------
        str
            Confirmation message.
        """
        time.sleep(seconds)
        return f"Waited {seconds}s"

    def tap_text(self, text: str, *, partial: bool = False) -> str:
        """Find an element by its visible text and tap on it.

        Uses UI Automator to dump the view hierarchy and find the element.

        Parameters
        ----------
        text
            The visible text to search for.
        partial
            If True, match elements containing the text (substring match).
            If False (default), match exact text only.

        Returns
        -------
        str
            Confirmation message.

        Raises
        ------
        AdbError
            If the element is not found or tap fails.
        """
        import re
        import xml.etree.ElementTree as ET

        # Dump UI hierarchy to device
        self._run(["shell", "uiautomator", "dump", "/sdcard/ui_dump.xml"])

        # Pull the dump file
        result = self._run(["shell", "cat", "/sdcard/ui_dump.xml"])
        xml_content = result.stdout

        # Parse XML and find element
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            raise AdbError(f"Failed to parse UI dump: {exc}")

        # Collect ALL matching elements (includes hint/placeholder text)
        matches = []
        for elem in root.iter("node"):
            elem_text = elem.get("text", "")
            content_desc = elem.get("content-desc", "")
            hint = elem.get("hint", "")  # Placeholder text for input fields

            if partial:
                text_lower = text.lower()
                if (text_lower in elem_text.lower() or 
                    text_lower in content_desc.lower() or
                    text_lower in hint.lower()):
                    matches.append(elem)
            else:
                if elem_text == text or content_desc == text or hint == text:
                    matches.append(elem)

        if not matches:
            raise AdbError(f"Element with text '{text}' not found on screen")

        # Prefer interactive elements (buttons, inputs) over static text
        def element_priority(elem: ET.Element) -> int:
            """Higher score = better match for tapping."""
            score = 0
            # Clickable elements are highest priority
            if elem.get("clickable") == "true":
                score += 100
            # Check element class for interactive types
            elem_class = elem.get("class", "")
            if "Button" in elem_class:
                score += 50
            if "EditText" in elem_class or "Input" in elem_class:
                score += 50
            if "CheckBox" in elem_class or "Switch" in elem_class or "Radio" in elem_class:
                score += 40
            # Focusable elements are somewhat interactive
            if elem.get("focusable") == "true":
                score += 10
            return score

        # Sort by priority (highest first) and pick the best match
        matches.sort(key=element_priority, reverse=True)
        found_element = matches[0]

        # Extract bounds and calculate center
        bounds_str = found_element.get("bounds", "")
        # bounds format: "[left,top][right,bottom]"
        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
        if not match:
            raise AdbError(f"Could not parse bounds: {bounds_str}")

        left, top, right, bottom = map(int, match.groups())
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        # Tap the center of the element
        self._run(["shell", "input", "tap", str(center_x), str(center_y)])
        return f"Tapped on element with text '{text}' at ({center_x}, {center_y})"

    def tap_and_type(
        self,
        target_text: str,
        input_text: str,
        *,
        partial: bool = False,
        delay_ms: int = 300,
    ) -> str:
        """Tap on an element (to focus it) and then type text.

        This is a compound action useful for input fields where you need
        to first tap the field to focus it, then type into it.

        Parameters
        ----------
        target_text
            The visible text of the element to tap (e.g., placeholder text like "My vault").
        input_text
            The text to type after tapping.
        partial
            If True, match elements containing the target_text (substring match).
        delay_ms
            Delay in milliseconds between tap and typing.

        Returns
        -------
        str
            Confirmation message.

        Raises
        ------
        AdbError
            If the element is not found or actions fail.
        """
        import time

        # First tap on the target element to focus it
        tap_result = self.tap_text(target_text, partial=partial)

        # Small delay to let the keyboard appear and field focus
        time.sleep(delay_ms / 1000.0)

        # Clear existing text by moving to end and deleting backwards
        # This is more reliable than Ctrl+A on Android
        self._run(["shell", "input", "keyevent", "123"])  # KEYCODE_MOVE_END
        time.sleep(0.05)
        # Delete existing text - send multiple DEL key events in one command
        # 30 KEYCODE_DEL (67) events to clear typical field content
        self._run(["shell", "input", "keyevent"] + ["67"] * 30)
        time.sleep(0.15)

        # Now type the new text
        type_result = self.type_text(input_text)

        return f"{tap_result}; Cleared existing text; {type_result}"

    def dump_ui_texts(self) -> list[str]:
        """Extract all visible text labels from the current screen.

        Uses UI Automator to dump the view hierarchy and extract text.
        Also extracts hint (placeholder) text from input fields.

        Returns
        -------
        list[str]
            List of visible text labels (text, content-desc, and hint values).
        """
        import xml.etree.ElementTree as ET

        # Dump UI hierarchy to device
        self._run(["shell", "uiautomator", "dump", "/sdcard/ui_dump.xml"])

        # Pull the dump file
        result = self._run(["shell", "cat", "/sdcard/ui_dump.xml"])
        xml_content = result.stdout

        # Parse XML and extract texts
        texts: list[str] = []
        try:
            root = ET.fromstring(xml_content)
            for elem in root.iter("node"):
                elem_text = elem.get("text", "").strip()
                content_desc = elem.get("content-desc", "").strip()
                hint = elem.get("hint", "").strip()
                if elem_text:
                    texts.append(elem_text)
                if content_desc and content_desc != elem_text:
                    texts.append(content_desc)
                # Include hint (placeholder text) for input fields
                if hint and hint != elem_text and hint != content_desc:
                    texts.append(hint)
        except ET.ParseError:
            pass  # Return empty list on parse failure

        return texts

    def exists_text(self, text: str, *, partial: bool = False) -> bool:
        """Check if text exists on the current screen.

        Parameters
        ----------
        text
            The text to search for.
        partial
            If True, match elements containing the text (substring match).

        Returns
        -------
        bool
            True if text is found, False otherwise.
        """
        ui_texts = self.dump_ui_texts()
        if partial:
            text_lower = text.lower()
            return any(text_lower in t.lower() for t in ui_texts)
        return text in ui_texts

    def scroll_until_text(
        self,
        text: str,
        *,
        direction: str = "down",
        max_swipes: int = 5,
        partial: bool = False,
    ) -> str:
        """Scroll the screen until the specified text is visible.

        Parameters
        ----------
        text
            The text to search for.
        direction
            Direction to scroll: "up", "down", "left", "right".
        max_swipes
            Maximum number of swipe attempts.
        partial
            If True, use substring matching for text.

        Returns
        -------
        str
            Confirmation message.

        Raises
        ------
        AdbError
            If text is not found after max_swipes.
        """
        # Get screen dimensions for scroll calculations
        width, height = self.get_screen_size()
        cx, cy = width // 2, height // 2

        # Define swipe vectors based on direction
        swipe_map = {
            "down": (cx, int(height * 0.7), cx, int(height * 0.3)),
            "up": (cx, int(height * 0.3), cx, int(height * 0.7)),
            "left": (int(width * 0.7), cy, int(width * 0.3), cy),
            "right": (int(width * 0.3), cy, int(width * 0.7), cy),
        }

        if direction not in swipe_map:
            raise AdbError(f"Invalid scroll direction: {direction}")

        x1, y1, x2, y2 = swipe_map[direction]

        for attempt in range(max_swipes):
            if self.exists_text(text, partial=partial):
                return f"Found text '{text}' after {attempt} scroll(s)"

            self.swipe(x1, y1, x2, y2, 300)
            self.wait(0.5)

        # Final check after last swipe
        if self.exists_text(text, partial=partial):
            return f"Found text '{text}' after {max_swipes} scroll(s)"

        raise AdbError(f"Text '{text}' not found after {max_swipes} scroll(s) {direction}")

    def back(self) -> str:
        """Press the back button.

        Returns
        -------
        str
            Confirmation message.
        """
        return self.send_key_event(4)  # KEYCODE_BACK

    def home(self) -> str:
        """Press the home button.

        Returns
        -------
        str
            Confirmation message.
        """
        return self.send_key_event(3)  # KEYCODE_HOME

    def relaunch_app(self, package: str) -> str:
        """Force stop and relaunch an app.

        Parameters
        ----------
        package
            The package name to relaunch.

        Returns
        -------
        str
            Confirmation message.
        """
        self.force_stop(package)
        self.wait(0.5)
        self.launch_app(package)
        return f"Relaunched app: {package}"

    def get_current_activity(self) -> str:
        """Get the current foreground activity.

        Returns
        -------
        str
            The current activity name (e.g., "com.example/.MainActivity").
        """
        result = self._run(["shell", "dumpsys", "activity", "activities"])
        # Parse the output to find the focused activity
        for line in result.stdout.splitlines():
            if "mResumedActivity" in line or "mFocusedActivity" in line:
                # Extract activity name from the line
                parts = line.split()
                for part in parts:
                    if "/" in part and "." in part:
                        return part.strip()
        return ""

