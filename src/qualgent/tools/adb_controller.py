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

