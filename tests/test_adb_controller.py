"""Unit tests for AdbController (no real ADB required)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qualgent.tools.adb_controller import AdbController, AdbError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller() -> AdbController:
    """Return an AdbController with no device serial."""
    return AdbController()


@pytest.fixture
def controller_with_serial() -> AdbController:
    """Return an AdbController with a specific device serial."""
    return AdbController(device_serial="emulator-5554")


# ---------------------------------------------------------------------------
# _base_cmd tests
# ---------------------------------------------------------------------------


def test_base_cmd_no_serial(controller: AdbController) -> None:
    """Without a serial, base command is just ['adb']."""
    assert controller._base_cmd() == ["adb"]


def test_base_cmd_with_serial(controller_with_serial: AdbController) -> None:
    """With a serial, base command includes -s <serial>."""
    assert controller_with_serial._base_cmd() == ["adb", "-s", "emulator-5554"]


# ---------------------------------------------------------------------------
# take_screenshot tests
# ---------------------------------------------------------------------------


def test_take_screenshot_saves_to_path(
    controller: AdbController, tmp_path: Path
) -> None:
    """take_screenshot writes PNG bytes and returns confirmation."""
    fake_png = b"\x89PNG\r\n\x1a\n...fake image data..."
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_png
    mock_result.stderr = b""

    screenshot_path = tmp_path / "screenshot.png"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.take_screenshot(screenshot_path)

    # Verify command construction
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "exec-out", "screencap", "-p"]

    # Verify file was written
    assert screenshot_path.exists()
    assert screenshot_path.read_bytes() == fake_png

    # Verify confirmation string
    assert "screenshot.png" in result
    assert "Saved screenshot" in result


def test_take_screenshot_with_serial(
    controller_with_serial: AdbController, tmp_path: Path
) -> None:
    """take_screenshot includes -s <serial> when serial is set."""
    fake_png = b"\x89PNG\r\n\x1a\n"
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_png
    mock_result.stderr = b""

    screenshot_path = tmp_path / "out.png"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller_with_serial.take_screenshot(screenshot_path)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "-s", "emulator-5554", "exec-out", "screencap", "-p"]


# ---------------------------------------------------------------------------
# tap_coordinates tests
# ---------------------------------------------------------------------------


def test_tap_coordinates(controller: AdbController) -> None:
    """tap_coordinates sends correct command."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.tap_coordinates(100, 200)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "tap", "100", "200"]
    assert result == "Tapped at (100, 200)"


def test_tap_coordinates_with_serial(controller_with_serial: AdbController) -> None:
    """tap_coordinates includes -s <serial> when serial is set."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller_with_serial.tap_coordinates(50, 75)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "-s", "emulator-5554", "shell", "input", "tap", "50", "75"]


# ---------------------------------------------------------------------------
# type_text tests
# ---------------------------------------------------------------------------


def test_type_text_encodes_spaces(controller: AdbController) -> None:
    """type_text converts spaces to %s."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.type_text("hello world")

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "text", "hello%sworld"]
    assert result == "Typed text: 'hello world'"


def test_type_text_no_spaces(controller: AdbController) -> None:
    """type_text with no spaces passes text directly."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller.type_text("nospaces")

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "text", "nospaces"]


def test_type_text_escapes_special_chars(controller: AdbController) -> None:
    """type_text escapes shell metacharacters."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller.type_text("test'quote")

    cmd = mock_run.call_args[0][0]
    # Single quote should be escaped with backslash
    assert cmd == ["adb", "shell", "input", "text", "test\\'quote"]


# ---------------------------------------------------------------------------
# send_key_event tests
# ---------------------------------------------------------------------------


def test_send_key_event_home(controller: AdbController) -> None:
    """send_key_event sends HOME key (3)."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.send_key_event(3)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "keyevent", "3"]
    assert result == "Sent key event: 3"


def test_send_key_event_back(controller: AdbController) -> None:
    """send_key_event sends BACK key (4)."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller.send_key_event(4)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "keyevent", "4"]


def test_send_key_event_enter(controller: AdbController) -> None:
    """send_key_event sends ENTER key (66)."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller.send_key_event(66)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "keyevent", "66"]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_adb_error_on_command_failure(controller: AdbController) -> None:
    """AdbError is raised when command fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "error: device not found"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(AdbError) as exc_info:
            controller.tap_coordinates(0, 0)

    assert exc_info.value.returncode == 1
    assert "device not found" in str(exc_info.value)


def test_timeout_raises_adb_error(controller: AdbController) -> None:
    """Timeout during command execution raises AdbError."""
    controller_with_timeout = AdbController(timeout_s=5.0)

    with patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["adb"], timeout=5)
    ):
        with pytest.raises(AdbError) as exc_info:
            controller_with_timeout.tap_coordinates(0, 0)

    assert "timed out" in str(exc_info.value).lower()

