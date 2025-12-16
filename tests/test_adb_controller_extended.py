"""Unit tests for extended AdbController methods (no real ADB required)."""

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
# App lifecycle tests
# ---------------------------------------------------------------------------


def test_launch_app(controller: AdbController) -> None:
    """launch_app uses monkey to launch the app."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.launch_app("md.obsidian")

    cmd = mock_run.call_args[0][0]
    assert cmd == [
        "adb", "shell", "monkey",
        "-p", "md.obsidian",
        "-c", "android.intent.category.LAUNCHER",
        "1",
    ]
    assert "md.obsidian" in result


def test_force_stop(controller: AdbController) -> None:
    """force_stop sends am force-stop command."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.force_stop("md.obsidian")

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "am", "force-stop", "md.obsidian"]
    assert "md.obsidian" in result


def test_clear_app_data(controller: AdbController) -> None:
    """clear_app_data sends pm clear command."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.clear_app_data("md.obsidian")

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "pm", "clear", "md.obsidian"]
    assert "md.obsidian" in result


def test_is_package_installed_true(controller: AdbController) -> None:
    """is_package_installed returns True when package is in list."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "package:md.obsidian\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = controller.is_package_installed("md.obsidian")

    assert result is True


def test_is_package_installed_false(controller: AdbController) -> None:
    """is_package_installed returns False when package not in list."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = controller.is_package_installed("com.fake.app")

    assert result is False


# ---------------------------------------------------------------------------
# Screen / input helper tests
# ---------------------------------------------------------------------------


def test_get_screen_size(controller: AdbController) -> None:
    """get_screen_size parses wm size output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Physical size: 1080x1920\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        width, height = controller.get_screen_size()

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "wm", "size"]
    assert width == 1080
    assert height == 1920


def test_get_screen_size_with_override(controller: AdbController) -> None:
    """get_screen_size handles override size output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Physical size: 1080x1920\nOverride size: 720x1280\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        width, height = controller.get_screen_size()

    # Should return first valid size
    assert width == 1080
    assert height == 1920


def test_swipe(controller: AdbController) -> None:
    """swipe sends correct command with coordinates and duration."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.swipe(100, 200, 100, 800, 500)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "swipe", "100", "200", "100", "800", "500"]
    assert "100" in result and "800" in result


def test_swipe_default_duration(controller: AdbController) -> None:
    """swipe uses default duration if not specified."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller.swipe(0, 0, 100, 100)

    cmd = mock_run.call_args[0][0]
    assert cmd[-1] == "300"  # Default duration


def test_long_press(controller: AdbController) -> None:
    """long_press performs swipe from/to same point with duration."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = controller.long_press(500, 500, 1500)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "shell", "input", "swipe", "500", "500", "500", "500", "1500"]
    assert "1500ms" in result


def test_wait() -> None:
    """wait sleeps for specified time."""
    with patch("time.sleep") as mock_sleep:
        result = AdbController.wait(2.5)

    mock_sleep.assert_called_once_with(2.5)
    assert "2.5" in result


# ---------------------------------------------------------------------------
# Serial handling for new methods
# ---------------------------------------------------------------------------


def test_launch_app_with_serial(controller_with_serial: AdbController) -> None:
    """launch_app includes -s <serial> when serial is set."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller_with_serial.launch_app("md.obsidian")

    cmd = mock_run.call_args[0][0]
    assert cmd[:3] == ["adb", "-s", "emulator-5554"]


def test_swipe_with_serial(controller_with_serial: AdbController) -> None:
    """swipe includes -s <serial> when serial is set."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        controller_with_serial.swipe(0, 0, 100, 100)

    cmd = mock_run.call_args[0][0]
    assert cmd[:3] == ["adb", "-s", "emulator-5554"]

