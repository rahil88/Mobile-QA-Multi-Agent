"""Runner module - orchestrates the Planner-Executor-Supervisor loop."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from qualgent.agent.executor import Executor
from qualgent.agent.planner import Planner, PlannerError
from qualgent.agent.supervisor import Supervisor, SupervisorError
from qualgent.agent.types import (
    Action,
    ActionType,
    ErrorType,
    Observation,
    StepResult,
    TestCase,
    TestResult,
    TestStatus,
)
from qualgent.llm.gemini_client import GeminiClient
from qualgent.llm.openai_client import OpenAIClient
from qualgent.tools.adb_controller import AdbController, AdbError

__all__ = ["Runner", "RunReport"]

# Defaults for execution budgets
MAX_ITERATIONS = 20
DEFAULT_MAX_RETRIES_PER_STEP = 5
DEFAULT_MAX_SCROLLS_PER_STEP = 3


class RunReport:
    """Collects and formats test run results."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.results: list[TestResult] = []
        self.start_time = datetime.now()
        self.end_time: datetime | None = None

    def add_result(self, result: TestResult) -> None:
        """Add a test result."""
        self.results.append(result)

    def finalize(self) -> None:
        """Mark the run as complete."""
        self.end_time = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "run_dir": str(self.run_dir),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else None
            ),
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == TestStatus.PASSED),
                "failed": sum(1 for r in self.results if r.status == TestStatus.FAILED),
                "errors": sum(1 for r in self.results if r.status == TestStatus.ERROR),
            },
            "results": [self._result_to_dict(r) for r in self.results],
        }

    def _result_to_dict(self, result: TestResult) -> dict[str, Any]:
        """Convert a TestResult to dict."""
        return {
            "test_id": result.test_case.id,
            "test_name": result.test_case.name,
            "should_pass": result.test_case.should_pass,
            "status": result.status.value,
            "duration_seconds": result.duration_seconds,
            "error_message": result.error_message,
            "verdict": (
                {
                    "status": result.verdict.status.value,
                    "evidence": result.verdict.evidence,
                    "expected_vs_actual": result.verdict.expected_vs_actual,
                    "confidence": result.verdict.confidence,
                }
                if result.verdict
                else None
            ),
            "screenshots": [str(p) for p in result.screenshots],
            "step_count": len(result.steps),
        }

    def save(self) -> Path:
        """Save report to JSON file."""
        report_path = self.run_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return report_path

    def print_summary(self) -> None:
        """Print a console summary."""
        data = self.to_dict()
        summary = data["summary"]

        print("\n" + "=" * 60)
        print("QA TEST RUN SUMMARY")
        print("=" * 60)
        print(f"Total: {summary['total']} | "
              f"Passed: {summary['passed']} | "
              f"Failed: {summary['failed']} | "
              f"Errors: {summary['errors']}")
        print("-" * 60)

        for result in data["results"]:
            status_icon = {
                "passed": "✓",
                "failed": "✗",
                "error": "!",
            }.get(result["status"], "?")

            expected = "should pass" if result["should_pass"] else "should fail"
            actual = result["status"]

            # Check if the outcome matches expectation
            matched = (result["should_pass"] and actual == "passed") or \
                      (not result["should_pass"] and actual == "failed")
            match_str = "(as expected)" if matched else "(UNEXPECTED)"

            print(f"  [{status_icon}] {result['test_id']}: {result['test_name']}")
            print(f"      {expected} -> {actual} {match_str}")
            if result.get("error_message"):
                print(f"      Error: {result['error_message'][:100]}")

        print("=" * 60)
        print(f"Report saved to: {self.run_dir / 'report.json'}")
        print()


class Runner:
    """Orchestrates the test execution loop.

    Parameters
    ----------
    adb
        AdbController for device interaction.
    gemini
        GeminiClient for LLM calls.
    run_dir
        Directory to store run artifacts (screenshots, report).
    package
        App package name.
    """

    def __init__(
        self,
        adb: AdbController,
        llm_client: GeminiClient | OpenAIClient,
        run_dir: Path,
        package: str = "md.obsidian",
        fresh: bool = False,
        max_retries_per_step: int = DEFAULT_MAX_RETRIES_PER_STEP,
        max_scrolls_per_step: int = DEFAULT_MAX_SCROLLS_PER_STEP,
    ) -> None:
        self._adb = adb
        self._llm_client = llm_client
        self._run_dir = run_dir
        self._package = package
        self._fresh = fresh
        self._max_retries = max_retries_per_step
        self._max_scrolls = max_scrolls_per_step

        self._executor = Executor.from_adb(adb)
        self._planner = Planner(llm_client)
        self._supervisor = Supervisor(llm_client)
        self._report = RunReport(run_dir)

    def run_suite(self, tests: list[TestCase]) -> RunReport:
        """Run all tests in the suite.

        Parameters
        ----------
        tests
            List of test cases to run.

        Returns
        -------
        RunReport
            Complete run report.
        """
        print(f"\nStarting test run with {len(tests)} tests")
        print(f"Run directory: {self._run_dir}")
        print("-" * 40)

        for i, test in enumerate(tests):
            # Only clear app data on first test (sequential mode)
            is_first_test = (i == 0)
            result = self.run_test(test, fresh_start=is_first_test and self._fresh)
            self._report.add_result(result)

        self._report.finalize()
        self._report.save()
        self._report.print_summary()

        return self._report

    def _capture_observation(
        self,
        screenshot_path: Path,
        previous_action: Action | None = None,
        previous_result: StepResult | None = None,
        attempted_actions: list[str] | None = None,
    ) -> Observation:
        """Capture current screen observation (screenshot + UI texts)."""
        # Get UI texts from screen
        try:
            ui_texts = self._adb.dump_ui_texts()
        except AdbError:
            ui_texts = []

        # Get current activity (optional)
        try:
            activity = self._adb.get_current_activity()
        except AdbError:
            activity = ""

        return Observation(
            screenshot_path=screenshot_path,
            ui_texts=ui_texts,
            activity=activity,
            previous_action=previous_action,
            previous_result=previous_result,
            attempted_actions=attempted_actions or [],
        )

    def run_test(self, test: TestCase, *, fresh_start: bool = False) -> TestResult:
        """Run a single test case with observation-driven planning and recovery.

        Parameters
        ----------
        test
            The test case to run.
        fresh_start
            If True, clear app data before starting this test.
            In sequential mode, only the first test gets fresh_start=True.

        Returns
        -------
        TestResult
            Result of the test.
        """
        print(f"\n[TEST] {test.id}: {test.name}")
        print(f"  Description: {test.description[:80]}...")

        start_time = datetime.now()
        test_dir = self._run_dir / "screenshots" / test.id
        test_dir.mkdir(parents=True, exist_ok=True)

        steps: list[StepResult] = []
        screenshots: list[Path] = []
        action_history: list[str] = []

        try:
            # Setup: force stop and optionally clear data, then launch app
            self._adb.force_stop(self._package)
            if fresh_start:
                print("  [Setup] Clearing app data for fresh start...")
                self._adb.clear_app_data(self._package)
            AdbController.wait(1.0)
            self._adb.launch_app(self._package)
            AdbController.wait(2.0)  # Wait for app to load

            # Take initial screenshot
            initial_screenshot = test_dir / "000_initial.png"
            self._adb.take_screenshot(initial_screenshot)
            screenshots.append(initial_screenshot)

            # Main execution loop with observation-driven planning
            iteration = 0
            is_complete = False
            previous_action: Action | None = None
            previous_result: StepResult | None = None
            attempted_this_step: list[str] = []
            retry_count = 0
            scroll_count = 0
            back_tried = False
            relaunch_tried = False

            while iteration < MAX_ITERATIONS and not is_complete:
                iteration += 1
                current_screenshot = screenshots[-1]

                # Add delay between steps to avoid rate limits (especially for OpenAI)
                if iteration > 1:
                    import time
                    time.sleep(1.0)  # 1 second delay between steps

                print(f"  [Step {iteration}] Capturing observation and planning...")

                # Capture observation (screenshot + UI dump)
                observation = self._capture_observation(
                    screenshot_path=current_screenshot,
                    previous_action=previous_action,
                    previous_result=previous_result,
                    attempted_actions=attempted_this_step,
                )

                # Log visible UI texts for debugging
                if observation.ui_texts:
                    print(f"    UI texts: {observation.ui_texts[:5]}{'...' if len(observation.ui_texts) > 5 else ''}")

                # Interim verification: check if goal is already achieved every 3 steps
                if iteration > 0 and iteration % 3 == 0:
                    try:
                        interim_verdict = self._supervisor.verify_step(
                            expected_result=test.expected_result,
                            screenshot_path=current_screenshot,
                            ui_texts=observation.ui_texts,
                        )
                        if interim_verdict.status == TestStatus.PASSED and interim_verdict.confidence > 0.8:
                            print(f"    [Supervisor] Goal achieved early! (confidence: {interim_verdict.confidence:.0%})")
                            is_complete = True
                            break
                    except SupervisorError:
                        pass  # Continue if interim check fails

                # Get plan from planner using observation
                try:
                    plan = self._planner.plan_next_action(
                        test_goal=test.description,
                        observation=observation,
                        previous_actions=action_history[-5:],
                        step_context=f"Expected: {test.expected_result}",
                    )
                except PlannerError as e:
                    print(f"    Planner error: {e}")
                    break

                if plan.is_complete:
                    print("  [Planner] Goal achieved!")
                    is_complete = True
                    break

                if not plan.actions:
                    print("  [Planner] No actions to take")
                    break

                # Execute the planned action (one at a time now)
                action = plan.actions[0]
                action_key = f"{action.action_type.value}:{action.params}"

                print(f"    Executing: {action.action_type.value} - {action.description}")
                action_history.append(f"{action.action_type.value}: {action.description}")

                result = self._executor.execute(action)
                steps.append(result)
                previous_action = action
                previous_result = result

                if result.success:
                    # Success - reset recovery counters
                    attempted_this_step = []
                    retry_count = 0
                    scroll_count = 0
                    back_tried = False
                    relaunch_tried = False
                    AdbController.wait(0.5)
                else:
                    # Failure - try recovery
                    print(f"    Action failed: {result.error_message}")
                    attempted_this_step.append(action_key)
                    retry_count += 1

                    if retry_count >= self._max_retries:
                        print(f"    [Recovery] Max retries ({self._max_retries}) exceeded")

                        # Escalation ladder
                        if result.error_type == ErrorType.ELEMENT_NOT_FOUND and scroll_count < self._max_scrolls:
                            print(f"    [Recovery] Trying scroll down (attempt {scroll_count + 1}/{self._max_scrolls})")
                            self._adb.swipe(540, 1500, 540, 500, 300)
                            scroll_count += 1
                            retry_count = 0  # Reset retry count after scroll
                            AdbController.wait(0.5)
                        elif not back_tried:
                            print("    [Recovery] Trying BACK button")
                            self._adb.back()
                            back_tried = True
                            retry_count = 0
                            AdbController.wait(0.5)
                        elif not relaunch_tried:
                            print("    [Recovery] Relaunching app")
                            self._adb.relaunch_app(self._package)
                            relaunch_tried = True
                            retry_count = 0
                            AdbController.wait(2.0)
                        else:
                            print("    [Recovery] All recovery options exhausted")
                            break

                # Take screenshot after action
                step_screenshot = test_dir / f"{iteration:03d}_step.png"
                self._adb.take_screenshot(step_screenshot)
                screenshots.append(step_screenshot)

            # Final verification by supervisor
            print("  [Supervisor] Verifying final state...")
            final_screenshot = screenshots[-1]

            # Capture final observation for supervisor
            final_observation = self._capture_observation(screenshot_path=final_screenshot)

            try:
                verdict = self._supervisor.verify_test_completion(
                    test_goal=test.description,
                    expected_result=test.expected_result,
                    final_screenshot=final_screenshot,
                    action_history=action_history,
                    ui_texts=final_observation.ui_texts,
                )
            except SupervisorError as e:
                print(f"  [Supervisor] Error: {e}")
                verdict = None

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Determine final status
            if verdict:
                status = verdict.status
                print(f"  [Result] {status.value.upper()} (confidence: {verdict.confidence:.0%})")
            else:
                status = TestStatus.ERROR
                print("  [Result] ERROR - could not determine verdict")

            return TestResult(
                test_case=test,
                status=status,
                verdict=verdict,
                steps=steps,
                screenshots=screenshots,
                duration_seconds=duration,
            )

        except AdbError as e:
            duration = (datetime.now() - start_time).total_seconds()
            print(f"  [Result] ERROR - ADB failure: {e}")
            return TestResult(
                test_case=test,
                status=TestStatus.ERROR,
                steps=steps,
                screenshots=screenshots,
                duration_seconds=duration,
                error_message=str(e),
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            print(f"  [Result] ERROR - Unexpected: {e}")
            return TestResult(
                test_case=test,
                status=TestStatus.ERROR,
                steps=steps,
                screenshots=screenshots,
                duration_seconds=duration,
                error_message=str(e),
            )


def load_suite(suite_path: Path) -> tuple[str, list[TestCase]]:
    """Load a test suite from YAML file.

    Returns
    -------
    tuple
        (app_package, list of TestCase)
    """
    with open(suite_path) as f:
        data = yaml.safe_load(f)

    package = data.get("app_package", "md.obsidian")
    tests: list[TestCase] = []

    for test_data in data.get("tests", []):
        tests.append(
            TestCase(
                id=test_data["id"],
                name=test_data["name"],
                description=test_data["description"].strip(),
                expected_result=test_data["expected_result"].strip(),
                should_pass=test_data.get("should_pass", True),
            )
        )

    return package, tests


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run QA tests on an Android app using ADB + Gemini",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="Path to test suite YAML file",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="emulator-5554",
        help="ADB device serial (default: emulator-5554)",
    )
    parser.add_argument(
        "--package",
        type=str,
        default=None,
        help="App package name (overrides suite file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for run artifacts (default: runs/<timestamp>)",
    )
    parser.add_argument(
        "--test-id",
        action="append",
        default=None,
        help="Run only a specific test id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List tests in the suite and exit.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openai"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear app data before each test (fresh install state)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use. Defaults: gemini=gemini-2.0-flash, openai=gpt-5-mini",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES_PER_STEP,
        help=f"Max retries per step before escalating (default: {DEFAULT_MAX_RETRIES_PER_STEP})",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=DEFAULT_MAX_SCROLLS_PER_STEP,
        help=f"Max scroll attempts when element not found (default: {DEFAULT_MAX_SCROLLS_PER_STEP})",
    )

    args = parser.parse_args()

    # Setup run directory
    if args.output:
        run_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load test suite
    suite_package, tests = load_suite(args.suite)
    package = args.package or suite_package

    # List tests and exit if requested
    if args.list_tests:
        print(f"Loaded {len(tests)} tests from {args.suite}")
        for t in tests:
            exp = "should_pass" if t.should_pass else "should_fail"
            print(f"  - {t.id}: {t.name} ({exp})")
        sys.exit(0)

    # Filter to specific test(s) if requested
    if args.test_id:
        wanted = set(args.test_id)
        tests = [t for t in tests if t.id in wanted]
        if not tests:
            print(f"ERROR: None of the requested --test-id values were found: {sorted(wanted)}")
            sys.exit(1)

    print(f"Loaded {len(tests)} tests from {args.suite}")
    print(f"Target device: {args.serial}")
    print(f"App package: {package}")

    # Initialize components
    adb = AdbController(device_serial=args.serial, timeout_s=30.0)

    # Verify device and package
    if not adb.is_package_installed(package):
        print(f"ERROR: Package {package} is not installed on {args.serial}")
        sys.exit(1)

    # Initialize LLM client based on provider
    if args.provider == "openai":
        model = args.model or "gpt-5-mini"
        llm_client = OpenAIClient(model=model)
        print(f"Using provider: OpenAI, model: {model}")
    else:
        model = args.model or "gemini-2.0-flash"
        llm_client = GeminiClient(model=model)
        print(f"Using provider: Gemini, model: {model}")

    # Run tests
    runner = Runner(
        adb,
        llm_client,
        run_dir,
        package,
        fresh=args.fresh,
        max_retries_per_step=args.max_retries,
        max_scrolls_per_step=args.max_scrolls,
    )
    report = runner.run_suite(tests)

    # Exit with error code if any tests had unexpected outcomes
    unexpected = 0
    for result in report.results:
        expected_pass = result.test_case.should_pass
        actual_pass = result.status == TestStatus.PASSED
        if expected_pass != actual_pass:
            unexpected += 1

    if unexpected > 0:
        print(f"\n{unexpected} test(s) had unexpected outcomes!")

    sys.exit(0)


if __name__ == "__main__":
    main()

