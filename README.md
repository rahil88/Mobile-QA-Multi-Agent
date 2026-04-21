# Qualgent — Mobile QA Multi-Agent

An intelligent QA automation system for Android apps that uses vision AI and multi-agent orchestration to dynamically plan, execute, and verify test scenarios — no hard-coded test steps required.

## How It Works

Instead of scripting fixed UI interactions, Qualgent uses LLM agents to observe the actual screen state and decide what to do next:

1. **Planner** — analyzes a screenshot + visible UI texts and produces the next action
2. **Executor** — translates the planned action into ADB commands on the device
3. **Supervisor** — uses vision AI to verify whether the test goal was achieved

This makes tests resilient to minor UI changes and enables natural-language test definitions.

```
Test Suite (YAML)
       │
       ▼
  Runner.run_suite()
       │
       ├─ 1. Setup: force-stop + clear-data + launch
       ├─ 2. Observe: screenshot + dump UI texts + get activity
       ├─ 3. Plan: Planner(LLM) → Action
       ├─ 4. Execute: ADB command
       ├─ 5. Recover (on failure): retry → scroll → back → relaunch
       ├─ 6. Verify (every 3 steps): Supervisor early-exit check
       └─ 7. Final verdict: Supervisor → PASSED / FAILED + evidence
```

## Requirements

- Python 3.11+
- `adb` in your PATH with a connected Android device or emulator
- A Google Gemini API key and/or OpenAI API key

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/rahil88/Mobile-QA-Multi-Agent.git
cd Mobile-QA-Multi-Agent
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Usage

```bash
# Run all tests in a suite (Gemini, default emulator)
qualgent --suite src/qualgent/suites/obsidian_suite.yaml

# Target a specific device
qualgent --suite src/qualgent/suites/obsidian_suite.yaml --serial emulator-5554

# Use OpenAI instead of Gemini
qualgent --suite src/qualgent/suites/obsidian_suite.yaml --provider openai

# Run a single test by ID
qualgent --suite src/qualgent/suites/obsidian_suite.yaml --test-id T1_create_vault

# List available tests without running
qualgent --suite src/qualgent/suites/obsidian_suite.yaml --list-tests

# Clear app data before every test (fresh install state)
qualgent --suite src/qualgent/suites/obsidian_suite.yaml --fresh
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | required | Path to YAML test suite |
| `--serial` | `emulator-5554` | ADB device serial |
| `--package` | from suite | Override app package name |
| `--output` | `runs/<timestamp>` | Output directory for artifacts |
| `--test-id` | all tests | Run one specific test (repeatable) |
| `--list-tests` | — | Print tests and exit |
| `--provider` | `gemini` | LLM provider: `gemini` or `openai` |
| `--model` | provider default | Model name override |
| `--fresh` | false | Clear app data before each test |
| `--max-retries` | `5` | Retries per failed step |
| `--max-scrolls` | `3` | Scroll attempts before giving up |

## Writing Test Suites

Tests are defined in YAML with natural-language goals:

```yaml
app_package: md.obsidian

tests:
  - id: T1_create_vault
    name: Create and Enter Vault
    description: >
      Open the app, create a new local vault named 'InternVault',
      and confirm it opens successfully.
    expected_result: >
      The vault 'InternVault' is created and the main note list
      is visible.
    should_pass: true

  - id: T2_create_note
    name: Create a Note
    description: >
      Inside the vault, create a new note titled 'Meeting Notes'
      with the body text 'Daily Standup'.
    expected_result: >
      A note titled 'Meeting Notes' exists and contains 'Daily Standup'.
    should_pass: true
```

## Output

Each run produces artifacts under `runs/<timestamp>/`:

```
runs/20250421_120000/
├── report.json          # Summary with verdicts, confidence, and timing
└── screenshots/
    └── T1_create_vault/
        ├── 000_initial.png
        ├── 001_step.png
        └── ...
```

`report.json` includes per-test status, evidence from the Supervisor, confidence scores, step counts, and total run duration.

## Project Structure

```
src/qualgent/
├── agent/
│   ├── runner.py        # Orchestrates test execution and report generation
│   ├── planner.py       # LLM-based action planning
│   ├── supervisor.py    # LLM-based verdict verification
│   ├── executor.py      # Translates actions to ADB commands
│   └── types.py         # Shared data structures
├── llm/
│   ├── gemini_client.py # Google Gemini API client (vision + retry)
│   └── openai_client.py # OpenAI API client (vision + retry)
├── tools/
│   └── adb_controller.py # Android Debug Bridge wrapper
└── suites/
    └── obsidian_suite.yaml # Example test suite
tests/                   # Unit and integration tests
```

## Running Tests

```bash
pytest tests/ -v
```

## Design Notes

The architecture follows the [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) pattern, which maps naturally to the QA workflow: plan an action, execute it, verify the result. Key design choices:

- **Observation-driven planning**: The Planner receives both a screenshot and a structured list of visible UI texts, grounding its decisions in what is actually on screen.
- **Error recovery escalation**: On step failure, the system tries retrying, then scrolling, then pressing back, then relaunching the app before failing the test.
- **Interim verification**: The Supervisor checks for early goal completion every 3 steps, avoiding unnecessary actions once the goal is met.
- **Swappable LLM providers**: Both Gemini and OpenAI clients implement the same interface with built-in exponential backoff for rate limits.
