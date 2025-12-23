# Framework Decision Memo

## Framework Selected: Google Agent Development Kit (ADK)

After evaluating available options, I selected **Google's Agent Development Kit (ADK)** 
as the architectural foundation for this mobile QA automation system.

### Why ADK?

1. **Multi-Agent Architecture**: ADK's Supervisor-Planner-Executor pattern directly 
   maps to QA automation needs—plan tests, execute actions, verify results.

2. **Tool Ecosystem**: ADK's tool abstraction inspired the action vocabulary 
   (`tap_text`, `scroll_until_text`, `tap_and_type`, etc.) that the Planner decides 
   to use and the Executor runs.

3. **Gemini Optimization**: ADK is designed for Gemini's multimodal capabilities, 
   which are essential for screenshot-based UI understanding and reasoning.

### Implementation Approach

I implemented ADK's core patterns with custom ADB tooling to achieve tight integration 
with Android-specific requirements (UIAutomator dumps, shell commands, screenshot capture).

The result is a modular system where the LLM provider can be swapped with a single 
flag (`--provider gemini|openai`), demonstrating the framework's flexibility.

