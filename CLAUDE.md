# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the agent
uv run main.py

# Add a dependency
uv add <package>

# Run a one-off Python expression inside the venv
uv run python -c "..."
```

## Architecture

This is a single-file agentic learning project (`main.py`). All logic lives there — no modules, no packages.

### How the agent works (ReAct loop)

`run_agent()` implements the ReAct pattern (Reason → Act → Observe → repeat):

1. **Request** — `contents` (full conversation history) + `config` (tool schemas) are sent to `generate_content` on every call. The model has no memory of its own; the entire history is replayed each turn.
2. **Tool call** — if the response contains a `function_call` part, the code runs the matching Python function and appends both the model's reply and the tool result to `contents`.
3. **Terminal reply** — if the response contains only text (no function calls), the loop breaks.

### Key objects

| Object | What it is |
|---|---|
| `contents` | `list[types.Content]` — grows each turn; alternates `role="user"` / `role="model"` |
| `config` | `GenerateContentConfig(tools=[...])` — tool schemas sent on every API call |
| `tools` | `types.Tool` with `function_declarations` — descriptions the model reads to decide when to call a tool |

### Logging

Two handlers on the `"agent"` logger:
- **Console** — `INFO` and above (API call start/finish, tool invocations, errors)
- **`agent.log`** — `DEBUG` and above (full request/response bodies serialised as JSON, token counts)

`_to_loggable()` recursively converts SDK objects (pydantic models, plain objects) to JSON-serialisable dicts for debug logging.

### Error handling in the API call loop

Retries up to 3 times. `429` extracts the `retryDelay` from the response details. `5xx` uses exponential backoff. Non-retryable `4xx` errors exit immediately.
