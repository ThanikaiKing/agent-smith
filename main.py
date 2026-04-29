import json
import logging
import math
import time
from google import genai
from google.genai import types
from google.genai import errors

# ─────────────────────────────────────────────
# LOGGING SETUP
#
# Two handlers:
#   • Console — INFO and above (key events only)
#   • agent.log — DEBUG and above (everything, for review)
#
# Levels used in this file:
#   DEBUG   → token counts, full tool results, response metadata
#   INFO    → each API call start/finish, tool invocations
#   WARNING → retries (rate limit, server errors)
#   ERROR   → terminal failures that stop execution
# ─────────────────────────────────────────────

log = logging.getLogger("agent")
log.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S")

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_fmt)

_file = logging.FileHandler("agent.log")
_file.setLevel(logging.DEBUG)
_file.setFormatter(_fmt)

log.addHandler(_console)
log.addHandler(_file)

client = genai.Client()

def _to_loggable(obj):
    """Recursively convert SDK objects to plain dicts/lists so they can be JSON-serialised."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [_to_loggable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_loggable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):          # pydantic models (most SDK types)
        return _to_loggable(obj.model_dump())
    if hasattr(obj, "__dict__"):            # plain objects
        return _to_loggable({k: v for k, v in obj.__dict__.items() if not k.startswith("_")})
    return str(obj)

# ─────────────────────────────────────────────
# TOOLS — These are the agent's "hands"
# You define WHAT the tool does in Python.
# The LLM decides WHEN to call it.
# ─────────────────────────────────────────────

def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def web_search(query: str) -> str:
    """Simulate a web search (we'll wire real search in Phase 2)."""
    # Mocked results for now — enough to see the agent USE the tool
    mock_results = {
        "population of india": "India's population is approximately 1.44 billion (2024).",
        "python latest version": "Python 3.13 was released in October 2024.",
        "capital of france": "The capital of France is Paris.",
    }
    query_lower = query.lower()
    for key, val in mock_results.items():
        if any(word in query_lower for word in key.split()):
            return val
    return f"Search result for '{query}': No specific data found, but this is where real search results would appear."

# ─────────────────────────────────────────────
# TOOL DEFINITIONS — What you tell Gemini
# about each tool (name, description, parameters)
# ─────────────────────────────────────────────

tools = types.Tool(
    function_declarations=[
        {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use this for any arithmetic or math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression e.g. '2 ** 10' or '(15 * 4) / 3'"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "web_search",
            "description": "Search the web for current information about any topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    }
                },
                "required": ["query"]
            }
        }
    ]
)

config = types.GenerateContentConfig(tools=[tools])

# ─────────────────────────────────────────────
# TOOL EXECUTOR — Routes tool calls to real Python functions
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "calculate":
        return calculate(tool_input["expression"])
    elif tool_name == "web_search":
        return web_search(tool_input["query"])
    else:
        return f"Unknown tool: {tool_name}"

# ─────────────────────────────────────────────
# THE AGENT LOOP — This is the ReAct pattern
#
# Reason  → Gemini thinks about what to do
# Act     → Gemini calls a tool
# Observe → We run the tool and give result back
# Repeat  → Until Gemini stops calling tools
# ─────────────────────────────────────────────

def run_agent(user_message: str):
    print(f"\n{'='*50}")
    print(f"USER: {user_message}")
    print(f"{'='*50}")
    log.info("Agent started | prompt=%r", user_message[:80])

    contents = [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]

    turn = 0

    # The loop — keeps going until Gemini stops making function calls
    while True:
        turn += 1
        response = None
        for attempt in range(1, 4):
            log.info(
                "API call | model=gemini-3.1-flash-lite-preview  turn=%d  attempt=%d  context_messages=%d",
                turn, attempt, len(contents),
            )
            log.debug(
                "REQUEST BODY:\n%s",
                json.dumps({"contents": _to_loggable(contents), "config": _to_loggable(config)}, indent=2),
            )
            t0 = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    contents=contents,
                    config=config,
                )
                elapsed = time.perf_counter() - t0
                u = response.usage_metadata
                log.info(
                    "API response | latency=%.2fs  tokens(prompt=%s out=%s total=%s)  finish=%s",
                    elapsed,
                    u.prompt_token_count,
                    u.candidates_token_count,
                    u.total_token_count,
                    response.candidates[0].finish_reason,
                )
                log.debug(
                    "RESPONSE BODY:\n%s",
                    json.dumps(_to_loggable(response.candidates), indent=2),
                )
                break
            except errors.ClientError as e:
                if e.code == 429:
                    # Extract suggested retry delay from the API response details
                    retry_delay = None
                    try:
                        for detail in e.details.get("error", {}).get("details", []):
                            if "retryDelay" in detail:
                                retry_delay = int(detail["retryDelay"].rstrip("s"))
                                break
                    except (AttributeError, ValueError):
                        pass
                    wait = retry_delay if retry_delay else 2 ** (attempt + 4)
                    if attempt < 3:
                        log.warning("Rate limit (429) — retrying in %ds (attempt %d/3)", wait, attempt)
                        print(f"   [429 rate limit — retrying in {wait}s…]")
                        time.sleep(wait)
                    else:
                        log.error("Rate limit (429) — all retries exhausted, suggested wait=%ds", wait)
                        print(f"\nRate limit exceeded. Please wait {wait}s before retrying.")
                        return
                else:
                    # 4xx errors are caller mistakes — no point retrying
                    log.error("Client error (%d %s): %s", e.code, e.status, e.message)
                    print(f"\nAPI error ({e.code} {e.status}): {e.message}")
                    return
            except errors.ServerError as e:
                if attempt < 3:
                    wait = 2 ** attempt
                    log.warning("Server error (%d) — retrying in %ds (attempt %d/3)", e.code, wait, attempt)
                    print(f"   [{e.code} server error — retrying in {wait}s…]")
                    time.sleep(wait)
                else:
                    log.error("Server error (%d %s) — all retries exhausted: %s", e.code, e.status, e.message)
                    print(f"\nServer error ({e.code} {e.status}): {e.message}")
                    return
            except Exception as e:
                log.error("Unexpected error: %s", e, exc_info=True)
                print(f"\nUnexpected error: {e}")
                return
        if response is None:
            return

        parts = response.candidates[0].content.parts
        function_calls = [p for p in parts if p.function_call and p.function_call.name]

        # ── Did Gemini want to use a tool? ──
        if function_calls:
            # Add Gemini's response to history before sending tool results
            contents.append(response.candidates[0].content)

            tool_response_parts = []
            for part in function_calls:
                fc = part.function_call
                args = dict(fc.args)
                log.info("Tool call | name=%s  args=%s", fc.name, json.dumps(args))
                print(f"\n🔧 AGENT CALLS TOOL: {fc.name}")
                print(f"   Input: {json.dumps(args, indent=2)}")

                result = execute_tool(fc.name, args)
                log.debug("Tool result | name=%s  result=%r", fc.name, result)
                print(f"   Result: {result}")

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )

            contents.append(types.Content(role="user", parts=tool_response_parts))

        # ── Gemini is done — no more tool calls ──
        else:
            final_text = "".join(p.text for p in parts if p.text)
            log.info("Agent finished | turns=%d  response_chars=%d", turn, len(final_text))
            print(f"\n🤖 AGENT: {final_text}")
            break

# ─────────────────────────────────────────────
# RUN IT — Try these test prompts
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: Pure math — agent should use calculate tool
    run_agent("What is 2 to the power of 16, divided by 512?")

    # # Test 2: Search — agent should use web_search tool
    run_agent("What is the current population of India?")

    # # Test 3: Multi-tool — agent needs BOTH tools
    run_agent("Search for Python's latest version and tell me what 3.13 minus 2.7 equals.")
