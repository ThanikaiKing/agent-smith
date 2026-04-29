import json
import math
import time
from google import genai
from google.genai import types
from google.genai import errors

client = genai.Client()

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

    contents = [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]

    # The loop — keeps going until Gemini stops making function calls
    while True:
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config,
                )
                break
            except errors.ServerError as e:
                if e.status_code == 503 and attempt < 3:
                    wait = 2 ** attempt
                    print(f"   [503 unavailable, retrying in {wait}s…]")
                    time.sleep(wait)
                else:
                    raise

        parts = response.candidates[0].content.parts
        function_calls = [p for p in parts if p.function_call and p.function_call.name]

        # ── Did Gemini want to use a tool? ──
        if function_calls:
            # Add Gemini's response to history before sending tool results
            contents.append(response.candidates[0].content)

            tool_response_parts = []
            for part in function_calls:
                fc = part.function_call
                print(f"\n🔧 AGENT CALLS TOOL: {fc.name}")
                print(f"   Input: {json.dumps(dict(fc.args), indent=2)}")

                result = execute_tool(fc.name, dict(fc.args))
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
