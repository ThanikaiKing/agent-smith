# Agent Smith

A learning project for understanding Agentic AI — built phase by phase, from a bare ReAct loop to a real multi-agent personal productivity system.

## Goal

Work through the full stack of agentic AI concepts hands-on: tool calling, stateful workflows, multi-agent coordination, and real-world integrations — building something that actually gets used at the end.

## Roadmap

### Phase 1 — Foundation ✅
> Branch: `phase-1`

- Understand the ReAct loop (Reason → Act → Observe → repeat)
- Build a basic tool-calling agent with the Google Gemini API (Python)
- Add memory (simple in-context memory first)

### Phase 2 — Structure
> Branch: `phase-2`

- Learn LangGraph (controlled, stateful agent workflows)
- Build a workflow: goal → plan → tools → output
- Add external memory (persist state between runs)

### Phase 3 — Multi-Agent
- Learn CrewAI (multi-agent abstraction)
- Build a 2-agent system: Planner + Executor
- Add human-in-the-loop checkpoint

### Phase 4 — Real Project
- Personal productivity agent: TickTick + Gmail + Calendar
- Built locally using Claude Code + MCP

## Stack

| Phase | Key technology |
|---|---|
| 1 | Google Gemini API (`google-genai`) |
| 2 | LangGraph |
| 3 | CrewAI |
| 4 | MCP (Model Context Protocol) |

## Running

```bash
uv run main.py
```
