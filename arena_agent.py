import asyncio, json, os, re, uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Change these three lines ──────────────────────────────
AGENT_NAME  = "KachuaChap-v1"          # shown on the leaderboard
LINKEDIN_URL  = os.getenv("LINKEDIN_URL")
AGENT_STACK = "Python / ADK / Gemini" # describe your stack
MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # 2.0-flash has 0 free-tier quota

# ── Leave these as-is ─────────────────────────────────────
MCP_ENDPOINT   = "https://agent-arena-623774504237.asia-southeast1.run.app/mcp"
ID_TOKEN       = os.getenv("EPHEMERAL_JWT")  #https://agent-arena.dev/ MCP Documentattion tab
MAX_TURNS      = 20
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not ID_TOKEN:
    raise SystemExit(
        "EPHEMERAL_JWT is missing. Add it to .env or export it.\n"
        "Get a fresh token from https://agent-arena.dev/ (MCP Documentation tab).\n"
        "Tokens expire after ~1 hour."
    )
if not GEMINI_API_KEY:
    raise SystemExit("GEMINI_API_KEY is missing. Add it to .env or export it.")

# ── Run State ────────────────────────────────────────────

class RunState:
    """Shared mutable state — passed into every tool via closure."""
    def __init__(self):
        self.run_id          = str(uuid.uuid4())
        self.agent_id        = ""
        self.task_id         = ""
        self.current_level   = 1
        self.total_score     = 0
        self.tasks_attempted = 0
        self.level_history   = []

    def record(self, level, title, score, levelled_up):
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up: self.current_level = level + 1
        self.level_history.append({
            "level": level, "task": title,
            "score": score, "up": levelled_up
        })
        icon = "✓" if levelled_up else ("~" if score >= 70 else "✗")
        print(f"  {icon} L{level}  score={score}/100")


async def mcp_call(tool: str, args: dict, state: RunState) -> str:
    """Open a fresh MCP session, call one tool, return text result."""
    transport = StreamableHttpTransport(url=MCP_ENDPOINT)
    async with Client(transport, name="arena-agent") as c:
        result = await c.call_tool(tool, args)
    return "\n".join(
        getattr(b, "text", "")
        for b in result.content
        if getattr(b, "text", None)
    )

# Step 6 - Helper Tools

async def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for current facts, docs, or recent events.
    Use before answering any task that needs up-to-date information.
    Args:
        query: the search query string
        max_results: how many results to return (1-10)
    """
    max_results = max(1, min(max_results, 10))

    def _ddgs_search():
        from ddgs import DDGS
        return list(DDGS().text(query, max_results=max_results))

    try:
        hits = await asyncio.to_thread(_ddgs_search)
        if hits:
            return "\n---\n".join(
                f"{r.get('title', 'Untitled')}\n{r.get('body', '').strip()}\n{r.get('href', '')}"
                for r in hits
            )
    except Exception as ex:
        print(f"  [web_search] ddgs failed: {ex}")

    # Fallback: DuckDuckGo instant-answer API (limited but no extra deps)
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": "1"}
    async with httpx.AsyncClient() as c:
        r = await c.get(url, params=params, timeout=10)
        data = r.json()
    results = []
    if data.get("AbstractText"):
        results.append(data["AbstractText"])
    for t in data.get("RelatedTopics", [])[:max_results]:
        if "Text" in t:
            results.append(t["Text"])
        elif "Topics" in t:
            for sub in t["Topics"][:max_results - len(results)]:
                if "Text" in sub:
                    results.append(sub["Text"])
    return "\n---\n".join(results) or "No results found."

async def run_code(code: str, timeout: int = 10) -> str:
    """Execute Python code in a subprocess and return its output.
    Use to verify algorithm correctness or compute results before submitting.
    Args:
        code: valid Python source code
        timeout: max seconds (default 10)
    """
    import sys
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"Timed out after {timeout}s"
    o = out.decode().strip()
    e = err.decode().strip()
    return (f"stdout:\n{o}\nstderr:\n{e}" if e else o) or "(no output)"

# Step 4 - Arena Tools

def make_arena_tools(state: RunState):
    """Returns the four Arena tool functions with state captured via closure."""
    async def register_agent(name: str, stack: str) -> str:
        """Register this agent. Call once at start. Returns AGENT_ID."""
        result = await mcp_call("register_agent",
            {"idToken": ID_TOKEN, "name": name, "stack": stack}, state)
        m = re.search(r"AGENT_ID:\s*(\S+)", result)
        if m: state.agent_id = m.group(1)
        return result

    async def get_tasks(agent_id: str) -> str:
        """Fetch the current task. Returns JSON with id, title, description."""
        result = await mcp_call("get_tasks",
            {"idToken": ID_TOKEN, "agentId": agent_id}, state)
        try:
            data = json.loads(result)
            if "id" in data: state.task_id = data["id"]
        except: pass
        return result

    async def submit_task(agent_id: str, task_id: str, content: str) -> str:
        """Submit your answer. Scored 0-100. Score >= 70 means LEVEL_UP."""
        result = await mcp_call("submit_task", {
            "idToken": ID_TOKEN, "agentId": agent_id,
            "taskId": task_id, "content": content,
            "metadata": {"agent_name": AGENT_NAME, "model": MODEL},
        }, state)
        return result

    async def skip_task(agent_id: str, task_id: str) -> str:
        """Skip a task. Call when stuck."""
        return await mcp_call("skip_task",
            {"idToken": ID_TOKEN, "agentId": agent_id, "taskId": task_id}, state)

    return [register_agent, get_tasks, submit_task, skip_task, web_search, run_code]

SYSTEM_PROMPT = f"""
You are an autonomous agent competing in the Agent Arena.
Your goal: solve as many tasks as possible to reach the highest level.

EXACT STEPS TO FOLLOW:
1. Call register_agent(name="{AGENT_NAME}", stack="{AGENT_STACK}")
2. Call get_tasks(agent_id) — read the task description carefully
3. Think through a thorough, complete answer. Aim for 90+/100.
4. Call submit_task(agent_id, task_id, content=YOUR_FULL_ANSWER)
5. If LEVEL_UP in result → go to step 2 for next level
6. If NO_TASKS in result → you are done, print a summary and stop
7. Never submit the same task_id twice. Never ask for confirmation.

Helper tools (use exact names): web_search, run_code.
- web_search(query): search the web for current facts before answering.
- run_code(code): execute Python to verify logic or compute answers.
Only call tools that exist above. Never invent tool names.
"""

def build_agent(state: RunState) -> LlmAgent:
    return LlmAgent(
        name="arena_agent",
        model=MODEL,
        instruction=SYSTEM_PROMPT,
        tools=make_arena_tools(state),
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )

# Step 5 - Runner Loop -- 

async def run_turn(runner, session_id, message, max_retries=5):
    """Send one message; collect and return the agent's final text reply."""
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )
    for attempt in range(max_retries):
        try:
            final = ""
            async for event in runner.run_async(
                user_id="arena-user", session_id=session_id, new_message=content
            ):
                if not event.content: continue
                for part in event.content.parts:
                    if getattr(part, "function_call", None):
                        print(f"  → [{part.function_call.name}]")
                    elif getattr(part, "text", None) and event.turn_complete:
                        final = part.text
            return final
        except Exception as ex:
            err = str(ex)
            if "429" not in err and "RESOURCE_EXHAUSTED" not in err:
                raise
            wait = min(60, 15 * (attempt + 1))
            print(f"  [rate limit] waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries for model {MODEL}")

async def main():
    state = RunState()
    agent = build_agent(state)
    sessions = InMemorySessionService()
    await sessions.create_session(
        app_name="arena", user_id="arena-user", session_id=state.run_id
    )
    runner = Runner(agent=agent, session_service=sessions, app_name="arena")

    print(f"\n=== AGENT ARENA  |  {AGENT_NAME}  |  {MODEL} ===")

    # Turn 1: Kickoff
    reply = await run_turn(runner, state.run_id,
        "Start now. Register, get your first task, solve it fully, submit. "
        "Then keep fetching and submitting until NO_TASKS."
    )
    if reply: print(f"\n[agent] {reply[:300]}")

    # Turns 2-N: nudge after each level-up
    for turn in range(2, MAX_TURNS + 1):
        if state.tasks_attempted == 0: break
        reply = await run_turn(runner, state.run_id,
            "Continue — get the next task, solve it, submit. Stop on NO_TASKS."
        )
        if reply: print(f"\n[agent] {reply[:300]}")
        if any(kw in reply.lower() for kw in ("no_tasks", "finished", "complete")):
            break

    print(f"\n=== DONE  level={state.current_level}  score={state.total_score} ===")

if __name__ == "__main__":
    asyncio.run(main())