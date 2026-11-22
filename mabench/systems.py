"""The compared systems as LangGraph graphs. Only coordination differs between them."""
import json
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import Context, run_agent
from .budget import BudgetExceeded

HANDOFF_KEYS = ("root_cause", "relevant_files", "tests_run", "observed_failures")
BASE = "You are working in a Python repository to fix the reported issue. Use the tools. Call finish when done."
SINGLE = BASE + " Locate the cause, edit the source, verify with tests, then finish."
ARCHITECT = BASE + " You are the Architect: find the root cause and describe the fix. Do not edit files."
CODER = BASE + " You are the Coder: implement the fix in source files. Do not modify tests."
TESTER = BASE + " You are the Tester: run relevant tests. Do not edit files. Call finish with passed=true only if the fix works."
STRUCTURED = (" The finish summary must be a JSON object with keys root_cause (string), relevant_files (list), "
              "tests_run (list), and observed_failures (list).")


class TeamState(TypedDict, total=False):
    issue: str
    handoff: str
    record: list
    fix_loops: int
    passed: bool
    stop: str


def parse_handoff(text: str) -> dict:
    """Keep the schema fields; a malformed reply survives as notes instead of being lost."""
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"notes": text}
    if not isinstance(data, dict):
        return {"notes": text}
    return {k: data[k] for k in HANDOFF_KEYS if k in data} or {"notes": text}


def _guarded(fn):
    def node(state):
        try:
            return fn(state)
        except BudgetExceeded as e:
            return {"stop": e.reason}
    return node


def _single_graph(ctx):
    def agent(state):
        run_agent(ctx, "single", SINGLE, f"Issue:\n{state['issue']}", use_compaction=True)
        return {"stop": "finished"}

    g = StateGraph(TeamState)
    g.add_node("agent", _guarded(agent))
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile()


def _team_graph(ctx):
    flags = ctx.cfg.flags
    suffix = STRUCTURED if flags.structured_handoff else ""

    def task(state):
        parts = [f"Issue:\n{state['issue']}"]
        if state.get("handoff"):
            parts.append(f"Handoff from previous agent:\n{state['handoff']}")
        if flags.shared_record and state.get("record"):
            parts.append("Completed work so far:\n" + "\n".join(state["record"]))
        return "\n\n".join(parts)

    def step(state, role, system):
        r = run_agent(ctx, role, system + suffix, task(state))
        handoff = json.dumps(parse_handoff(r.summary)) if flags.structured_handoff else r.summary
        ctx.trace.emit("handoff", role, chars=len(handoff), content=handoff[:2000])
        entry = f"{role}: edited {sorted(set(r.edited)) or 'nothing'}; ran {r.commands[-5:] or 'nothing'}"
        return r, {"handoff": handoff, "record": state.get("record", []) + [entry]}

    def architect(state):
        return step(state, "architect", ARCHITECT)[1]

    def coder(state):
        return step(state, "coder", CODER)[1]

    def tester(state):
        r, update = step(state, "tester", TESTER)
        if r.passed:
            return {**update, "passed": True, "stop": "finished"}
        loops = state.get("fix_loops", 0) + 1
        update["fix_loops"] = loops
        if flags.fix_loop_limit and loops >= ctx.cfg.max_fix_loops:
            update["stop"] = "fix_loop_limit"
        return update

    def go_on(next_node):
        return lambda state: END if state.get("stop") else next_node

    g = StateGraph(TeamState)
    for name, fn in (("architect", architect), ("coder", coder), ("tester", tester)):
        g.add_node(name, _guarded(fn))
    g.add_edge(START, "architect")
    g.add_conditional_edges("architect", go_on("coder"), ["coder", END])
    g.add_conditional_edges("coder", go_on("tester"), ["tester", END])
    g.add_conditional_edges("tester", go_on("coder"), ["coder", END])
    return g.compile()


def run_system(cfg, issue: dict, llm, sandbox, trace, ledger) -> dict:
    ctx = Context(cfg, llm, sandbox, ledger, trace)
    graph = _single_graph(ctx) if cfg.system == "single" else _team_graph(ctx)
    started = time.time()
    trace.emit("start", "system", system=cfg.system, instance_id=issue["instance_id"], budget=cfg.token_budget)
    state = graph.invoke({"issue": issue["problem_statement"]},
                         {"recursion_limit": cfg.max_steps * 3 + 10})
    patch = sandbox.diff()  # always collected, so a budget stop still submits partial work
    tool_calls = sum(e["kind"] == "tool" for e in trace.events)
    result = {"instance_id": issue["instance_id"], "system": cfg.system, "budget": cfg.token_budget,
              "model": cfg.model, "stop": state.get("stop", "unknown"), "patch": patch,
              "tokens": ledger.used, "input_tokens": ledger.input_tokens, "output_tokens": ledger.output_tokens,
              "by_role": dict(ledger.by_role), "llm_calls": ledger.steps, "tool_calls": tool_calls,
              "fix_loops": state.get("fix_loops", 0), "latency_s": round(time.time() - started, 3)}
    trace.emit("end", "system", stop=result["stop"], tokens=result["tokens"], patch_chars=len(patch))
    return result
