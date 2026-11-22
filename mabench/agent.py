"""The agent loop shared by every role in every system."""
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, execute_tool, truncate

KEEP_RECENT_MESSAGES = 8
NUDGE = "Continue by calling a tool. Call finish when your turn is done."


@dataclass
class Context:
    cfg: object
    llm: object
    sandbox: object
    ledger: object
    trace: object


@dataclass
class AgentResult:
    summary: str = ""
    passed: bool = False
    edited: list = field(default_factory=list)
    commands: list = field(default_factory=list)


def _chars(messages) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def _complete(ctx, role, messages):
    chars = _chars(messages)
    allowed = ctx.ledger.reserve(chars)
    ctx.trace.emit("llm_request", role, context_chars=chars)
    r = ctx.llm.complete(messages, TOOL_SCHEMAS, min(allowed, ctx.cfg.max_output_tokens), ctx.cfg.temperature)
    ctx.ledger.charge(role, r.input_tokens, r.output_tokens)
    ctx.trace.emit("llm_response", role, input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                   used=ctx.ledger.used, budget=ctx.ledger.limit, text=truncate(r.text, 1000))
    return r


def compact(ctx, role, messages):
    """Summarize older turns; the summary call is charged like any other."""
    if _chars(messages) < ctx.cfg.compact_at_chars or len(messages) <= KEEP_RECENT_MESSAGES + 2:
        return messages
    cut = len(messages) - KEEP_RECENT_MESSAGES
    while cut < len(messages) and messages[cut]["role"] != "assistant":  # never orphan a tool result
        cut += 1
    if cut <= 2 or cut >= len(messages):
        return messages
    old = "\n".join(f"{m['role']}: {truncate(m.get('content') or '', 2000)}" for m in messages[2:cut])
    request = [messages[0], {"role": "user", "content":
               "Summarize this work log: root-cause beliefs, files inspected, edits, test results, next steps.\n" + old}]
    summary = _complete(ctx, role, request).text
    ctx.trace.emit("compaction", role, before_chars=_chars(messages))
    return messages[:2] + [{"role": "user", "content": f"Summary of earlier work:\n{summary}"}] + messages[cut:]


def run_agent(ctx: Context, role: str, system: str, task: str, use_compaction: bool = False) -> AgentResult:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    result = AgentResult()
    while True:
        if use_compaction:
            messages = compact(ctx, role, messages)
        r = _complete(ctx, role, messages)
        messages.append({"role": "assistant", "content": r.text, "tool_calls": r.tool_calls, "raw": r.raw})
        if not r.tool_calls:
            messages.append({"role": "user", "content": NUDGE})
            continue
        for c in r.tool_calls:
            if c.name == "finish":
                result.summary = str(c.args.get("summary", ""))
                result.passed = c.args.get("passed") is True
                ctx.trace.emit("finish", role, summary=truncate(result.summary, 2000), passed=result.passed)
                return result
            output = execute_tool(ctx.sandbox, c.name, c.args)
            ok = not output.startswith("ERROR")
            if c.name == "edit_file" and ok:
                result.edited.append(c.args.get("path"))
            if c.name == "run_command":
                result.commands.append(c.args.get("command"))
            ctx.trace.emit("tool", role, tool=c.name, args={k: truncate(str(v), 300) for k, v in c.args.items()},
                           ok=ok, output_chars=len(output), output=truncate(output, 500))
            messages.append({"role": "tool", "name": c.name, "id": c.id, "content": output})
