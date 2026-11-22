import json

import pytest

from mabench.agent import Context, compact
from mabench.budget import Ledger
from mabench.config import SYSTEMS, RunConfig
from mabench.llm import FakeLLM, LLMResponse, ToolCall, call
from mabench.sample import FIX, HANDOFF, ISSUE, TEST, make_repo, run_sample, script
from mabench.sandbox import LocalSandbox
from mabench.systems import parse_handoff, run_system
from mabench.trace import Trace


def execute(tmp_path, system, steps, budget=20_000, **cfg):
    box = LocalSandbox(make_repo(tmp_path / "repo"))
    config = RunConfig(system, budget, "scripted", **cfg)
    trace = Trace()
    return run_system(config, ISSUE, FakeLLM(steps), box, trace, Ledger(budget, config.max_steps)), trace


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_every_system_fixes_sample(tmp_path, system):
    result, _ = execute(tmp_path, system, script("single" if system == "single" else system))
    assert result["stop"] == "finished"
    assert "+    return a + b" in result["patch"]
    assert result["tokens"] == sum(result["by_role"].values())


def test_team_roles_all_spend_budget(tmp_path):
    result, _ = execute(tmp_path, "baseline", script("baseline"))
    assert set(result["by_role"]) == {"architect", "coder", "tester"}


def failing_tester():
    return LLMResponse(tool_calls=[ToolCall("finish", {"summary": HANDOFF, "passed": False})])


def test_fix_loop_limit_stops_improved_team(tmp_path):
    steps = [call("finish", summary=HANDOFF), FIX, call("finish", summary=HANDOFF), failing_tester(),
             call("finish", summary=HANDOFF), failing_tester()]
    result, _ = execute(tmp_path, "improved", steps)
    assert (result["stop"], result["fix_loops"]) == ("fix_loop_limit", 2)


def test_baseline_has_no_fix_loop_limit(tmp_path):
    steps = [call("finish", summary="a"), FIX, call("finish", summary="c"), failing_tester()]
    steps += [call("finish", summary="c"), failing_tester()] * 2
    steps += [call("finish", summary="c"), LLMResponse(tool_calls=[ToolCall("finish", {"summary": "ok", "passed": True})])]
    result, _ = execute(tmp_path, "baseline", steps)
    assert (result["stop"], result["fix_loops"]) == ("finished", 3)


def test_budget_stop_keeps_partial_patch(tmp_path):
    steps = [FIX] + [call("read_file", path="calc.py")] * 50
    result, _ = execute(tmp_path, "single", steps, budget=300)
    assert result["stop"] == "token_budget"
    assert "+    return a + b" in result["patch"]


def test_reply_without_tool_call_is_nudged(tmp_path):
    steps = [LLMResponse(text="thinking")] + script("single")
    result, trace = execute(tmp_path, "single", steps)
    assert result["stop"] == "finished" and result["llm_calls"] == 6


def test_structured_handoff_reaches_next_agent(tmp_path):
    _, trace = execute(tmp_path, "improved", script("improved"))
    first = next(e for e in trace.events if e["kind"] == "handoff")
    assert json.loads(first["content"])["relevant_files"] == ["calc.py"]


def test_parse_handoff_handles_fences_and_garbage():
    assert parse_handoff('```json\n{"root_cause": "x", "extra": 1}\n```') == {"root_cause": "x"}
    assert parse_handoff("not json") == {"notes": "not json"}
    assert parse_handoff("[1]") == {"notes": "[1]"}


def test_compaction_keeps_tool_pairs_and_charges_budget():
    cfg = RunConfig("single", 100_000, "scripted", compact_at_chars=100)
    ledger = Ledger(100_000, 20)
    ctx = Context(cfg, FakeLLM([LLMResponse(text="summary")]), None, ledger, Trace())
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(10):
        messages += [{"role": "assistant", "content": "a", "tool_calls": [], "raw": None},
                     {"role": "tool", "name": "search", "content": "x" * 50}]
    out = compact(ctx, "single", messages)
    assert len(out) < len(messages) and ledger.steps == 1
    assert "summary" in out[2]["content"] and out[3]["role"] == "assistant"


def test_sample_writes_traces(tmp_path):
    results = run_sample(tmp_path)
    assert [r["stop"] for r in results] == ["finished"] * 3
    assert (tmp_path / "improved" / "20000" / ISSUE["instance_id"] / "trace.jsonl").exists()
    assert TEST
