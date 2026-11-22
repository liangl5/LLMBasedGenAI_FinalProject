import pytest

from mabench.budget import BudgetExceeded, Ledger, SpendCapReached, SpendGuard
from mabench.config import CREDIT_CONFIRMED_ENV, RunConfig
from mabench.llm import VertexLLM, parse_gemini, to_gemini, ToolCall
from mabench.sandbox import LocalSandbox, repo_path
from mabench.tools import execute_tool, truncate


def test_ledger_stops_before_overspending():
    ledger = Ledger(1000, 10)
    assert ledger.reserve(400) == 900
    ledger.charge("coder", 800, 150)
    with pytest.raises(BudgetExceeded) as e:
        ledger.reserve(400)
    assert e.value.reason == "token_budget"
    assert ledger.by_role["coder"] == 950


def test_ledger_step_limit():
    ledger = Ledger(10_000, 1)
    ledger.charge("a", 1, 1)
    with pytest.raises(BudgetExceeded) as e:
        ledger.reserve(0)
    assert e.value.reason == "max_steps"


def test_spend_guard_survives_restart_and_caps(tmp_path):
    path = tmp_path / "spend.jsonl"
    guard = SpendGuard(path, 1.0, 1.0, 2.0)
    assert guard.admit(400_000)        # worst case $0.80
    assert not guard.admit(200_000)    # would reach $1.20
    guard.release(400_000)
    guard.add(500_000, 0)
    assert SpendGuard(path, 1.0, 1.0, 2.0).total == pytest.approx(0.5)
    with pytest.raises(SpendCapReached):
        guard.add(600_000, 0)


@pytest.mark.parametrize("bad", ["../x", "/etc/passwd", "a/../../x", "/testbed/../etc"])
def test_repo_path_refuses_escape(bad):
    with pytest.raises(ValueError):
        repo_path(bad)


def test_repo_path_accepts_inside():
    assert repo_path("/testbed/django/db.py") == "django/db.py"
    assert repo_path("./django/../a.py") == "a.py"


def test_tools_report_errors_as_text(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n")
    box = LocalSandbox(tmp_path)
    assert "outside" in execute_tool(box, "read_file", {"path": "../secret"})
    assert "matched 2" in execute_tool(box, "edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "y"})
    assert execute_tool(box, "nope", {}).startswith("ERROR")
    assert execute_tool(box, "read_file", {"wrong": 1}).startswith("ERROR")
    assert execute_tool(box, "search", {"pattern": "("}).startswith("ERROR")


def test_edit_creates_new_file_but_not_over_existing(tmp_path):
    (tmp_path / "a.py").write_text("x")
    box = LocalSandbox(tmp_path)
    assert execute_tool(box, "edit_file", {"path": "n/b.py", "old_text": "", "new_text": "y"}) == "created n/b.py"
    assert execute_tool(box, "edit_file", {"path": "a.py", "old_text": "", "new_text": "y"}).startswith("ERROR")
    assert (tmp_path / "a.py").read_text() == "x"


def test_truncate_keeps_both_ends():
    out = truncate("a" * 50 + "b" * 50, 20)
    assert out.startswith("a") and out.endswith("b") and "omitted" in out


def test_vertex_locked_without_confirmation(monkeypatch):
    monkeypatch.delenv(CREDIT_CONFIRMED_ENV, raising=False)
    with pytest.raises(PermissionError, match="locked"):
        VertexLLM("any-model")


def test_vertex_refuses_other_projects_even_when_confirmed(monkeypatch):
    monkeypatch.setenv(CREDIT_CONFIRMED_ENV, "1")
    with pytest.raises(PermissionError, match="refusing"):
        VertexLLM("any-model", project="stalwart-glider-491120-j7")


def test_gemini_replays_raw_parts_and_groups_tool_results():
    raw = [{"functionCall": {"name": "search", "args": {}}, "thoughtSignature": "sig"}]
    _, contents = to_gemini([
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [ToolCall("search", {})], "raw": raw},
        {"role": "tool", "name": "search", "content": "1"}, {"role": "tool", "name": "search", "content": "2"}])
    assert contents[1]["parts"][0]["thoughtSignature"] == "sig"
    assert len(contents) == 3 and len(contents[2]["parts"]) == 2


def test_parse_gemini_handles_empty_candidate():
    r = parse_gemini({"candidates": [{"finishReason": "MAX_TOKENS"}],
                      "usageMetadata": {"promptTokenCount": 5, "thoughtsTokenCount": 7}})
    assert r.tool_calls == [] and (r.input_tokens, r.output_tokens) == (5, 7)


def test_config_rejects_unknown_system():
    with pytest.raises(ValueError):
        RunConfig("nope", 100, "m")
