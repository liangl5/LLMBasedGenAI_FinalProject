import pandas as pd
import pytest

from mabench.analysis import holm, mcnemar_exact, paired_tests, plot_resolved, plot_roles, summarize
from mabench.diagnose import trace_metrics


def frame():
    rows = []
    for budget in (50_000, 100_000):
        for cat in ("single", "multi"):
            for i in range(8):
                for system, hit in (("single", i < 6), ("baseline", i < 2), ("improved", i < 4)):
                    roles = {"single": 100} if system == "single" else {"architect": 40, "coder": 40, "tester": 40}
                    rows.append({"instance_id": f"{cat}-{i}", "system": system, "budget": budget, "category": cat,
                                 "resolved": hit, "tokens": 100 + i * (system != "single") * 10,
                                 "latency_s": 2.0, "tool_calls": 5, "by_role": roles})
    return pd.DataFrame(rows)


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(0, 0) == 1.0


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_summary_rates_and_cost_per_resolved():
    s = summarize(frame())
    row = s[(s.system == "single") & (s.budget == 50_000) & (s.category == "single")].iloc[0]
    assert row.resolved_rate == 0.75 and row.tokens_per_resolved == pytest.approx(800 / 6)


def test_summary_no_resolved_is_nan_not_infinite():
    df = frame().assign(resolved=False)
    assert summarize(df).tokens_per_resolved.isna().all()


def test_paired_tests_use_common_issues():
    df = frame()
    df = df[~((df.system == "baseline") & (df.instance_id == "single-7"))]
    t = paired_tests(df)
    row = t[(t.a == "baseline") & (t.b == "single") & (t.category == "single") & (t.budget == 50_000)].iloc[0]
    assert row.n == 7 and (row.only_a, row.only_b) == (0, 4)
    assert (t.mcnemar_p_holm >= t.mcnemar_p).all()


def test_figures_render(tmp_path):
    df = frame()
    plot_resolved(summarize(df), tmp_path / "r.png")
    plot_roles(df, tmp_path / "t.png")
    assert (tmp_path / "r.png").stat().st_size > 1000 and (tmp_path / "t.png").stat().st_size > 1000


def tool(role, name, ok=True, **args):
    return {"kind": "tool", "role": role, "tool": name, "ok": ok, "args": args}


def test_trace_metrics_find_waste_and_lost_hypothesis():
    events = [{"kind": "llm_request", "role": "architect", "context_chars": 10},
              tool("architect", "read_file", path="a.py"),
              {"kind": "handoff", "role": "architect", "chars": 9, "content": "cause in a.py"},
              {"kind": "llm_request", "role": "coder", "context_chars": 30},
              tool("coder", "read_file", path="a.py"), tool("coder", "read_file", path="a.py"),
              tool("coder", "search", pattern="x"), tool("coder", "search", pattern="x"),
              tool("coder", "edit_file", path="b.py", old_text="", new_text="y"),
              {"kind": "finish", "role": "tester", "passed": False}]
    m = trace_metrics(events)
    assert (m["repeated_reads"], m["repeated_searches"], m["cross_role_repeats"]) == (2, 1, 1)
    assert (m["max_context_chars"], m["handoff_chars"], m["failed_test_rounds"]) == (30, 9, 1)
    assert m["hypothesis_lost"] is True
