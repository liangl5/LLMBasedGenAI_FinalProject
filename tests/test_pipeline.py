import json
import threading

import pytest

from mabench.budget import SpendGuard
from mabench.config import RunConfig
from mabench.dataset import build_manifest, category, changed_files
from mabench.demo import list_runs, read_trace
from mabench.evaluate import write_predictions
from mabench.llm import FakeLLM
from mabench.run import run_matrix, run_one, select
from mabench.sample import ISSUE, make_repo, run_sample, script
from mabench.sandbox import LocalSandbox

ONE = "diff --git a/a.py b/a.py\n+x\n"
TWO = ONE + "diff --git a/b.py b/b.py\n+y\n"


def rows(n_single=40, n_multi=32):
    out = [{"repo": "django/django", "instance_id": f"s-{i}", "patch": ONE, "problem_statement": "p"} for i in range(n_single)]
    out += [{"repo": "django/django", "instance_id": f"m-{i}", "patch": TWO, "problem_statement": "p"} for i in range(n_multi)]
    return out + [{"repo": "other/repo", "instance_id": "x", "patch": ONE, "problem_statement": "p"}]


def test_category():
    assert changed_files(TWO) == ["a.py", "b.py"]
    assert (category(ONE), category(TWO)) == ("single", "multi")


def test_manifest_is_disjoint_balanced_and_deterministic():
    m = build_manifest(rows())
    dev, test = {i["instance_id"] for i in m["dev"]}, {i["instance_id"] for i in m["test"]}
    assert len(dev) == 14 and len(test) == 50 and not dev & test and "x" not in dev | test
    assert sum(i["category"] == "multi" for i in m["test"]) == 25
    assert build_manifest(list(reversed(rows()))) == m


def test_manifest_refuses_too_few_issues():
    with pytest.raises(ValueError, match="multi"):
        build_manifest(rows(n_multi=31))


def test_select_limits_each_category():
    items = [{"category": "single"}] * 3 + [{"category": "multi"}] * 3
    assert len(select(items, 2)) == 4


def job(tmp_path, n=[0]):
    n[0] += 1
    return lambda item: LocalSandbox(make_repo(tmp_path / f"repo{n[0]}_{threading.get_ident()}"))


def test_resume_does_not_pay_twice(tmp_path):
    item = {**ISSUE, "category": "single"}
    cfg = RunConfig("single", 20_000, "scripted")
    guard = SpendGuard(tmp_path / "spend.jsonl", 10, 1, 1)
    llm = FakeLLM(script("single"))
    assert run_one(cfg, item, tmp_path / "e", lambda: llm, job(tmp_path), guard, threading.Event()) == "done"
    spent = guard.total
    assert run_one(cfg, item, tmp_path / "e", lambda: pytest.fail("called LLM"), job(tmp_path), guard, threading.Event()) == "skipped"
    assert guard.total == spent > 0


def test_cap_blocks_before_any_call(tmp_path):
    item = {**ISSUE, "category": "single"}
    guard = SpendGuard(tmp_path / "spend.jsonl", 0.001, 1, 1)
    counts = run_matrix([RunConfig("single", 20_000, "m")], [item], tmp_path / "e",
                        lambda: pytest.fail("called LLM"), job(tmp_path), guard)
    assert counts == {"blocked": 1} and guard.total == 0


def test_failed_run_is_recorded_and_retryable(tmp_path):
    item = {**ISSUE, "category": "single"}
    cfg = RunConfig("single", 20_000, "scripted")
    guard = SpendGuard(tmp_path / "spend.jsonl", 10, 1, 1)
    assert run_one(cfg, item, tmp_path / "e", lambda: FakeLLM([]), job(tmp_path), guard, threading.Event()) == "error"
    cell = tmp_path / "e" / "single" / "20000" / ISSUE["instance_id"]
    assert (cell / "error.json").exists() and not (cell / "result.json").exists()
    assert run_one(cfg, item, tmp_path / "e", lambda: FakeLLM(script("single")), job(tmp_path), guard, threading.Event()) == "done"
    assert not (cell / "error.json").exists()


def test_predictions_format(tmp_path):
    run_sample(tmp_path)
    out = write_predictions(tmp_path)
    path, ids = out["improved__20000"]
    row = json.loads(path.read_text().splitlines()[0])
    assert ids == [ISSUE["instance_id"]] and row["model_name_or_path"] == "improved__20000"
    assert "return a + b" in row["model_patch"]


def test_demo_serves_only_listed_runs(tmp_path):
    run_sample(tmp_path)
    assert len(list_runs(tmp_path)) == 3
    assert read_trace(tmp_path, "single", "20000", ISSUE["instance_id"])[0]["kind"] == "start"
    with pytest.raises(KeyError):
        read_trace(tmp_path, "..", "..", "secret")
