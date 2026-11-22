"""Scripted, zero-cost runs on a tiny repository for tests and demo rehearsal."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .budget import Ledger
from .config import RunConfig
from .llm import FakeLLM, LLMResponse, ToolCall, call
from .sandbox import LocalSandbox
from .systems import run_system
from .trace import Trace

ISSUE = {"instance_id": "sample__calc-1", "problem_statement": "add(2, 3) returns -1 instead of 5."}
TEST = f'"{sys.executable}" test_calc.py'
FIX = call("edit_file", path="calc.py", old_text="return a - b", new_text="return a + b")
HANDOFF = json.dumps({"root_cause": "add subtracts", "relevant_files": ["calc.py"],
                      "tests_run": [], "observed_failures": []})


def make_repo(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (path / "test_calc.py").write_text("from calc import add\nassert add(2, 3) == 5\nprint('ok')\n")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    return path


def script(system: str) -> list:
    if system == "single":
        return [call("search", pattern="def add"), call("read_file", path="calc.py"), FIX,
                call("run_command", command=TEST), call("finish", summary="fixed add")]
    return [call("read_file", path="calc.py"), call("finish", summary=HANDOFF),
            call("read_file", path="calc.py"), FIX, call("finish", summary=HANDOFF),
            call("run_command", command=TEST),
            LLMResponse(tool_calls=[ToolCall("finish", {"summary": HANDOFF, "passed": True})])]


def run_sample(out_dir, systems=("single", "baseline", "improved"), budget=20_000) -> list:
    results = []
    for system in systems:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            sandbox = LocalSandbox(make_repo(tmp))
            cfg = RunConfig(system, budget, "scripted")
            cell = Path(out_dir) / system / str(budget) / ISSUE["instance_id"]
            trace = Trace(cell / "trace.jsonl")
            result = run_system(cfg, ISSUE, FakeLLM(script(system)), sandbox, trace, Ledger(budget, cfg.max_steps))
            trace.close()
            (cell / "result.json").write_text(json.dumps(result, indent=2))
            results.append(result)
    return results
