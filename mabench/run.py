"""Run an experiment matrix. Defaults to a dry run; spending requires --paid and the credit lock."""
import argparse
import json
import os
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .budget import Ledger, SpendCapReached, SpendGuard
from .config import SYSTEMS, RunConfig
from .systems import run_system
from .trace import Trace

IMAGE_TEMPLATE = "swebench/sweb.eval.x86_64.{instance_id}:latest"


def image_name(instance_id: str, template: str = IMAGE_TEMPLATE) -> str:
    """Match SWE-bench's published names: lowercase, with `__` written as `_1776_`."""
    return template.format(instance_id=instance_id.lower().replace("__", "_1776_"))


def select(items, per_category=None):
    if per_category is None:
        return list(items)
    seen, out = {}, []
    for item in items:
        seen[item["category"]] = seen.get(item["category"], 0) + 1
        if seen[item["category"]] <= per_category:
            out.append(item)
    return out


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic, so a crash can't leave a half-written "finished" result


def run_one(cfg, item, exp, make_llm, make_sandbox, guard, halt) -> str:
    cell = Path(exp) / cfg.system / str(cfg.token_budget) / item["instance_id"]
    if (cell / "result.json").exists():
        return "skipped"
    if halt.is_set() or not guard.admit(cfg.token_budget):
        halt.set()
        return "blocked"
    cell.mkdir(parents=True, exist_ok=True)
    trace, sandbox = Trace(cell / "trace.jsonl"), None
    try:
        sandbox = make_sandbox(item)
        ledger = Ledger(cfg.token_budget, cfg.max_steps, guard.add)
        result = run_system(cfg, item, make_llm(), sandbox, trace, ledger)
        result["category"] = item["category"]
        _write_json(cell / "result.json", result)
        (cell / "error.json").unlink(missing_ok=True)
        return "done"
    except SpendCapReached as e:
        halt.set()
        _write_json(cell / "error.json", {"error": str(e)})
        return "capped"
    except Exception as e:  # one failed issue must not lose the rest of the matrix; rerun retries it
        _write_json(cell / "error.json", {"error": f"{type(e).__name__}: {e}"})
        return "error"
    finally:
        trace.close()
        if sandbox:
            sandbox.close()
        guard.release(cfg.token_budget)


def run_matrix(cfgs, items, exp, make_llm, make_sandbox, guard, workers: int = 1) -> dict:
    halt, counts = threading.Event(), {}
    jobs = [(c, i) for c in cfgs for i in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status in pool.map(lambda j: run_one(j[0], j[1], exp, make_llm, make_sandbox, guard, halt), jobs):
            counts[status] = counts.get(status, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--systems", nargs="+", default=["single", "baseline", "improved"], choices=sorted(SYSTEMS))
    p.add_argument("--budgets", nargs="+", type=int, required=True)
    p.add_argument("--per-category", type=int, help="limit issues per category")
    p.add_argument("--exp", required=True, help="output directory, e.g. results/pilot")
    p.add_argument("--model", default="unset")
    p.add_argument("--usd-cap", type=float, default=0.0)
    p.add_argument("--usd-in", type=float, default=0.0, help="USD per million input tokens")
    p.add_argument("--usd-out", type=float, default=0.0, help="USD per million output tokens")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--docker", default="docker", help='docker command, e.g. "wsl -d Ubuntu-20.04 --exec docker"')
    p.add_argument("--image-template", default=IMAGE_TEMPLATE)
    p.add_argument("--paid", action="store_true", help="actually call the LLM")
    a = p.parse_args()

    items = select(json.loads(Path(a.manifest).read_text())[a.split], a.per_category)
    cfgs = [RunConfig(s, b, a.model) for s in a.systems for b in a.budgets]
    todo = [(c, i) for c in cfgs for i in items
            if not (Path(a.exp) / c.system / str(c.token_budget) / i["instance_id"] / "result.json").exists()]
    worst = sum(c.token_budget for c, _ in todo) / 1e6 * max(a.usd_in, a.usd_out)
    print(f"{len(cfgs) * len(items)} runs, {len(todo)} remaining, worst-case ${worst:.2f}, cap ${a.usd_cap:.2f}")
    if not a.paid:
        print("dry run: nothing was called")
        return
    if a.model == "unset" or a.usd_cap <= 0 or a.usd_in <= 0 or a.usd_out <= 0:
        p.error("--paid requires --model, --usd-cap, --usd-in, and --usd-out")

    from .llm import VertexLLM
    from .sandbox import DockerSandbox
    VertexLLM(a.model)  # fail on the credit lock before any container starts
    docker = shlex.split(a.docker)
    guard = SpendGuard(Path(a.exp) / "spend.jsonl", a.usd_cap, a.usd_in, a.usd_out)
    counts = run_matrix(cfgs, items, a.exp, lambda: VertexLLM(a.model),
                        lambda i: DockerSandbox(image_name(i["instance_id"], a.image_template), docker),
                        guard, a.workers)
    print(counts, f"total spend ${guard.total:.2f}")


if __name__ == "__main__":
    main()
