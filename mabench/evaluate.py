"""Score saved patches with the official SWE-bench harness and record resolved issues."""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Verified"


def write_predictions(exp) -> dict:
    """Write one predictions file per system and budget; return {label: (path, instance_ids)}."""
    exp, groups = Path(exp), defaultdict(list)
    for f in exp.glob("*/*/*/result.json"):
        r = json.loads(f.read_text())
        groups[f"{r['system']}__{r['budget']}"].append(r)
    out = {}
    for label, rows in sorted(groups.items()):
        path = exp / f"predictions_{label}.jsonl"
        rows.sort(key=lambda r: r["instance_id"])
        path.write_text("".join(json.dumps({"instance_id": r["instance_id"], "model_name_or_path": label,
                                            "model_patch": r["patch"]}) + "\n" for r in rows))
        out[label] = (path, [r["instance_id"] for r in rows])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp")
    p.add_argument("--run-id", default="mabench")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    exp, resolved = Path(a.exp).resolve(), {}
    for label, (path, ids) in write_predictions(exp).items():
        run_id = f"{a.run_id}_{label}"
        subprocess.run([sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", DATASET,
                        "--split", "test", "--predictions_path", str(path), "--instance_ids", *ids,
                        "--max_workers", str(a.workers), "--run_id", run_id], cwd=exp, check=True)
        report = exp / f"{label}.{run_id}.json"
        if not report.exists():
            raise FileNotFoundError(f"SWE-bench did not write {report}")
        resolved[label] = json.loads(report.read_text())["resolved_ids"]
    (exp / "resolved.json").write_text(json.dumps(resolved, indent=1))
    print({k: len(v) for k, v in resolved.items()})


if __name__ == "__main__":
    main()
