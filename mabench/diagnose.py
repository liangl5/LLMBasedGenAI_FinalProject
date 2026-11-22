"""RQ1: measure where a run lost budget, directly from its trace."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def trace_metrics(events: list) -> dict:
    seen, repeated_reads, repeated_searches, cross_role = {}, 0, 0, 0
    contexts, handoff_chars, failed_tests, architect_files, coder_files = [], 0, 0, set(), set()
    for e in events:
        kind, role = e["kind"], e["role"]
        if kind == "llm_request":
            contexts.append(e["context_chars"])
        elif kind == "handoff":
            handoff_chars += e["chars"]
            if role == "architect":
                architect_files |= set(re.findall(r"[\w./-]+\.py", e["content"]))
        elif kind == "finish" and role == "tester" and not e["passed"]:
            failed_tests += 1
        elif kind == "tool":
            args = e["args"]
            if e["tool"] == "edit_file" and role == "coder" and e["ok"]:
                coder_files.add(args.get("path"))
            if e["tool"] in ("read_file", "search"):
                key = (e["tool"], args.get("path"), args.get("pattern"), args.get("start_line"))
                if key in seen:
                    repeated_reads += e["tool"] == "read_file"
                    repeated_searches += e["tool"] == "search"
                    cross_role += role not in seen[key]
                    seen[key].add(role)
                else:
                    seen[key] = {role}
    # A hypothesis is "lost" when the Architect named files and the Coder edited none of them.
    lost = bool(architect_files and coder_files and not architect_files & coder_files)
    return {"repeated_reads": repeated_reads, "repeated_searches": repeated_searches,
            "cross_role_repeats": cross_role, "max_context_chars": max(contexts, default=0),
            "handoff_chars": handoff_chars, "failed_test_rounds": failed_tests, "hypothesis_lost": lost}


def collect(exp) -> pd.DataFrame:
    rows = []
    for f in Path(exp).glob("*/*/*/result.json"):
        r = json.loads(f.read_text())
        trace = f.with_name("trace.jsonl").read_text(encoding="utf-8").splitlines()
        rows.append({"system": r["system"], "budget": r["budget"], "instance_id": r["instance_id"],
                     "category": r.get("category"), "stop": r["stop"], "tokens": r["tokens"],
                     **trace_metrics([json.loads(l) for l in trace])})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp")
    a = p.parse_args()
    df = collect(a.exp)
    if df.empty:
        raise SystemExit(f"no finished runs under {a.exp}")
    out = Path(a.exp) / "diagnosis.csv"
    df.to_csv(out, index=False)
    numeric = df.select_dtypes(["number", "bool"]).drop(columns="budget")
    print(numeric.groupby([df.system, df.budget]).mean().round(2).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
