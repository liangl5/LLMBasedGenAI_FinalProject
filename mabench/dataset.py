"""Select Django issues from SWE-bench Verified into disjoint development and test sets.

Only 32 Django issues change more than one file, so 7 dev + 25 test uses all of them.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

N_DEV, N_TEST, MIN_TEST = 7, 25, 25


def changed_files(patch: str) -> list:
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", patch, re.M)))


def category(patch: str) -> str:
    return "single" if len(changed_files(patch)) == 1 else "multi"


def build_manifest(rows, n_dev: int = N_DEV, n_test: int = N_TEST) -> dict:
    """Order by a hash of the ID so selection is deterministic and independent of difficulty."""
    if n_test < MIN_TEST:
        raise ValueError(f"the proposal requires at least {MIN_TEST} test issues per category")
    groups = {"single": [], "multi": []}
    for r in rows:
        if r["repo"] == "django/django":
            groups[category(r["patch"])].append(
                {"instance_id": r["instance_id"], "category": category(r["patch"]),
                 "problem_statement": r["problem_statement"]})
    manifest = {"dev": [], "test": []}
    for name, items in groups.items():
        if len(items) < n_dev + n_test:
            raise ValueError(f"{name}: need {n_dev + n_test} issues, found {len(items)}")
        items.sort(key=lambda x: hashlib.sha256(x["instance_id"].encode()).hexdigest())
        manifest["dev"] += items[:n_dev]
        manifest["test"] += items[n_dev:n_dev + n_test]
    return manifest


def main():
    import pandas as pd
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", default="data/swebench_verified.parquet")
    p.add_argument("--out", default="data/manifest.json")
    a = p.parse_args()
    manifest = build_manifest(pd.read_parquet(a.parquet).to_dict("records"))
    Path(a.out).write_text(json.dumps(manifest, indent=1))
    print(f"wrote {a.out}: {len(manifest['dev'])} dev, {len(manifest['test'])} test")


if __name__ == "__main__":
    main()
