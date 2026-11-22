# Diagnosing and Improving Multi-Agent Coding Teams Under a Fixed Token Budget

COMSE6998-007, Fall 2026. Zhenjie Zhu, Haotian Lei, Luke Liang.

Do multi-agent coding teams beat one strong agent when both get the **same token budget**? This repository compares three LangGraph systems on Django bugs from SWE-bench Verified:

| System | Design |
|---|---|
| `single` | One agent with a terminal loop and context compaction |
| `baseline` | Architect, Coder, and Tester with free-form handoffs |
| `improved` | The team plus structured handoffs, a shared work record, and a fix-loop limit |

Each improvement can be switched off for ablations (`improved-no-handoff`, `improved-no-record`, `improved-no-limit`). All systems share the model, sampling settings, tools, step limit, and token budget. The budget counts every prompt, handoff, and retry.

> **Status:** the harness is built and tested offline. No experiment has been run, so this repository contains no results yet.

## Research questions

1. Where does a typical multi-agent team lose performance and budget?
2. Which targeted changes reduce those losses, and by how much?
3. Under the same budget, how do the single agent, baseline team, and improved team compare in resolved rate, cost, and latency?

## Repository layout

| Path | Purpose |
|---|---|
| `mabench/systems.py` | The LangGraph systems |
| `mabench/agent.py`, `tools.py`, `sandbox.py` | Shared agent loop, tools, and Docker sandbox |
| `mabench/budget.py`, `llm.py` | Token ledger, dollar cap, and Vertex AI client |
| `mabench/dataset.py` | Disjoint development and test issue selection |
| `mabench/run.py`, `evaluate.py` | Experiment runner and official SWE-bench scoring |
| `mabench/diagnose.py`, `analysis.py` | Waste metrics, statistics, and figures |
| `mabench/demo.py` | Side-by-side live and replay viewer |

## Installation

Requires Python 3.10+, Git, and x86-64 Docker. On Windows, use Docker inside WSL2.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

To include the real-container test (downloads a 2.9 GB image, makes no LLM calls):

```powershell
$env:MABENCH_DOCKER = "wsl -d Ubuntu-20.04 --exec docker"
pytest tests/test_docker_integration.py
```

## Try the demo at no cost

```powershell
python -m mabench.demo results/sample --sample
```

Open <http://127.0.0.1:8765>. The sample runs are scripted and do not represent model performance.

## Dataset

```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest -Uri "https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Verified/parquet/default/test/0.parquet" -OutFile data/swebench_verified.parquet
python -m mabench.dataset
```

SWE-bench Verified has 231 Django issues: 199 single-file and 32 multi-file. The manifest takes 7 development and 25 test issues per category, chosen by a hash of the issue ID. All design decisions must use `dev` only.

## Running experiments

**Paid runs are locked.** A run needs `--paid`, a dollar cap, token prices, and `MABENCH_CREDIT_CONFIRMED=1`. Without `--paid`, the command only prints the run count and worst-case cost.

```powershell
# 1. Preview
python -m mabench.run --split dev --per-category 3 --budgets 50000 100000 200000 --exp results/pilot --usd-in <IN> --usd-out <OUT>

# 2. Pilot, only after the credit is confirmed
$env:MABENCH_CREDIT_CONFIRMED = "1"
python -m mabench.run --split dev --per-category 3 --budgets 50000 100000 200000 --exp results/pilot `
  --model <MODEL> --usd-in <IN> --usd-out <OUT> --usd-cap 10 --paid --docker "wsl -d Ubuntu-20.04 --exec docker"

# 3. Diagnose the baseline
python -m mabench.diagnose results/pilot

# 4. Final matrix, using the two chosen improvements
python -m mabench.run --split test --systems single baseline improved improved-no-handoff improved-no-record `
  --budgets <B1> <B2> <B3> --exp results/final --model <MODEL> --usd-in <IN> --usd-out <OUT> --usd-cap <CAP> --paid `
  --docker "wsl -d Ubuntu-20.04 --exec docker"

# 5. Score inside WSL with Python 3.10+ and `pip install -e ".[eval]"`
python -m mabench.evaluate results/final

# 6. Tables and figures
python -m mabench.analysis results/final
```

Rerunning a command skips finished issues and retries failed ones. Spending is logged per LLM call in `spend.jsonl`, so the cap holds across restarts.

## Statistics

Resolved rates are compared on the same issues with exact McNemar tests, token use with Wilcoxon signed-rank tests, and p-values are Holm-corrected.

## Troubleshooting

| Problem | Fix |
|---|---|
| `paid calls are locked` | Intended until the credit is confirmed |
| `refusing project` | Only the configured class project is allowed |
| `gcloud` reauthentication error | `$env:CLOUDSDK_ACTIVE_CONFIG_NAME="genai-final"; gcloud auth login` |
| Vertex HTTP 404 | The model is unavailable in that project or location |
| Docker commands fail on Windows | Use `--exec` in the WSL command so quoting is preserved |
| Disk fills up | Images are about 2.9 GB each. Run `docker image prune -a` between batches |
| A run shows `error.json` | Read the error, then rerun the same command to retry it |
