# Multi-Agent Coding Teams Under a Fixed Token Budget: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible harness that compares a single agent, a baseline Architect/Coder/Tester team, and an improved team on SWE-bench Verified Django bugs under identical token budgets.

**Architecture:** One `mabench` Python package. Every system is a LangGraph graph over the same agent loop, tools, sandbox, and token ledger, so the only experimental variable is coordination. Each run writes a JSONL trace, which drives diagnosis, analysis, and the demo.

**Tech Stack:** Python 3.10+, LangGraph, Vertex AI REST (Gemini), Docker, SWE-bench harness, pandas, SciPy, matplotlib, pytest.

**Spec:** `docs/Project_Proposal_zz3558_hl3945_ll3945.pdf`; grading in `docs/Project-Deliverables-slides.pdf`.

## Global Constraints

- No paid LLM call until the user confirms the $1,000 credit offsets Gemini charges.
- Only project `fresh-booster-489119-q5` may be used. `stalwart-glider-491120-j7` and the Education billing account must never be used.
- All systems share base model, sampling settings, toolset, maximum steps, and token budget.
- The budget counts every prompt, handoff, and retry.
- Django only, single-file and multi-file categories, 25 to 40 test issues each. Design decisions use separate development issues.
- Primary metric: resolved rate from the official SWE-bench harness.
- Report and repo are due 2026-12-15.

## Review Focus

1. A run that exhausts its budget mid-edit still submits its partial patch.
2. A crash or restart never pays twice for a finished run.
3. A malformed model reply or tool call does not kill the run.
4. A tool path outside the repository is refused.
5. Missing credit confirmation or a forbidden project stops before any network call.

---

### Task 1: Core harness
**Files:** `mabench/{config,budget,llm,sandbox,tools,trace,agent}.py`, `tests/test_core.py`
**Produces:** `RunConfig`, `SYSTEMS`, `Ledger`, `BudgetExceeded`, `FakeLLM`, `VertexLLM`, `LocalSandbox`, `DockerSandbox`, `execute_tool`, `Trace`, `run_agent`.
- [ ] Test budget stops, path refusal, tool errors, and paid-call guards.
- [ ] Implement and pass.

### Task 2: The three systems and switchable improvements
**Files:** `mabench/systems.py`, `mabench/sample.py`, `tests/test_systems.py`
**Produces:** `run_system(cfg, issue, llm, sandbox, trace, ledger) -> dict`.
- [ ] Test each system end to end on a scripted bug, the fix-loop limit, and partial patch on budget stop.
- [ ] Implement and pass.

### Task 3: Dataset, runner, and evaluation
**Files:** `mabench/{dataset,run,evaluate}.py`, `tests/test_pipeline.py`
- [ ] Test category split, dev/test separation, resume, and spend cap.
- [ ] Implement and pass. Default mode is a dry run.

### Task 4: Diagnosis and analysis
**Files:** `mabench/{diagnose,analysis}.py`, `tests/test_analysis.py`
- [ ] Test waste metrics, McNemar, Wilcoxon, and Holm correction on known inputs.
- [ ] Implement and pass.

### Task 5: Demo and documentation
**Files:** `mabench/demo.py`, `mabench/demo.html`, `README.md`
- [ ] Side-by-side live view over traces, with sample-trace replay as the backup.
- [ ] README: install, environment, usage, reproduction, troubleshooting.

### Blocked until credit is confirmed
- [ ] Choose a stable model, run development pilot, set budget levels.
- [ ] Diagnose baseline on development issues and choose the two improvements.
- [ ] Full run, evaluation, report, and presentation.
