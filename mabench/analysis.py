"""RQ2/RQ3: summary tables, paired significance tests, and report figures."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, wilcoxon

SYSTEM_COLORS = {"single": "#2a78d6", "baseline": "#eb6834", "improved": "#1baf7a"}
ROLE_COLORS = {"architect": "#6250d6", "coder": "#e87ba4", "tester": "#eda100", "single": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load(exp) -> pd.DataFrame:
    exp = Path(exp)
    resolved_path = exp / "resolved.json"
    if not resolved_path.exists():
        raise FileNotFoundError(f"{resolved_path} missing; run mabench.evaluate first")
    resolved = {k: set(v) for k, v in json.loads(resolved_path.read_text()).items()}
    rows = []
    for f in exp.glob("*/*/*/result.json"):
        r = json.loads(f.read_text())
        r["resolved"] = r["instance_id"] in resolved.get(f"{r['system']}__{r['budget']}", set())
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["system", "budget", "category"])
    out = g.agg(n=("resolved", "size"), resolved=("resolved", "sum"), total_tokens=("tokens", "sum"),
                mean_tokens=("tokens", "mean"), mean_latency_s=("latency_s", "mean"),
                mean_tool_calls=("tool_calls", "mean")).reset_index()
    out["resolved_rate"] = out.resolved / out.n
    out["tokens_per_resolved"] = out.total_tokens.where(out.resolved > 0) / out.resolved.where(out.resolved > 0)
    return out


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    return 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)


def holm(pvalues: list) -> list:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compare systems on the same issues; only issues finished by both enter a test."""
    rows = []
    for (budget, cat), part in df.groupby(["budget", "category"]):
        for a, b in combinations(sorted(part.system.unique()), 2):
            x = part[part.system == a].set_index("instance_id")
            y = part[part.system == b].set_index("instance_id")
            ids = x.index.intersection(y.index)
            if ids.empty:
                continue
            x, y = x.loc[ids], y.loc[ids]
            only_a, only_b = int((x.resolved & ~y.resolved).sum()), int((y.resolved & ~x.resolved).sum())
            diff = x.tokens - y.tokens
            token_p = 1.0 if (diff == 0).all() else float(wilcoxon(diff).pvalue)
            rows.append({"budget": budget, "category": cat, "a": a, "b": b, "n": len(ids),
                         "rate_a": x.resolved.mean(), "rate_b": y.resolved.mean(), "only_a": only_a,
                         "only_b": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b), "tokens_wilcoxon_p": token_p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mcnemar_p_holm"] = holm(out.mcnemar_p.tolist())
        out["tokens_p_holm"] = holm(out.tokens_wilcoxon_p.tolist())
    return out


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)


def plot_resolved(summary: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    data = summary[summary.system.isin(SYSTEM_COLORS)]
    budgets = sorted(data.budget.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, cat, title in zip(axes, ("single", "multi"), ("Single-file bugs", "Multi-file bugs")):
        for system, color in SYSTEM_COLORS.items():
            s = data[(data.system == system) & (data.category == cat)].sort_values("budget")
            ax.plot(s.budget.map(budgets.index), s.resolved_rate, color=color, linewidth=2, marker="o",
                    markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=2, label=system)
        ax.set_xticks(range(len(budgets)), [f"{b // 1000:,}k" for b in budgets])
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Token budget per issue", color=MUTED)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1)
        _style(ax)
    axes[0].set_ylabel("Resolved rate", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, labelcolor=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def plot_roles(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = df[df.system.isin(SYSTEM_COLORS)]
    roles = pd.DataFrame(data.by_role.tolist(), index=data.index).fillna(0)
    means = roles.groupby([data.system, data.budget]).mean()
    labels = [f"{s} @ {b // 1000:,}k" for s, b in means.index]
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.42 * len(means) + 1)))
    left = pd.Series(0.0, index=means.index)
    for role, color in ROLE_COLORS.items():
        if role in means:
            ax.barh(labels, means[role], left=left, color=color, edgecolor="#fcfcfb", linewidth=2, label=role)
            left += means[role]
    ax.invert_yaxis()
    ax.set_xlabel("Mean tokens per issue", color=MUTED)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp")
    a = p.parse_args()
    df = load(a.exp)
    if df.empty:
        raise SystemExit(f"no finished runs under {a.exp}")
    out = Path(a.exp) / "analysis"
    out.mkdir(exist_ok=True)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    paired_tests(df).to_csv(out / "paired_tests.csv", index=False)
    plot_resolved(summary, out / "resolved_rate.png")
    plot_roles(df, out / "tokens_by_role.png")
    print(f"wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
