"""
Generate plots from test_judgments.json.
Writes plots/deltas.png, plots/by_task.png, plots/by_judge.png.

Usage:
    python plot.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = Path(__file__).parent / "data" / "test_judgments.json"
PLOTS_DIR = Path(__file__).parent / "plots"

CONDITIONS = ["true", "upward", "downward"]
CONDITION_LABELS = {"true": "True", "upward": "Upward", "downward": "Downward"}
CONDITION_COLORS = {"true": "#4C72B0", "upward": "#55A868", "downward": "#C44E52"}

_DISPLAY = {
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-4o-mini":   "GPT-4o-mini",
    "gpt-4-turbo":   "GPT-4 Turbo",
    "gpt-4o":        "GPT-4o",
}


def compute_deltas(judgments):
    blind = {}
    for j in judgments:
        if j["condition"] == "blind" and j["score"] is not None:
            blind[(j["response_model"], j["judge_model"], j["prompt_id"], j["rep"])] = j["score"]

    deltas = defaultdict(list)
    for j in judgments:
        if j["condition"] == "blind" or j["score"] is None:
            continue
        key = (j["response_model"], j["judge_model"], j["prompt_id"], j["rep"])
        b = blind.get(key)
        if b is None:
            continue
        deltas[(j["condition"], j["task_type"], j["judge_model"])].append(j["score"] - b)
    return deltas


def plot_deltas(deltas, out_path):
    means = {}
    sems = {}
    for cond in CONDITIONS:
        vals = []
        for (c, _, _), v in deltas.items():
            if c == cond:
                vals.extend(v)
        means[cond] = np.mean(vals) if vals else 0
        sems[cond] = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(CONDITIONS))
    bars = ax.bar(
        x,
        [means[c] for c in CONDITIONS],
        yerr=[sems[c] for c in CONDITIONS],
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        capsize=5,
        width=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Mean score delta vs. blind")
    ax.set_title("Attribution bias: score delta by condition")
    for bar, cond in zip(bars, CONDITIONS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.05 if means[cond] >= 0 else -0.15),
            f"{means[cond]:+.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_by_task(deltas, out_path):
    task_types = sorted({t for (_, t, _) in deltas})
    x = np.arange(len(task_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cond in enumerate(CONDITIONS):
        means = []
        sems = []
        for tt in task_types:
            agg = []
            for (c, t, _), v in deltas.items():
                if c == cond and t == tt:
                    agg.extend(v)
            means.append(np.mean(agg) if agg else 0)
            sems.append(np.std(agg, ddof=1) / np.sqrt(len(agg)) if len(agg) > 1 else 0)
        ax.bar(
            x + i * width,
            means,
            yerr=sems,
            width=width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            capsize=4,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width)
    ax.set_xticklabels([tt.capitalize() for tt in task_types])
    ax.set_ylabel("Mean score delta vs. blind")
    ax.set_title("Attribution bias by task type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_by_judge(deltas, out_path):
    judge_models = sorted({jm for (_, _, jm) in deltas})
    x = np.arange(len(judge_models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cond in enumerate(CONDITIONS):
        means = []
        sems = []
        for jm in judge_models:
            agg = []
            for (c, _, j), v in deltas.items():
                if c == cond and j == jm:
                    agg.extend(v)
            means.append(np.mean(agg) if agg else 0)
            sems.append(np.std(agg, ddof=1) / np.sqrt(len(agg)) if len(agg) > 1 else 0)
        ax.bar(
            x + i * width,
            means,
            yerr=sems,
            width=width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            capsize=4,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width)
    ax.set_xticklabels([_DISPLAY.get(jm, jm) for jm in judge_models], rotation=15, ha="right")
    ax.set_ylabel("Mean score delta vs. blind")
    ax.set_title("Attribution bias by judge model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    judgments = json.loads(DATA_FILE.read_text())
    PLOTS_DIR.mkdir(exist_ok=True)
    deltas = compute_deltas(judgments)

    plot_deltas(deltas, PLOTS_DIR / "deltas.png")
    plot_by_task(deltas, PLOTS_DIR / "by_task.png")
    plot_by_judge(deltas, PLOTS_DIR / "by_judge.png")


if __name__ == "__main__":
    main()
