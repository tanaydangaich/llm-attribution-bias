"""
Generate plots from test_judgments.json.
Writes plots/deltas.png, plots/by_task.png, plots/by_judge.png, plots/inversions.png.

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


def compute_inversions(judgments):
    """
    For each (judge, prompt, rep), compare all pairs of response models.
    Inversion = blind ranking flipped under attributed condition.
    Returns dict: {(condition, task_type): (inversions, total)}
    """
    blind = {}
    task_for = {}
    for j in judgments:
        key = (j["response_model"], j["judge_model"], j["prompt_id"], j["rep"])
        if j["condition"] == "blind" and j["score"] is not None:
            blind[key] = j["score"]
            task_for[key] = j["task_type"]

    cond_scores = defaultdict(dict)
    for j in judgments:
        if j["condition"] == "blind" or j["score"] is None:
            continue
        key = (j["response_model"], j["judge_model"], j["prompt_id"], j["rep"])
        cond_scores[j["condition"]][key] = j["score"]

    results = defaultdict(lambda: [0, 0])  # (cond, task_type) -> [inv, total]

    for cond, scores in cond_scores.items():
        seen = set()
        for (rm1, jm, pid, rep), s1_cond in scores.items():
            k1 = (rm1, jm, pid, rep)
            b1 = blind.get(k1)
            if b1 is None:
                continue
            for (rm2, jm2, pid2, rep2), s2_cond in scores.items():
                if jm2 != jm or pid2 != pid or rep2 != rep or rm2 == rm1:
                    continue
                pair = tuple(sorted([rm1, rm2]))
                pair_key = (cond, pair, jm, pid, rep)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                k2 = (rm2, jm, pid, rep)
                b2 = blind.get(k2)
                if b2 is None or b1 == b2:
                    continue
                tt = task_for.get(k1, "unknown")
                results[(cond, tt)][1] += 1
                blind_order = b1 > b2
                cond_order = s1_cond > s2_cond
                if blind_order != cond_order:
                    results[(cond, tt)][0] += 1

    return results


def plot_inversions(judgments, out_path):
    inv_data = compute_inversions(judgments)
    task_types = sorted({tt for (_, tt) in inv_data})
    width = 0.25
    x = np.arange(len(task_types))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: inversion rate by condition (aggregated across task types)
    ax = axes[0]
    overall = {}
    for cond in CONDITIONS:
        inv, tot = 0, 0
        for tt in task_types:
            d = inv_data.get((cond, tt), [0, 0])
            inv += d[0]; tot += d[1]
        overall[cond] = inv / tot * 100 if tot else 0

    bars = ax.bar(
        np.arange(len(CONDITIONS)),
        [overall[c] for c in CONDITIONS],
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        width=0.5,
    )
    for bar, cond in zip(bars, CONDITIONS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{overall[cond]:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Ranking inversion rate (%)")
    ax.set_title("Inversion rate by condition")
    ax.set_ylim(0, max(overall.values()) * 1.25 + 5)

    # Right: inversion rate by condition × task type
    ax = axes[1]
    for i, cond in enumerate(CONDITIONS):
        rates = []
        for tt in task_types:
            d = inv_data.get((cond, tt), [0, 0])
            rates.append(d[0] / d[1] * 100 if d[1] else 0)
        ax.bar(x + i * width, rates, width=width,
               label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond])

    ax.set_xticks(x + width)
    ax.set_xticklabels([tt.capitalize() for tt in task_types])
    ax.set_ylabel("Ranking inversion rate (%)")
    ax.set_title("Inversion rate by condition × task type")
    ax.legend()

    fig.suptitle("Attribution-induced ranking inversions", fontweight="bold")
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
    plot_inversions(judgments, PLOTS_DIR / "inversions.png")


if __name__ == "__main__":
    main()
