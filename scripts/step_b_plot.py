"""Plot the two frozen robustness tradeoffs from Step B scored metrics."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs/step_b_plugplay"
MODELS = ("Hammer-Repro 1.5B", "B2 1.5B", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct")
POLICIES = ("Keep-Name", "Always-Mask", "Length-64", "RelGate-0.50")
COLORS = {"Hammer-Repro 1.5B": "#5F6572", "B2 1.5B": "#167F77",
          "Qwen2.5-7B-Instruct": "#B45F31", "Llama-3.1-8B-Instruct": "#7358A2"}
MARKERS = {"Keep-Name": "o", "Always-Mask": "s", "Length-64": "^", "RelGate-0.50": "D"}
SHORT = {"Hammer-Repro 1.5B": "Hammer", "B2 1.5B": "B2",
         "Qwen2.5-7B-Instruct": "Qwen 7B", "Llama-3.1-8B-Instruct": "Llama 8B"}


def nondominated(points):
    return sorted((x, y) for x, y in points if not any(
        (u >= x and v >= y) and (u > x or v > y) for u, v in points))


def main():
    with (OUT / "eight_view_metrics.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(r["model"], r["policy"], r["view"]): float(r["tool_acc"]) * 100 for r in rows}
    assert len(lookup) == 4 * 4 * 8
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), dpi=180)
    panels = (("Coarse-Desc", "Name-Swapped", "Coarse description vs swapped name"),
              ("Distractor-Swapped", "Target-Swapped", "Distractor vs target swap"))
    offsets = {
        (0, "Hammer-Repro 1.5B"): (-38, 5), (0, "Qwen2.5-7B-Instruct"): (7, -14),
        (0, "B2 1.5B"): (-25, 11), (0, "Llama-3.1-8B-Instruct"): (9, 10),
        (1, "Hammer-Repro 1.5B"): (-40, 6), (1, "Qwen2.5-7B-Instruct"): (7, -14),
        (1, "B2 1.5B"): (10, -15), (1, "Llama-3.1-8B-Instruct"): (10, 11),
    }
    for panel_i, (ax, (xview, yview, title)) in enumerate(zip(axes, panels)):
        for model in MODELS:
            points = [(lookup[model, p, xview], lookup[model, p, yview]) for p in POLICIES]
            frontier = nondominated(points)
            if len(frontier) > 1:
                ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                        color=COLORS[model], alpha=.5, linewidth=1.3, zorder=1)
            for policy, (x, y) in zip(POLICIES, points):
                is_gate = policy == "RelGate-0.50"
                ax.scatter(x, y, s=77 if is_gate else 45, marker=MARKERS[policy],
                           color=COLORS[model] if is_gate else "white", edgecolors=COLORS[model],
                           linewidths=1.6 if is_gate else 1.2, zorder=3)
            x, y = points[-1]
            ax.annotate(SHORT[model], (x, y), xytext=offsets[panel_i, model], textcoords="offset points",
                        color=COLORS[model], fontsize=8, weight="semibold")
        ax.set(xlabel=f"{xview} Tool Acc (%)", ylabel=f"{yview} Tool Acc (%)", title=title)
        ax.grid(True, color="#E3E5E8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=.17, y=.18)
    model_legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m],
                           markeredgecolor=COLORS[m], label=SHORT[m], markersize=7) for m in MODELS]
    policy_legend = [Line2D([0], [0], marker=MARKERS[p], color="#444", linestyle="none",
                            markerfacecolor="#444" if p == "RelGate-0.50" else "white",
                            label=p, markersize=7) for p in POLICIES]
    fig.legend(handles=model_legend + policy_legend, loc="lower center", ncol=8,
               bbox_to_anchor=(.5, .012), frameon=False, fontsize=8)
    fig.suptitle("Frozen RelGate-0.50 across caller models", fontsize=13, weight="semibold", y=.98)
    fig.text(.5, .095, "Lines connect nondominated observed policies within each model; exploratory views only.",
             ha="center", fontsize=8, color="#555")
    fig.subplots_adjust(left=.065, right=.98, top=.87, bottom=.27, wspace=.24)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"dual_pareto.{ext}", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
