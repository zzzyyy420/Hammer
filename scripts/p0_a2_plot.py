"""Overlay the frozen Qwen NLI baselines on Step B's two Pareto planes."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from step_b_plot import COLORS, MARKERS, MODELS, POLICIES, SHORT, nondominated


ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / 'research_outputs/step_b_plugplay'
OUT = ROOT / 'research_outputs/p0_a2'
QWEN = 'Qwen2.5-7B-Instruct'


def main():
    with (STEP / 'eight_view_metrics.csv').open(encoding='utf-8', newline='') as f:
        old = list(csv.DictReader(f))
    with (OUT / 'seven_view_metrics.csv').open(encoding='utf-8', newline='') as f:
        new = list(csv.DictReader(f))
    lookup = {(r['model'], r['policy'], r['view']): float(r['tool_acc']) * 100 for r in old}
    assert len(lookup) == 4 * 4 * 8
    for r in new:
        key = r['model'], r['policy'], r['view']
        val = float(r['tool_acc']) * 100
        if key in lookup:
            assert abs(lookup[key] - val) < 1e-10
        lookup[key] = val
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), dpi=180)
    planes = (('Coarse-Desc', 'Name-Swapped', 'Coarse description vs swapped name'),
              ('Distractor-Swapped', 'Target-Swapped', 'Distractor vs target swap'))
    extra = {'NLI-0.50': ('X', '#C0353A'), 'NLI-Matched': ('P', '#1F6995')}
    for panel, (ax, (xview, yview, title)) in enumerate(zip(axes, planes)):
        for model in MODELS:
            points = [(lookup[model, p, xview], lookup[model, p, yview]) for p in POLICIES]
            frontier_points = points + ([(lookup[model, p, xview], lookup[model, p, yview])
                                         for p in extra] if model == QWEN else [])
            frontier = nondominated(frontier_points)
            if len(frontier) > 1:
                ax.plot([v[0] for v in frontier], [v[1] for v in frontier],
                        color=COLORS[model], alpha=.4, linewidth=1.2, zorder=1)
            for policy, (x, y) in zip(POLICIES, points):
                gate = policy == 'RelGate-0.50'
                ax.scatter(x, y, s=70 if gate else 43, marker=MARKERS[policy],
                           color=COLORS[model] if gate else 'white', edgecolors=COLORS[model],
                           linewidths=1.5 if gate else 1.2, zorder=3)
            if model == QWEN:
                for policy, (mark, color) in extra.items():
                    x, y = lookup[model, policy, xview], lookup[model, policy, yview]
                    ax.scatter(x, y, s=103, marker=mark, color=color, edgecolors='white',
                               linewidths=.75, zorder=4)
                    offset = ((-62, 9) if policy == 'NLI-0.50' else (6, -15)) if panel == 1 else (6, 6)
                    ax.annotate(policy, (x, y), xytext=offset, textcoords='offset points',
                                color=color, fontsize=7)
        ax.set(xlabel=f'{xview} Tool Acc (%)', ylabel=f'{yview} Tool Acc (%)', title=title)
        ax.grid(True, color='#E3E5E8', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=.16, y=.19)
    models = [Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS[m],
                     markeredgecolor=COLORS[m], label=SHORT[m], markersize=7) for m in MODELS]
    policies = [Line2D([0], [0], marker=MARKERS[p], color='#444', linestyle='none',
                       markerfacecolor='#444' if p == 'RelGate-0.50' else 'white',
                       label=p, markersize=7) for p in POLICIES]
    policies += [Line2D([0], [0], marker=mark, color='none', markerfacecolor=color,
                        markeredgecolor=color, label=p, markersize=8)
                 for p, (mark, color) in extra.items()]
    fig.legend(handles=models + policies, loc='lower center', ncol=5,
               bbox_to_anchor=(.5, .015), frameon=False, fontsize=8)
    fig.suptitle('Frozen Step B planes with Qwen NLI baselines', fontsize=13, weight='semibold', y=.98)
    fig.text(.5, .095, 'Axes and historical points unchanged; exploratory views only.',
             ha='center', fontsize=8, color='#555')
    fig.subplots_adjust(left=.065, right=.98, top=.87, bottom=.29, wspace=.24)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'dual_pareto.{ext}', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


if __name__ == '__main__':
    main()
