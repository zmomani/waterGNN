"""Step 11 - summary of experiments E1 to E3."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from common import *

RES = ROOT / "results"
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})
EXP = [("E1_nodes", "E1\ncontin.\nsupply"), ("E2", "E2\ntransfer"),
       ("E3_blind", "E3\nsupply-\nblind"), ("E3_aware", "E3\nsupply-\naware")]
R = {k: json.loads((RES / f"detection_{k}.json").read_text()) for k, _ in EXP}
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(wspace=0.42))
x = np.arange(len(EXP))
for ax, key, title, fmt in [(axs[0], "relative", "(a) Score, % of maximum", "{:.0f}"),
                            (axs[1], "TP", "(b) Leaks found, of 23", "{:.0f}"),
                            (axs[2], "FP", "(c) False alarms", "{:.0f}")]:
    for i, (m, c) in enumerate([("GNN", BLUE), ("Ridge", ORANGE)]):
        v = np.array([R[k][m]["test_2019"][key] for k, _ in EXP]) * (100 if key == "relative" else 1)
        b = ax.bar(x + (i - 0.5) * 0.38, v, width=0.34, color=c, label=m)
        for xi, vi in zip(x + (i - 0.5) * 0.38, v):
            ax.text(xi, vi + (1.5 if vi >= 0 else -1.5) * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 60, fmt.format(vi),
                    ha="center", va="bottom" if vi >= 0 else "top", fontsize=6, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [l for _, l in EXP], fontsize=6.2); ax.set_title(title); ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
axs[1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
axs[2].legend(loc="upper right", fontsize=7)
fig.savefig(RES / "fig_summary_E1_E3.pdf"); fig.savefig(RES / "fig_summary_E1_E3.png")
rows = {k: {m: R[k][m]["test_2019"] | {"delay_h": R[k][m]["median_hours_to_detection"], "distance_m": R[k][m]["median_distance_m"]} for m in ("GNN", "Ridge")} for k, _ in EXP}
(RES / "summary_E1_E3.json").write_text(json.dumps(rows, indent=1))
