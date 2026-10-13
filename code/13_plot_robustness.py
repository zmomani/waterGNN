"""Step 13 - figures and summary table for the robustness tests (steps 10 and 12)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

RES = ROOT / "results"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = dict(GNN="#2a78d6", Ridge="#eb6834", MLP="#3a9b6b", Ridge_refitted="#eb6834")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})
load = lambda n: json.loads((RES / f"{n}.json").read_text())
sens, sched, mlp = load("robustness_sensors"), load("robustness_schedule"), load("robustness_mlp")

# ---------------------------------------------------------------- reconstruction
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw=dict(wspace=0.42))
K = [0, 3, 6, 11, 16]
for ax, regime, title in [(axs[0], "CWS", "(a) Continuous: sensors fail"), (axs[1], "S48", "(b) Intermittent: sensors fail")]:
    series = {m: sens[regime] for m in ("GNN", "Ridge", "Ridge_refitted")}
    for m, ls, lab in [("GNN", "-", "GNN"), ("Ridge", "-", "Ridge"), ("Ridge_refitted", "--", "Ridge, refitted"), ("MLP", "-", "MLP")]:
        get = (lambda k, f: mlp["sensors"][regime][str(k)][f]) if m == "MLP" else (lambda k, f: sens[regime][str(k)][m][f])
        ax.plot(K, [get(k, "mean") for k in K], ls, color=COL[m], lw=1.4, marker="o", ms=3, label=lab)
        ax.fill_between(K, [get(k, "min") for k in K], [get(k, "max") for k in K], color=COL[m], alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xticks(K); ax.set_xlabel("failed sensors, of 33"); ax.set_title(title)
axs[0].set_ylabel("error of expected pressure [m]"); fig.legend(*axs[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
ax = axs[2]; T = ["S24", "S48", "S72"]; x = np.arange(3)
vals = dict(GNN=[sched[f"S48->{t}"]["GNN"] for t in T], Ridge=[sched[f"S48->{t}"]["Ridge"] for t in T],
            MLP=[mlp["schedule"][f"S48->{t}"] for t in T])
for i, m in enumerate(("GNN", "Ridge", "MLP")):
    ax.bar(x + (i - 1) * 0.27, vals[m], width=0.24, color=COL[m])
    for xi, v in zip(x + (i - 1) * 0.27, vals[m]):
        ax.text(xi, v * 1.12, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK)
ax.set_yscale("log"); ax.set_xticks(x, ["S24\n(unseen)", "S48\n(trained)", "S72\n(unseen)"]); ax.grid(axis="x", visible=False)
ax.set_title("(c) Schedule changes"); ax.margins(y=0.2)
fig.savefig(RES / "fig_robust_reconstruction.pdf"); fig.savefig(RES / "fig_robust_reconstruction.png")

# --------------------------------------------------------------------- detection
CASES = [("E3_blind", "S48\nas trained"), ("R_S24_blind", "S24\nunseen"), ("R_S72_blind", "S72\nunseen"),
         ("R_drop6_S48", "S48\n6 failed"), ("E1", "contin.\nas trained"), ("R_drop6_CWS", "contin.\n6 failed")]
det = {}
for key, _ in CASES:
    if key == "E1":
        det[key] = load("detection_E1_nodes") | load("detection_E1_MLP")
    else:
        det[key] = load(f"detection_{key}")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw=dict(wspace=0.25))
x = np.arange(len(CASES)); LOW = -100
for ax, field, title in [(axs[0], "relative", "(a) Score, % of maximum"), (axs[1], "FP", "(b) False alarms")]:
    for i, m in enumerate(("GNN", "Ridge", "MLP")):
        v = np.array([det[k][m]["test_2019"][field] for k, _ in CASES]) * (100 if field == "relative" else 1)
        shown = np.maximum(v, LOW) if field == "relative" else v
        ax.bar(x + (i - 1) * 0.27, shown, width=0.24, color=COL[m], label=m)
        for xi, vi, si in zip(x + (i - 1) * 0.27, v, shown):
            ax.text(xi, si + (1 if si >= 0 else -1) * (2.5 if field == "relative" else 0), f"{vi:.0f}", ha="center",
                    va="bottom" if si >= 0 else "top", fontsize=5.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x, [l for _, l in CASES], fontsize=6.2); ax.set_title(title)
    ax.grid(axis="x", visible=False); ax.axvline(3.5, color=GRID, lw=1)
axs[0].set_ylim(LOW - 12, 60); axs[1].set_yscale("symlog", linthresh=10); fig.legend(*axs[1].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
fig.savefig(RES / "fig_robust_detection.pdf"); fig.savefig(RES / "fig_robust_detection.png")

rows = {k: {m: det[k][m]["test_2019"] | dict(delay_h=det[k][m]["median_hours_to_detection"], distance_m=det[k][m]["median_distance_m"])
            for m in ("GNN", "Ridge", "MLP")} for k, _ in CASES}
(RES / "summary_robustness.json").write_text(json.dumps(rows, indent=1))
for k in rows:
    for m in rows[k]:
        r = rows[k][m]; print(f"{k:12s} {m:6s} TP {r['TP']:.0f} FP {r['FP']:.0f} score {r['score']:.0f} rel {100*r['relative']:.0f}% delay {r['delay_h']} dist {r['distance_m']}")
