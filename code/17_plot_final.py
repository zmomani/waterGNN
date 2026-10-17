"""Step 17 - figures and tables of the restructured results section.

Final method: tuned GNN with the normal residual. Comparisons: ridge and MLP with the same residual
and the same detector rules; the disagreement residual (first design) as an ablation.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

RES = ROOT / "results"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"GNN": "#2a78d6", "GNN original": "#9ec3ee", "Ridge": "#eb6834", "MLP": "#3a9b6b"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})
load = lambda n: json.loads((RES / f"{n}.json").read_text())

# conditions: (label, normal-residual tuned file, normal-residual original-GNN file, disagreement files tuned/original)
CASES = [("E1 continuous", "N_tuned_E1", "N_hide_E1", "T_E1", "E1_nodes"),
         ("E2 transfer", "N_tuned_E2", "N_hide_E2", "T_E2", "E2"),
         ("E3 intermittent S48", "N_tuned_E3", "N_hide_E3", "T_E3", "E3_blind"),
         ("E4 S24 unseen", "N_tuned_S24", "N_hide_S24", "T_S24", "R_S24_blind"),
         ("E4 S72 unseen", "N_tuned_S72", "N_hide_S72", "T_S72", "R_S72_blind"),
         ("E5 S48, 6 failed", "N_tuned_drop6_S48", "N_hide_drop6_S48", "T_drop6_S48", "R_drop6_S48"),
         ("E5 continuous, 6 failed", "T_drop6_CWS", None, "T_drop6_CWS", "R_drop6_CWS")]
rows = {}
for lab, nt, nh, dt, dh in CASES:
    r = {}
    if nt.startswith("N_"):
        d = load(f"detection_{nt}"); r["GNN"] = d["GNN"]; r["Ridge"] = d["Ridge"]
        if "MLP" in d:
            r["MLP"] = d["MLP"]
        r["GNN original"] = load(f"detection_{nh}")["GNN"]
    else:                                                  # continuous sensor failure: normal residual not run
        r = {}
    a = {}
    a["GNN"] = load(f"detection_{dt}")["GNN"]; o = load(f"detection_{dh}"); a["Ridge"] = o["Ridge"]; a["GNN original"] = o["GNN"]
    if "MLP" in o:
        a["MLP"] = o["MLP"]
    rows[lab] = dict(normal=r, disagreement=a)
(RES / "summary_final.json").write_text(json.dumps(rows, indent=1))


def cell(x):
    t = x["test_2019"]; return f"{t['TP']:.0f} & {t['FP']:.0f} & {100 * t['relative']:.0f}\\%"


print("TABLE normal residual (final method)")
for lab, r in rows.items():
    n = r["normal"]
    if not n:
        continue
    print(lab, "&", " & ".join(cell(n[m]) if m in n else "-- & -- & --" for m in ("GNN", "Ridge", "MLP")), "&",
          f"{100 * n['GNN original']['test_2019']['relative']:.0f}\\%", r"\\")
print("\nTABLE disagreement residual (ablation)")
for lab, r in rows.items():
    a = r["disagreement"]
    print(lab, "&", " & ".join(cell(a[m]) if m in a else "-- & -- & --" for m in ("GNN", "Ridge", "MLP")), "&",
          f"{100 * a['GNN original']['test_2019']['relative']:.0f}\\%", r"\\")

# ---- figure: final method, score and false alarms by condition, three models + original GNN
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(wspace=0.25))
labs = [l for l, *_ in CASES if rows[l]["normal"]]
x = np.arange(len(labs)); LOW = -60
short = {"E1 continuous": "E1\ncontin.", "E2 transfer": "E2\ntransfer", "E3 intermittent S48": "E3\nS48",
         "E4 S24 unseen": "E4\nS24", "E4 S72 unseen": "E4\nS72", "E5 S48, 6 failed": "E5\nS48, 6 failed"}
for ax, field, title in [(axs[0], "relative", "(a) Score (saving), % of maximum"), (axs[1], "FP", "(b) False alarms")]:
    for i, m in enumerate(("GNN", "Ridge", "MLP", "GNN original")):
        v = np.array([rows[l]["normal"][m]["test_2019"][field] * (100 if field == "relative" else 1) if m in rows[l]["normal"] else np.nan for l in labs])
        sh = np.maximum(v, LOW) if field == "relative" else v
        ax.bar(x + (i - 1.5) * 0.2, sh, width=0.19, color=C[m], label=m if m != "GNN" else "GNN (tuned)")
        for xi, vi, si in zip(x + (i - 1.5) * 0.2, v, sh):
            if np.isfinite(vi):
                ax.text(xi, si + (1.5 if si >= 0 else -1.5) * (1 if field == "relative" else 2), f"{vi:.0f}", ha="center",
                        va="bottom" if si >= 0 else "top", fontsize=4.8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x, [short[l] for l in labs], fontsize=6); ax.set_title(title)
    ax.grid(axis="x", visible=False)
axs[0].set_ylim(LOW - 8, 55); axs[1].set_yscale("symlog", linthresh=10)
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=6.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
fig.savefig(RES / "fig_final_detection.pdf"); fig.savefig(RES / "fig_final_detection.png")

# ---- figure: residual ablation, GNN tuned and ridge, both residuals
fig, ax = plt.subplots(figsize=(7.4, 2.5))
x = np.arange(len(labs))
series = [("GNN (tuned), normal residual", "normal", "GNN", C["GNN"], "-"), ("GNN (tuned), disagreement residual", "disagreement", "GNN", C["GNN"], "//"),
          ("Ridge, normal residual", "normal", "Ridge", C["Ridge"], "-"), ("Ridge, disagreement residual", "disagreement", "Ridge", C["Ridge"], "//")]
for i, (lab, res, m, col, hatch) in enumerate(series):
    v = np.array([rows[l][res][m]["test_2019"]["relative"] * 100 for l in labs]); sh = np.maximum(v, -60)
    ax.bar(x + (i - 1.5) * 0.2, sh, width=0.19, color=col, alpha=1 if hatch == "-" else 0.45, hatch=None if hatch == "-" else "///",
           edgecolor="white", lw=0.4, label=lab)
    for xi, vi, si in zip(x + (i - 1.5) * 0.2, v, sh):
        ax.text(xi, si + (1.5 if si >= 0 else -1.5), f"{vi:.0f}", ha="center", va="bottom" if si >= 0 else "top", fontsize=4.8, color=INK)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x, [short[l] for l in labs], fontsize=6); ax.grid(axis="x", visible=False)
ax.set_ylim(-68, 55); ax.set_ylabel("score (saving), % of maximum"); ax.set_title("Residual ablation")
fig.legend(*ax.get_legend_handles_labels(), fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.16))
fig.savefig(RES / "fig_residual_ablation.pdf"); fig.savefig(RES / "fig_residual_ablation.png")
