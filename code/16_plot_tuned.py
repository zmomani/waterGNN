"""Step 16 - the tuned GNN against the ridge baseline (and the original GNN) on every
dataset and scenario: reconstruction, robustness of reconstruction, detection, localisation."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

RES = ROOT / "results"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "W"          # "W": wide-grid detector settings, "T": original grid
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"GNN tuned": "#2a78d6", "GNN original": "#9ec3ee", "Ridge": "#eb6834", "MLP": "#3a9b6b"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})
load = lambda n: json.loads((RES / f"{n}.json").read_text())


def det(name_tuned, name_orig):
    """detection results: tuned GNN, original GNN, ridge (from the original-GNN run, identical settings), MLP"""
    t = load(f"detection_{name_tuned}"); o = load(f"detection_{name_orig}")
    out = {"GNN tuned": t["GNN"], "GNN original": o["GNN"], "Ridge": t["Ridge"]}
    if "MLP" in o:
        out["MLP"] = o["MLP"]
    return out


if PREFIX == "W":
    CASES = [("W_tuned_E3", "W_hide_E3", "S48\nas trained"), ("W_tuned_S24", "W_hide_S24", "S24\nunseen"),
             ("W_tuned_S72", "W_hide_S72", "S72\nunseen"), ("W_tuned_drop6_S48", "W_hide_drop6_S48", "S48\n6 failed"),
             ("W_tuned_E2", "W_hide_E2", "S48, model\nfrom contin."),
             ("W_tuned_E1", "W_hide_E1", "contin.\nas trained"), ("W_tuned_drop6_CWS", "W_hide_drop6_CWS", "contin.\n6 failed")]
else:
    CASES = [("T_E3", "E3_blind", "S48\nas trained"), ("T_S24", "R_S24_blind", "S24\nunseen"), ("T_S72", "R_S72_blind", "S72\nunseen"),
             ("T_drop6_S48", "R_drop6_S48", "S48\n6 failed"), ("T_E2", "E2", "S48, model\nfrom contin."),
             ("T_E1", "E1_nodes", "contin.\nas trained"), ("T_drop6_CWS", "R_drop6_CWS", "contin.\n6 failed")]
D = {lab: det(a, b) for a, b, lab in CASES}

sens_o, sens_t = load("robustness_sensors"), load("robustness_sensors_tuned")
sch_o, sch_t, mlp = load("robustness_schedule"), load("robustness_schedule_tuned"), load("robustness_mlp")
loc_o, loc_t = load("localisation_whole")["summary"], load("localisation_whole_tuned")["summary"]

fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.8), gridspec_kw=dict(wspace=0.28, hspace=0.55))
MODELS = ["GNN tuned", "GNN original", "Ridge"]

# (a) reconstruction error, as trained and under change -------------------------------------
ax = axs[0, 0]
conds = ["contin.\nas trained", "S48\nas trained", "S24\nunseen", "S72\nunseen", "S48\n6 failed", "contin.\n6 failed"]
SHORT = {"contin.\nas trained": "contin.", "S48\nas trained": "S48", "S24\nunseen": "S24\nunseen", "S72\nunseen": "S72\nunseen",
         "S48\n6 failed": "S48\n6 failed", "contin.\n6 failed": "contin.\n6 failed", "S48, model\nfrom contin.": "S48 from\ncontin."}
def rec(m, c):
    src_s, src_t = (sens_t, sch_t) if m == "GNN tuned" else (sens_o, sch_o); key = "GNN" if m.startswith("GNN") else "Ridge"
    if m == "MLP":
        return {"contin.\nas trained": mlp["sensors"]["CWS"]["0"]["mean"], "S48\nas trained": mlp["sensors"]["S48"]["0"]["mean"],
                "S24\nunseen": mlp["schedule"]["S48->S24"], "S72\nunseen": mlp["schedule"]["S48->S72"],
                "S48\n6 failed": mlp["sensors"]["S48"]["6"]["mean"], "contin.\n6 failed": mlp["sensors"]["CWS"]["6"]["mean"]}[c]
    return {"contin.\nas trained": src_s["CWS"]["0"][key]["mean"], "S48\nas trained": src_s["S48"]["0"][key]["mean"],
            "S24\nunseen": src_t["S48->S24"][key], "S72\nunseen": src_t["S48->S72"][key],
            "S48\n6 failed": src_s["S48"]["6"][key]["mean"], "contin.\n6 failed": src_s["CWS"]["6"][key]["mean"]}[c]
x = np.arange(len(conds)); w = 0.2
for i, m in enumerate(MODELS + ["MLP"]):
    v = [rec(m, c) for c in conds]
    ax.bar(x + (i - 1.5) * w, v, width=w * 0.92, color=C[m], label=m)
ax.set_yscale("log"); ax.set_xticks(x, [SHORT[c] for c in conds], fontsize=5.6); ax.grid(axis="x", visible=False)
ax.set_ylabel("error of expected pressure [m]"); ax.set_title("(a) Reconstruction accuracy")
ax.legend(fontsize=6, ncol=2, loc="upper left")

# (b) sensor failure curve, intermittent ---------------------------------------------------
ax = axs[0, 1]; K = [0, 3, 6, 11, 16]
for m, src, key in [("GNN tuned", sens_t, "GNN"), ("GNN original", sens_o, "GNN"), ("Ridge", sens_o, "Ridge")]:
    ax.plot(K, [src["S48"][str(k)][key]["mean"] for k in K], "-o", ms=3, lw=1.3, color=C[m], label=m)
    ax.fill_between(K, [src["S48"][str(k)][key]["min"] for k in K], [src["S48"][str(k)][key]["max"] for k in K], color=C[m], alpha=0.15, lw=0)
ax.plot(K, [sens_o["S48"][str(k)]["Ridge_refitted"]["mean"] for k in K], "--", color=C["Ridge"], lw=1.1, label="Ridge, refitted")
ax.set_yscale("log"); ax.set_xticks(K); ax.set_xlabel("failed sensors, of 33"); ax.set_title("(b) Sensors fail, intermittent S48")
ax.legend(fontsize=6)

# (c) detection score ------------------------------------------------------------------------
ax = axs[1, 0]; x = np.arange(len(CASES)); LOW = -100
for i, m in enumerate(MODELS):
    v = np.array([D[lab][m]["test_2019"]["relative"] * 100 for _, _, lab in CASES]); s = np.maximum(v, LOW)
    ax.bar(x + (i - 1) * 0.27, s, width=0.25, color=C[m], label=m)
    for xi, vi, si in zip(x + (i - 1) * 0.27, v, s):
        ax.text(xi, si + (2 if si >= 0 else -2), f"{vi:.0f}", ha="center", va="bottom" if si >= 0 else "top", fontsize=5, color=INK)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x, [SHORT[lab] for _, _, lab in CASES], fontsize=5.6); ax.grid(axis="x", visible=False)
ax.set_ylim(LOW - 15, 65); ax.set_ylabel("score, % of maximum"); ax.set_title("(c) Detection (2019 test year)")
ax.axvline(4.5, color=GRID, lw=1)

# (d) localisation ---------------------------------------------------------------------------
ax = axs[1, 1]
lconds = [("S48 as trained", "E3 S48"), ("S24 unseen", "S24 unseen"), ("S72 unseen", "S72 unseen"),
          ("S48, 6 failed", "S48, 6 sensors failed"), ("contin. as trained", "E1 continuous"), ("contin., 6 failed", "Continuous, 6 sensors failed")]
x = np.arange(len(lconds))
for i, (m, src, key) in enumerate([("GNN tuned", loc_t, "GNN"), ("GNN original", loc_o, "GNN"), ("Ridge", loc_o, "Ridge")]):
    v = [src[k][key]["peak_within_300"] for _, k in lconds]
    ax.bar(x + (i - 1) * 0.27, v, width=0.25, color=C[m], label=m)
    for xi, vi in zip(x + (i - 1) * 0.27, v):
        ax.text(xi, vi + 0.3, f"{vi}", ha="center", va="bottom", fontsize=5.5, color=INK)
ax.set_xticks(x, [l.replace(", ", ",\n").replace(" as ", "\nas ").replace(" unseen", "\nunseen") for l, _ in lconds], fontsize=5.6)
ax.grid(axis="x", visible=False); ax.set_ylabel("leaks located within 300 m, of 23"); ax.set_title("(d) Localisation, all 23 leaks")
ax.axvline(3.5, color=GRID, lw=1); ax.set_ylim(0, 23)
fig.savefig(RES / f"fig_tuned_vs_baseline_{PREFIX}.pdf"); fig.savefig(RES / f"fig_tuned_vs_baseline_{PREFIX}.png")

rows = {}
for a, b, lab in CASES:
    rows[lab.replace("\n", " ")] = {m: dict(D[lab][m]["test_2019"], delay_h=D[lab][m]["median_hours_to_detection"],
                                           distance_m=D[lab][m]["median_distance_m"], kappa=D[lab][m]["kappa"], h=D[lab][m]["h"])
                                    for m in D[lab]}
(RES / f"summary_tuned_{PREFIX}.json").write_text(json.dumps(rows, indent=1))
for lab, r in rows.items():
    for m, v in r.items():
        print(f"{lab:26s} {m:13s} TP {v['TP']:2.0f} FP {v['FP']:5.0f} share {100*v['relative']:6.0f}%  kappa {v['kappa']} h {v['h']}")
