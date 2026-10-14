"""Step 15 - hyperparameter sweep of the GNN: comparison table and convergence curves.

Validation error is the mean absolute error of the expected pressure at unmonitored,
pressurised nodes in the leak-free November-December 2018 of the S48 run. The 2019
test year is not used.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

RES = ROOT / "results"
INK, MUTED, GRID, BLUE, ORANGE, GREEN = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#3a9b6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})

RUNS = [("s48_hide", "K10, 4 layers, 64 ch (original)"),
        ("sw_k3_l4_c64", "K3, 4 layers, 64 ch"), ("sw_k5_l8_c64", "K5, 8 layers, 64 ch"), ("sw_k10_l2_c64", "K10, 2 layers, 64 ch"),
        ("sw_k10_l4_c128", "K10, 4 layers, 128 ch"), ("sw_k3_l8_c128", "K3, 8 layers, 128 ch"),
        ("sw_k10_l4_c64_lr5e-4", "original, learning rate 5e-4"), ("sw_k10_l4_c64_wd1e-4", "original, weight decay 1e-4"),
        ("sw_k10_l4_c64_e16", "original + node embedding 16"), ("sw_k3_l4_c64_e16", "K3, 4 layers, 64 ch + emb. 16"),
        ("sw_k5_l8_c64_e16", "K5, 8 layers, 64 ch + emb. 16"), ("sw_k10_l4_c128_e16", "K10, 4 layers, 128 ch + emb. 16"),
        ("sw_k10_l4_c128_e64", "K10, 4 layers, 128 ch + emb. 64"),
        ("sw_k10_l4_c128_e16_ep36", "K10, 4 layers, 128 ch + emb. 16, 36 epochs"),
        ("sw_k10_l4_c128_e64_ep36", "K10, 4 layers, 128 ch + emb. 64, 36 epochs")]
rows = []
for tag, label in RUNS:
    f = RES / f"reconstruction_{tag}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text()); h = d["history"]; s = d["summary"]
    rows.append(dict(tag=tag, label=label, epochs=len(h), steps=len(h) * (s["samples_per_epoch"] // 32),
                     params=s["parameters"], minutes=s["training_minutes"],
                     final=h[-1]["val_mae_unmonitored_m"], best=min(r["val_mae_unmonitored_m"] for r in h),
                     history=[r["val_mae_unmonitored_m"] for r in h], train=[r["train_loss"] for r in h]))
rows.sort(key=lambda r: r["final"])
(RES / "sweep_S48.json").write_text(json.dumps(rows, indent=1))
print(f"{'configuration':45s} {'params':>8s} {'steps':>6s} {'min':>5s} {'val MAE [m]':>12s}")
for r in rows:
    print(f"{r['label']:45s} {r['params']:8d} {r['steps']:6d} {r['minutes']:5.1f} {r['final']:12.3f}")

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.3, width_ratios=[1.15, 1]))
ax = axs[0]
y = np.arange(len(rows))
cols = [BLUE if "emb" in r["label"] else "#9aa3ad" for r in rows]
ax.barh(y, [r["final"] for r in rows], color=cols, height=0.7)
for yi, r in zip(y, rows):
    ax.text(r["final"] + 0.01, yi, f"{r['final']:.2f}", va="center", fontsize=6, color=INK)
ax.set_yticks(y, [r["label"] for r in rows], fontsize=5.8); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("validation error of expected pressure [m]"); ax.set_title("(a) Configurations, intermittent S48")
ax.axvline(0.085, color=ORANGE, lw=1, ls="--"); ax.text(0.087, len(rows) - 0.6, "ridge", color=ORANGE, fontsize=6)
ax.axvline(0.022, color=GREEN, lw=1, ls="--"); ax.text(0.024, len(rows) - 0.6, "MLP", color=GREEN, fontsize=6)
ax = axs[1]
show = [r for r in rows if r["tag"] in ("s48_hide", "sw_k10_l4_c128", "sw_k10_l4_c64_e16", "sw_k10_l4_c128_e16",
                                        "sw_k10_l4_c128_e16_ep36", "sw_k10_l4_c128_e64_ep36")]
for r, c in zip(show, [INK, "#9aa3ad", "#7fb3ea", BLUE, "#1b4f8f", "#5a2d82"]):
    ax.plot(np.arange(1, len(r["history"]) + 1) * (r["steps"] // r["epochs"]), r["history"], color=c, lw=1.3,
            label=r["label"].replace("K10, 4 layers, ", ""))
ax.set_yscale("log"); ax.set_xlabel("training steps"); ax.set_ylabel("validation error [m]")
ax.set_title("(b) Convergence"); ax.legend(fontsize=5.6, loc="upper right")
fig.savefig(RES / "fig_sweep.pdf"); fig.savefig(RES / "fig_sweep.png")
