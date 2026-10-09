"""Step 7 - plots of the reconstruction results."""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gnn_data as g
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="cws"); ap.add_argument("--run", default="continuous_CWS_noleak")
args = ap.parse_args()
RES = ROOT / "results"
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})
res = json.loads((RES / f"reconstruction_{args.tag}.json").read_text())
err = np.load(RES / f"reconstruction_{args.tag}_errors.npz")
hist = res["history"]

# reference: always predicting each node's mean pressure
run = g.load_run(args.run, 2018, every=12)
split = int(np.searchsorted(run["time"], np.datetime64("2018-11-01")))
mean_err = np.abs(run["truth"][split:] - run["truth"][:split].mean(0)).mean(0)
res["summary"]["mean_only_mae_unmonitored_m"] = round(float(mean_err[g.UNMONITORED].mean()), 3)
(RES / f"reconstruction_{args.tag}.json").write_text(json.dumps(res, indent=1))

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15], wspace=0.35))
ax = axs[0]
ax.plot([h["minutes"] for h in hist], [h["val_mae_unmonitored_m"] for h in hist], color=BLUE, lw=1.6, marker="o", ms=3)
ax.axhline(res["summary"]["ridge_mae_unmonitored_m"], color=ORANGE, lw=1.2)
ax.text(hist[-1]["minutes"], res["summary"]["ridge_mae_unmonitored_m"] * 1.25, "ridge baseline", color=INK, ha="right", fontsize=7)
ax.text(hist[5]["minutes"], hist[4]["val_mae_unmonitored_m"], "GNN", color=INK, fontsize=7)
ax.set_yscale("log"); ax.set_xlabel("Training time (min)"); ax.set_ylabel("Validation error (m)")
ax.set_title("(a) Training")
vmax = float(np.percentile(err["gnn"][g.IS_JUNCTION], 98))
for ax, key, title in [(axs[1], "gnn", "(b) GNN error by node"), (axs[2], "ridge", "(c) Ridge error by node")]:
    for a, b in g.LINKS:
        ax.plot(*zip(g.XY[g.IDX[a]], g.XY[g.IDX[b]]), color="#d5d4cf", lw=0.4, zorder=1)
    sc = ax.scatter(g.XY[g.IS_JUNCTION, 0], g.XY[g.IS_JUNCTION, 1], s=3, c=err[key][g.IS_JUNCTION], cmap="Blues",
                    vmin=0, vmax=vmax, lw=0, zorder=2)
    ax.scatter(g.XY[g.S_IDX, 0], g.XY[g.S_IDX, 1], s=5, color=INK, zorder=3, lw=0)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
cb = fig.colorbar(sc, ax=axs[1:], fraction=0.025, pad=0.01); cb.set_label("Mean absolute error (m)", fontsize=7)
cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
fig.savefig(RES / f"fig_reconstruction_{args.tag}.pdf"); fig.savefig(RES / f"fig_reconstruction_{args.tag}.png")
print(json.dumps(res["summary"], indent=1))
