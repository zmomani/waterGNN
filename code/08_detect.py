"""Step 8 - leak detection and scoring (experiment E1: continuous supply).

  python 08_detect.py --tag cws_hide --noleak continuous_CWS_noleak --leaks continuous_CWS_leaks

Each sensor's expected reading is estimated from the other sensors (its group of
11 is hidden). The residual, expected minus measured, is standardised with
leak-free data and accumulated; an alarm names the pipe at the sensor with the
largest accumulated residual. Thresholds are tuned on the 2018 leaks and the
result is scored once on the 2019 leaks with the BattLeDIM rules.
"""
import argparse, itertools, json
import numpy as np
import pandas as pd
import torch
import gnn_data as g
import scoring
from gnn_model import Reconstructor
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="cws_hide")
ap.add_argument("--noleak", default="continuous_CWS_noleak")
ap.add_argument("--leaks", default="continuous_CWS_leaks")
ap.add_argument("--name", default="E1")
ap.add_argument("--every", type=int, default=12, help="evaluate every n-th 5-minute step")
args = ap.parse_args()
MODELS, RES = ROOT / "models", ROOT / "results"
norm = np.load(MODELS / f"norm_{args.tag}.npz"); MU, SD = norm["mu"], norm["sd"]
model = Reconstructor(g.EDGE_INDEX, g.N); model.load_state_dict(torch.load(MODELS / f"reconstructor_{args.tag}.pt")); model.eval()
SENSOR_PIPE = [next(n for n, l in scoring._wn.links() if s in (l.start_node_name, l.end_node_name)) for s in g.SENSORS]
SDIST = np.array([[scoring._DIST[a][b] for b in g.SENSORS] for a in g.SENSORS])      # pipe distance between sensors


def flat(run, k):
    """Inputs of the ridge baseline when group k is hidden."""
    keep = np.setdiff1d(np.arange(33), g.GROUPS[k])
    z = (run["sensors"] - MU[g.S_IDX]) / SD[g.S_IDX]
    return np.column_stack([z[:, keep], run["level"] / 4.0, g.time_features(run["time"]), np.ones(len(z))])


def expected_gnn(run):
    """Cached on disk: this is the slow step, and the workspace can restart."""
    cache = MODELS / f"expected_{args.tag}_{run['name']}_{args.every}.npy"
    if cache.exists():
        return np.load(cache)
    out = np.zeros_like(run["sensors"])
    with torch.no_grad():
        for i in range(0, len(run["time"]), 64):
            sub = {k: run[k][i:i + 64] for k in ("time", "sensors", "level")}
            x = torch.tensor(g.node_features(sub, MU, SD))
            for k in range(3):
                idx = g.S_IDX[g.GROUPS[k]]
                out[i:i + 64, g.GROUPS[k]] = model(g.hide(x, k))[:, idx].numpy() * SD[idx] + MU[idx]
            if i % 1280 == 0:
                print(run["name"], i, "/", len(run["time"]), flush=True)
    np.save(cache, out)
    return out


def expected_ridge(run, W):
    out = np.zeros_like(run["sensors"])
    for k in range(3):
        idx = g.S_IDX[g.GROUPS[k]]
        out[:, g.GROUPS[k]] = flat(run, k) @ W[k] * SD[idx] + MU[idx]
    return out


def detect(z, times, kappa, h, mute_radius=600.0, rearm_window=24):
    """One-sided cumulative sum per sensor. Returns [(pipe, time)]."""
    c = np.zeros(z.shape[1]); muted = np.zeros(z.shape[1], bool); out = []
    for t in range(len(z)):
        c[~muted] = np.maximum(0.0, c[~muted] + z[t, ~muted] - kappa)
        if muted.any() and t >= rearm_window:                      # re-arm when a sensor is back to normal
            calm = z[t - rearm_window:t].mean(0) < kappa
            c[muted & calm] = 0.0; muted &= ~calm
        if (c[~muted] > h).any():
            s = int(np.argmax(np.where(muted, -np.inf, c)))
            out.append((SENSOR_PIPE[s], times[t]))
            muted |= SDIST[s] <= mute_radius
    return out


# ------------------------------------------------------------- data ----------
ref = g.load_run(args.noleak, 2018, every=args.every); ref["name"] = f"{args.noleak}_2018"
val = g.load_run(args.leaks, 2018, every=args.every); val["name"] = f"{args.leaks}_2018"
test = g.load_run(args.leaks, 2019, every=args.every); test["name"] = f"{args.leaks}_2019"
labels = pd.read_csv(OUT / args.leaks / "leak_flows.csv.gz", index_col=0, parse_dates=True)
warm = int(np.searchsorted(ref["time"], np.datetime64("2018-01-29")))
split = int(np.searchsorted(ref["time"], np.datetime64("2018-11-01")))

tr = {k: v[warm:split] for k, v in ref.items() if k != "name"}
W = []
for k in range(3):
    A = flat(tr, k); Y = ((tr["sensors"] - MU[g.S_IDX]) / SD[g.S_IDX])[:, g.GROUPS[k]]
    W.append(np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y))

results, series = {}, {}
for mname, expect in [("GNN", expected_gnn), ("Ridge", lambda r: expected_ridge(r, W))]:
    r_ref = expect(ref) - ref["sensors"]
    m, s = r_ref[split:].mean(0), r_ref[split:].std(0)             # leak-free November-December
    z_val = (expect(val) - val["sensors"] - m) / s
    z_test = (expect(test) - test["sensors"] - m) / s
    best = None
    for kappa, h in itertools.product([0.5, 1.0, 1.5, 2.0], [50, 100, 200, 400, 800]):
        sc = scoring.score_detections(detect(z_val[warm:], val["time"][warm:], kappa, h), labels,
                                      "2018-01-29", "2018-12-31 23:55")
        if best is None or sc["score"] > best[0]:
            best = (sc["score"], kappa, h, sc)
    _, kappa, h, sc_val = best
    det = detect(z_test, test["time"], kappa, h)
    sc = scoring.score_detections(det, labels, "2019-01-01", "2019-12-31 23:55")
    d = sc.pop("detail"); sc_val.pop("detail")
    tp = d[d.kind == "TP"]
    starts = {p: max(labels.index[labels[p].to_numpy() > 0][0], pd.Timestamp("2019-01-01")) for p in tp.leak}
    delay = [(row.time - starts[row.leak]).total_seconds() / 3600 for row in tp.itertuples()]
    results[mname] = dict(kappa=kappa, h=h, validation_2018={k: round(float(v), 3) for k, v in sc_val.items()},
                          test_2019={k: round(float(v), 3) for k, v in sc.items()},
                          residual_sd_m=round(float(s.mean()), 4),
                          median_hours_to_detection=round(float(np.median(delay)), 1) if delay else None,
                          median_distance_m=round(float(tp.distance.median()), 1) if len(tp) else None)
    d.to_csv(RES / f"detections_{args.name}_{mname}.csv", index=False)
    series[mname] = dict(z=z_test.astype(np.float32), detail=d)
    print(mname, json.dumps(results[mname]), flush=True)
(RES / f"detection_{args.name}.json").write_text(json.dumps(results, indent=1))

# ------------------------------------------------------------- plots ---------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})
lf = labels.loc["2019"]
act = {p: lf.index[lf[p].to_numpy() > 0] for p in lf.columns}
act = dict(sorted(((p, t) for p, t in act.items() if len(t)), key=lambda kv: kv[1][0]))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, (mname, col) in zip(axs, [("GNN", BLUE), ("Ridge", ORANGE)]):
    d = series[mname]["detail"]
    for i, (p, t) in enumerate(act.items()):
        ax.plot([t[0], t[-1]], [i, i], color="#c9c8c3", lw=3.2, solid_capstyle="butt", zorder=1)
        hit = d[(d.kind == "TP") & (d.leak == p)]
        if len(hit):
            ax.scatter(hit.time, [i] * len(hit), s=22, color=col, edgecolor="white", lw=0.6, zorder=3)
    fp = d[d.kind == "FP"]
    ax.scatter(fp.time, [len(act) + 0.2] * len(fp), marker="x", s=18, color=INK, lw=0.9, zorder=3)
    r = results[mname]["test_2019"]
    ax.set_title(f"{mname}: {int(r['TP'])} of {int(r['leaks'])} found, {int(r['FP'])} false alarms", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01"))
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(list(range(len(act))) + [len(act) + 0.2], list(act) + ["false alarms"], fontsize=6.5)
axs[0].invert_yaxis()
fig.savefig(RES / f"fig_detection_{args.name}.pdf"); fig.savefig(RES / f"fig_detection_{args.name}.png")
