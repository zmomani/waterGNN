"""Step 9 - detection with a residual at every node (experiment E1, second form).

  python 09_detect_nodes.py

In 08_detect.py an alarm could only name the pipe at a sensor. Here each model
produces two pressure fields over all nodes: an expected one, estimated with a
group of sensors hidden (averaged over the three groups), and an observed one,
estimated from all sensors. Their difference is a residual at every node, so an
alarm can name the pipe at any node. The ridge baseline does the same with
linear maps from sensors to nodes.
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
ap.add_argument("--name", default="E1_nodes")
ap.add_argument("--every", type=int, default=12)
args = ap.parse_args()
MODELS, RES = ROOT / "models", ROOT / "results"
norm = np.load(MODELS / f"norm_{args.tag}.npz"); MU, SD = norm["mu"], norm["sd"]
model = Reconstructor(g.EDGE_INDEX, g.N); model.load_state_dict(torch.load(MODELS / f"reconstructor_{args.tag}.pt")); model.eval()
J = np.flatnonzero(g.IS_JUNCTION)                                    # the 782 junctions
JN = [g.NODES[i] for i in J]
NODE_PIPE = {}
for name, l in scoring._wn.links():
    NODE_PIPE.setdefault(l.start_node_name, name); NODE_PIPE.setdefault(l.end_node_name, name)
PIPES = [NODE_PIPE[n] for n in JN]
NDIST = np.array([[scoring._DIST[a][b] for b in JN] for a in JN], np.float32)


def field_gnn(run):
    """Residual field [T, junctions] in units of each node's normal spread."""
    cache = MODELS / f"field_{args.tag}_{run['name']}_{args.every}.npy"
    if cache.exists():
        return np.load(cache)
    out = np.zeros((len(run["time"]), len(J)), np.float32)
    with torch.no_grad():
        for i in range(0, len(run["time"]), 64):
            sub = {k: run[k][i:i + 64] for k in ("time", "sensors", "level")}
            x = torch.tensor(g.node_features(sub, MU, SD))
            observed = model(x)
            expected = sum(model(g.hide(x, k)) for k in range(3)) / 3
            out[i:i + 64] = (expected - observed)[:, J].numpy()
            if i % 1280 == 0:
                print(run["name"], i, "/", len(run["time"]), flush=True)
    np.save(cache, out)
    return out


def flat(run, keep):
    z = (run["sensors"] - MU[g.S_IDX]) / SD[g.S_IDX]
    return np.column_stack([z[:, keep], run["level"] / 4.0, g.time_features(run["time"]), np.ones(len(z))])


def fit_ridge(tr):
    Y = ((tr["truth"] - MU) / SD)[:, J]
    maps = []
    for keep in [np.arange(33)] + [np.setdiff1d(np.arange(33), g.GROUPS[k]) for k in range(3)]:
        A = flat(tr, keep)
        maps.append((keep, np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y)))
    return maps


def field_ridge(run, maps):
    observed = flat(run, maps[0][0]) @ maps[0][1]
    expected = sum(flat(run, keep) @ W for keep, W in maps[1:]) / 3
    return (expected - observed).astype(np.float32)


def detect(z, times, kappa, h, mute_radius=600.0, rearm_window=24):
    c = np.zeros(z.shape[1]); muted = np.zeros(z.shape[1], bool); out = []
    for t in range(len(z)):
        c[~muted] = np.maximum(0.0, c[~muted] + z[t, ~muted] - kappa)
        if muted.any() and t >= rearm_window and t % 6 == 0:
            calm = z[t - rearm_window:t].mean(0) < kappa
            c[muted & calm] = 0.0; muted &= ~calm
        if (c[~muted] > h).any():
            s = int(np.argmax(np.where(muted, -np.inf, c)))
            out.append((PIPES[s], times[t]))
            muted |= NDIST[s] <= mute_radius
    return out


ref = g.load_run(args.noleak, 2018, every=args.every); ref["name"] = f"{args.noleak}_2018"
val = g.load_run(args.leaks, 2018, every=args.every); val["name"] = f"{args.leaks}_2018"
test = g.load_run(args.leaks, 2019, every=args.every); test["name"] = f"{args.leaks}_2019"
labels = pd.read_csv(OUT / args.leaks / "leak_flows.csv.gz", index_col=0, parse_dates=True)
warm = int(np.searchsorted(ref["time"], np.datetime64("2018-01-29")))
split = int(np.searchsorted(ref["time"], np.datetime64("2018-11-01")))
maps = fit_ridge({k: v[warm:split] for k, v in ref.items() if k != "name"})

results, series = {}, {}
for mname, field in [("GNN", field_gnn), ("Ridge", lambda r: field_ridge(r, maps))]:
    f_ref = field(ref)
    m, s = f_ref[split:].mean(0), np.maximum(f_ref[split:].std(0), 1e-4)
    z_val, z_test = (field(val) - m) / s, (field(test) - m) / s
    best = None
    # four settings, all chosen on the 2018 leaks: slack, threshold, how far an alarm
    # silences its surroundings, and how long a node must be calm before it is re-armed
    for kappa, h, rad, calm in itertools.product([0.5, 1.0, 1.5, 2.0], [50, 200, 800, 3200], [600.0, 1200.0], [24, 168]):
        sc = scoring.score_detections(detect(z_val[warm:], val["time"][warm:], kappa, h, rad, calm), labels,
                                      "2018-01-29", "2018-12-31 23:55")
        if best is None or sc["score"] > best[0]:
            best = (sc["score"], kappa, h, sc, rad, calm)
    _, kappa, h, sc_val, rad, calm = best
    sc = scoring.score_detections(detect(z_test, test["time"], kappa, h, rad, calm), labels, "2019-01-01", "2019-12-31 23:55")
    d = sc.pop("detail"); sc_val.pop("detail")
    tp = d[d.kind == "TP"]
    starts = {p: max(labels.index[labels[p].to_numpy() > 0][0], pd.Timestamp("2019-01-01")) for p in tp.leak}
    delay = [(row.time - starts[row.leak]).total_seconds() / 3600 for row in tp.itertuples()]
    results[mname] = dict(kappa=kappa, h=h, mute_radius_m=rad, calm_hours=calm, validation_2018={k: round(float(v), 3) for k, v in sc_val.items()},
                          test_2019={k: round(float(v), 3) for k, v in sc.items()},
                          median_hours_to_detection=round(float(np.median(delay)), 1) if delay else None,
                          median_distance_m=round(float(tp.distance.median()), 1) if len(tp) else None)
    d.to_csv(RES / f"detections_{args.name}_{mname}.csv", index=False)
    series[mname] = d
    print(mname, json.dumps(results[mname]), flush=True)
(RES / f"detection_{args.name}.json").write_text(json.dumps(results, indent=1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.dpi": 200, "savefig.bbox": "tight"})
lf = labels.loc["2019"]
act = {p: lf.index[lf[p].to_numpy() > 0] for p in lf.columns}
act = dict(sorted(((p, t) for p, t in act.items() if len(t)), key=lambda kv: kv[1][0]))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, (mname, col) in zip(axs, [("GNN", BLUE), ("Ridge", ORANGE)]):
    d = series[mname]
    for i, (p, t) in enumerate(act.items()):
        ax.plot([t[0], t[-1]], [i, i], color="#c9c8c3", lw=3.2, solid_capstyle="butt", zorder=1)
        hit = d[(d.kind == "TP") & (d.leak == p)]
        if len(hit):
            ax.scatter(hit.time, [i] * len(hit), s=22, color=col, edgecolor="white", lw=0.6, zorder=3)
    fp = d[d.kind == "FP"]
    ax.scatter(fp.time, [len(act) + 0.2] * len(fp), marker="x", s=18, color=INK, lw=0.9, zorder=3)
    r = results[mname]["test_2019"]
    ax.set_title(f"{mname}: {int(r['TP'])} of {int(r['leaks'])} found, {int(r['FP'])} false alarms")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01"))
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(list(range(len(act))) + [len(act) + 0.2], list(act) + ["false alarms"], fontsize=6.5)
axs[0].invert_yaxis()
fig.savefig(RES / f"fig_detection_{args.name}.pdf"); fig.savefig(RES / f"fig_detection_{args.name}.png")
