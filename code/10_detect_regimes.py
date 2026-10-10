"""Step 10 - detection across supply regimes (experiments E1 to E4).

  E1  --tag cws_hide --ref continuous_CWS_noleak --val continuous_CWS_leaks --test continuous_CWS_leaks
  E2  --tag cws_hide --ref continuous_CWS_noleak --val continuous_CWS_leaks --test intermittent_S48_leaks
  E3  --tag s48_hide --ref intermittent_S48_noleak --val intermittent_S48_leaks --test intermittent_S48_leaks [--aware]

ref: leak-free 2018 run that fits the ridge baseline and the residual statistics.
val: 2018 run with leaks, used to choose the alarm settings.
test: run whose 2019 leaks are scored. Everything is taken at pressurised nodes
only: an unpressurised node neither adds to nor resets its accumulated residual.
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
ap.add_argument("--tag", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--ref", required=True); ap.add_argument("--val", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--aware", action="store_true")
ap.add_argument("--every", type=int, default=12)
args = ap.parse_args()
MODELS, RES = ROOT / "models", ROOT / "results"
norm = np.load(MODELS / f"norm_{args.tag}.npz"); MU, SD = norm["mu"], norm["sd"]
model = Reconstructor(g.EDGE_INDEX, g.N, in_channels=13 if args.aware else 11)
model.load_state_dict(torch.load(MODELS / f"reconstructor_{args.tag}.pt")); model.eval()
J = np.flatnonzero(g.IS_JUNCTION); JN = [g.NODES[i] for i in J]
NODE_PIPE = {}
for name, l in scoring._wn.links():
    NODE_PIPE.setdefault(l.start_node_name, name); NODE_PIPE.setdefault(l.end_node_name, name)
PIPES = [NODE_PIPE[n] for n in JN]
NDIST = np.array([[scoring._DIST[a][b] for b in JN] for a in JN], np.float32)
KEYS = ("time", "sensors", "level", "live", "since")


def field_gnn(run):
    cache = MODELS / f"field_{args.tag}_{run['name']}_{args.every}.npy"
    if cache.exists():
        return np.load(cache)
    out = np.zeros((len(run["time"]), len(J)), np.float32)
    with torch.no_grad():
        for i in range(0, len(run["time"]), 64):
            x = torch.tensor(g.node_features({k: run[k][i:i + 64] for k in KEYS}, MU, SD, args.aware))
            out[i:i + 64] = (sum(model(g.hide(x, k)) for k in range(3)) / 3 - model(x))[:, J].numpy()
            if i % 1920 == 0:
                print(run["name"], i, "/", len(run["time"]), flush=True)
    np.save(cache, out)
    return out


def flat(run, keep):
    ok = run["live"][:, g.S_IDX]
    z = np.where(ok, (run["sensors"] - MU[g.S_IDX]) / SD[g.S_IDX], 0.0)
    cols = [z[:, keep], run["level"] / 4.0, g.time_features(run["time"]), np.ones(len(z))]
    if args.aware:                                   # the baseline's form of supply awareness
        cols += [ok[:, keep].astype(float), run["state"].astype(float)]
    return np.column_stack(cols)


GROUP_OF = np.where(g.ALWAYS[J], -1, g.SECTOR[J])     # nodes that share a supply state share a fit


def fit_ridge(tr):
    Y = ((tr["truth"] - MU) / SD)[:, J]; L = tr["live"][:, J]
    maps = []
    for keep in [np.arange(33)] + [np.setdiff1d(np.arange(33), g.GROUPS[k]) for k in range(3)]:
        A = flat(tr, keep); W = np.zeros((A.shape[1], len(J)))
        for grp in np.unique(GROUP_OF):
            cols = np.flatnonzero(GROUP_OF == grp); rows = L[:, cols[0]]
            Ar = A[rows]
            W[:, cols] = np.linalg.solve(Ar.T @ Ar + 1e-3 * np.eye(A.shape[1]), Ar.T @ Y[rows][:, cols])
        maps.append((keep, W))
    return maps


def field_ridge(run, maps):
    return (sum(flat(run, keep) @ W for keep, W in maps[1:]) / 3 - flat(run, maps[0][0]) @ maps[0][1]).astype(np.float32)


def detect(z, live, times, kappa, h, mute_radius, calm_window):
    c = np.zeros(z.shape[1]); muted = np.zeros(z.shape[1], bool); out = []
    for t in range(len(z)):
        upd = live[t] & ~muted
        c[upd] = np.maximum(0.0, c[upd] + z[t, upd] - kappa)
        if muted.any() and t >= calm_window and t % 6 == 0:
            lw = live[t - calm_window:t]; n = lw.sum(0)
            mean = (z[t - calm_window:t] * lw).sum(0) / np.maximum(n, 1)
            calm = (n > 0) & (mean < kappa)
            c[muted & calm] = 0.0; muted &= ~calm
        if (c[~muted] > h).any():
            s = int(np.argmax(np.where(muted, -np.inf, c)))
            out.append((PIPES[s], times[t]))
            muted |= NDIST[s] <= mute_radius
    return out


def load(name, year):
    r = g.load_run(name, year, every=args.every); r["name"] = f"{name}_{year}"; return r

ref, val, test = load(args.ref, 2018), load(args.val, 2018), load(args.test, 2019)
lab_val = pd.read_csv(OUT / args.val / "leak_flows.csv.gz", index_col=0, parse_dates=True)
lab_test = pd.read_csv(OUT / args.test / "leak_flows.csv.gz", index_col=0, parse_dates=True)
warm = int(np.searchsorted(ref["time"], np.datetime64("2018-01-29")))
split = int(np.searchsorted(ref["time"], np.datetime64("2018-11-01")))
maps = fit_ridge({k: v[warm:split] for k, v in ref.items() if k != "name"})

results, series = {}, {}
for mname, field in [("GNN", field_gnn), ("Ridge", lambda r: field_ridge(r, maps))]:
    f_ref, l_ref = field(ref)[split:], ref["live"][split:][:, J]
    n = np.maximum(l_ref.sum(0), 1)
    m = (f_ref * l_ref).sum(0) / n
    s = np.maximum(np.sqrt((((f_ref - m) ** 2) * l_ref).sum(0) / n), 1e-4)
    zs = {}
    for key, run in (("val", val), ("test", test)):
        zs[key] = np.where(run["live"][:, J], np.clip((field(run) - m) / s, -50, 50), 0.0)
    best = None
    for kappa, h, rad, calm in itertools.product([0.5, 1.0, 1.5, 2.0], [50, 200, 800, 3200], [600.0, 1200.0], [24, 168]):
        sc = scoring.score_detections(detect(zs["val"][warm:], val["live"][warm:][:, J], val["time"][warm:], kappa, h, rad, calm),
                                      lab_val, "2018-01-29", "2018-12-31 23:55")
        if best is None or sc["score"] > best[0]:
            best = (sc["score"], kappa, h, sc, rad, calm)
    _, kappa, h, sc_val, rad, calm = best
    sc = scoring.score_detections(detect(zs["test"], test["live"][:, J], test["time"], kappa, h, rad, calm),
                                  lab_test, "2019-01-01", "2019-12-31 23:55")
    d = sc.pop("detail"); sc_val.pop("detail")
    tp = d[d.kind == "TP"]
    starts = {p: max(lab_test.index[lab_test[p].to_numpy() > 0][0], pd.Timestamp("2019-01-01")) for p in tp.leak}
    delay = [(row.time - starts[row.leak]).total_seconds() / 3600 for row in tp.itertuples()]
    results[mname] = dict(kappa=kappa, h=h, mute_radius_m=rad, calm_hours=calm,
                          validation_2018={k: round(float(v), 3) for k, v in sc_val.items()},
                          test_2019={k: round(float(v), 3) for k, v in sc.items()},
                          median_hours_to_detection=round(float(np.median(delay)), 1) if delay else None,
                          median_distance_m=round(float(tp.distance.median()), 1) if len(tp) else None)
    d.to_csv(RES / f"detections_{args.name}_{mname}.csv", index=False); series[mname] = d
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
lf = lab_test.loc["2019"]
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
