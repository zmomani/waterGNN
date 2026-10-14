"""Step 14 - localisation on its own, independent of the alarm threshold.

For every leak of the 2019 test year, the residual field of each model is averaged
over the active period of the leak (or its last week) (at pressurised nodes) and the node with the largest
mean residual is taken as the estimate. Reported: distance along the pipes from the
estimated pipe to the leaking pipe, for all 23 leaks, in every condition for which
fields are cached by step 10.
"""
import importlib, json, os
import numpy as np
import pandas as pd
import torch
import gnn_data as g
import scoring
from common import *

rb = importlib.import_module("12_robustness")
MODELS, RES = ROOT / "models", ROOT / "results"
SET = os.environ.get("GNN_SET", "hide")
J = rb.J; JN = [g.NODES[i] for i in J]
NODE_PIPE = {}
for name, l in scoring._wn.links():
    NODE_PIPE.setdefault(l.start_node_name, name); NODE_PIPE.setdefault(l.end_node_name, name)
PIPES = [NODE_PIPE[n] for n in JN]
WEEK = 168
import sys
WINDOW = sys.argv[1] if len(sys.argv) > 1 else "whole"   # "whole" active period or "last_week"


def rows_all(run):
    return np.arange(len(run["time"]))


def mlp_net(tag):
    net = torch.nn.Sequential(torch.nn.Linear(71, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, len(J)))
    net.load_state_dict(torch.load(MODELS / f"mlp_{tag.replace('_tuned', '_hide')}.pt")); return net.eval()


def fields(tag, train_regime, test_name, dead=None):
    """Residual fields (expected minus observed) of the three models on the 2019 test run."""
    ref = rb.load(train_regime); test = g.load_run(test_name, 2019, every=12)
    if dead is not None:
        test["dead"] = dead
    MU, SD = rb.norm(tag); tr = rb.train_rows(ref); idx = rows_all(test)
    out = {}
    sfx = "" if dead is None else "_drop6s0"
    cache = MODELS / f"field_{tag}_{test_name}_2019_12{sfx}.npy"
    out["GNN"] = np.load(cache) if cache.exists() else None
    maps = rb.fit_ridge(ref, tr, MU, SD)
    A = rb.flat(ref, tr, np.arange(33), MU, SD, False); Y = ((ref["truth"][tr] - MU) / SD)[:, J]; L = ref["live"][tr][:, J]
    W = np.zeros((A.shape[1], len(J)))
    for grp in np.unique(rb.GROUP_OF):
        cols = np.flatnonzero(rb.GROUP_OF == grp); Ar = A[L[:, cols[0]]]
        W[:, cols] = np.linalg.solve(Ar.T @ Ar + 1e-3 * np.eye(A.shape[1]), Ar.T @ Y[L[:, cols[0]]][:, cols])
    out["Ridge"] = rb.ridge_expected(maps, test, idx, MU, SD, False, dead) - rb.flat(test, idx, np.arange(33), MU, SD, False, dead) @ W
    net = mlp_net(tag)
    out["MLP"] = rb.mlp_expected(net, test, idx, MU, SD, dead) - net(torch.tensor(rb.mlp_inputs(test, idx, MU, SD, dead=dead))).detach().numpy()
    # standardise with leak-free November-December of the training regime, as the detector does
    f_ref = {m: (np.load(MODELS / f"field_{tag}_{ref_name(train_regime)}_2018_12.npy") if m == "GNN" else None) for m in out}
    te = rb.rows(ref, "2018-11-01", "2019-01-01"); lr = ref["live"][te][:, J]
    stats = {}
    for m in out:
        if out[m] is None:
            continue
        if m == "GNN":
            fr = f_ref[m][te]
        elif m == "Ridge":
            fr = rb.ridge_expected(maps, ref, te, MU, SD) - rb.flat(ref, te, np.arange(33), MU, SD, False) @ W
        else:
            fr = rb.mlp_expected(net, ref, te, MU, SD) - net(torch.tensor(rb.mlp_inputs(ref, te, MU, SD))).detach().numpy()
        n = np.maximum(lr.sum(0), 1); mu = (fr * lr).sum(0) / n
        sd = np.maximum(np.sqrt((((fr - mu) ** 2) * lr).sum(0) / n), 1e-4)
        stats[m] = (mu, sd)
    z = {m: np.where(test["live"][:, J], np.clip((out[m] - stats[m][0]) / stats[m][1], -50, 50), 0.0) for m in stats}
    return z, test


def ref_name(regime):
    return rb.RUN[regime]


NDIST = None


def peaks(score, k, radius=300.0):
    """Nodes that are the maximum of their 300 m neighbourhood, the k strongest."""
    global NDIST
    if NDIST is None:
        NDIST = np.array([[scoring._DIST[a][b] for b in JN] for a in JN], np.float32)
    cand = [i for i in np.argsort(-score) if np.isfinite(score[i]) and score[i] >= score[NDIST[i] <= radius].max()]
    return cand[:k]


def localise(z, test, test_name):
    lab = pd.read_csv(OUT / test_name / "leak_flows.csv.gz", index_col=0, parse_dates=True).loc["2019"]
    res = {m: [] for m in z}
    for p in lab.columns:
        on = lab.index[lab[p].to_numpy() > 0]
        if not len(on):
            continue
        t0 = int(np.searchsorted(test["time"], np.datetime64(on[0])))
        t1 = min(int(np.searchsorted(test["time"], np.datetime64(on[-1]))) + 1, len(test["time"]))
        if WINDOW == "last_week":
            t0 = max(t0, t1 - WEEK)
        live = test["live"][t0:t1][:, J]
        k = int((lab.iloc[t0 * 12:t1 * 12].to_numpy() > 0).any(0).sum())      # leaks active in the window
        for m in z:
            n = np.maximum(live.sum(0), 1); score = (z[m][t0:t1] * live).sum(0) / n
            score[live.sum(0) == 0] = -np.inf
            est = PIPES[int(np.argmax(score))]
            top = [PIPES[i] for i in peaks(score, k)]
            res[m].append(dict(leak=p, estimate=est, distance_m=float(scoring.pipe_distance(est, p)), active=k,
                               nearest_peak_m=float(min(scoring.pipe_distance(e, p) for e in top))))
    return res


CASES = [("E1 continuous", f"cws_{SET}", "CWS", "continuous_CWS_leaks", None),
         ("E3 S48", f"s48_{SET}", "S48", "intermittent_S48_leaks", None),
         ("S24 unseen", f"s48_{SET}", "S48", "intermittent_S24_leaks", None),
         ("S72 unseen", f"s48_{SET}", "S48", "intermittent_S72_leaks", None),
         ("S48, 6 sensors failed", f"s48_{SET}", "S48", "intermittent_S48_leaks", 6),
         ("Continuous, 6 sensors failed", f"cws_{SET}", "CWS", "continuous_CWS_leaks", 6)]
summary, detail = {}, {}
for label, tag, regime, test_name, drop in CASES:
    dead = None
    if drop:
        dead = np.zeros(33, bool); dead[np.random.default_rng(0).choice(33, drop, replace=False)] = True
    z, test = fields(tag, regime, test_name, dead)
    res = localise(z, test, test_name); detail[label] = res
    summary[label] = {m: dict(n=len(v), median_m=float(np.median([r["distance_m"] for r in v])),
                              within_300=int(sum(r["distance_m"] <= 300 for r in v)),
                              within_600=int(sum(r["distance_m"] <= 600 for r in v)),
                              peak_median_m=float(np.median([r["nearest_peak_m"] for r in v])),
                              peak_within_300=int(sum(r["nearest_peak_m"] <= 300 for r in v))) for m, v in res.items()}
    print(label, json.dumps(summary[label]), flush=True)
(RES / f"localisation_{WINDOW}{'' if SET == 'hide' else '_' + SET}.json").write_text(json.dumps(dict(summary=summary, detail=detail), indent=1))
