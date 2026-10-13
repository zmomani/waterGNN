"""Step 12 - does the model of normal pressure survive a change of conditions?

Three tests, each scored by the error of the expected pressure at unmonitored,
pressurised nodes in November-December 2018 (leak-free), in metres:

  python 12_robustness.py sensors    some of the 33 sensors fail after training
  python 12_robustness.py schedule   the supply schedule differs from the training one
  python 12_robustness.py months     only 1 or 3 months of training data are available

The expected pressure is obtained as in detection: one group of 11 sensors is
hidden at a time and the three estimates are averaged. Results are written to
results/robustness_<test>.json.
"""
import json, sys
import numpy as np
import torch
import gnn_data as g
from gnn_model import Reconstructor
from common import *

MODELS, RES = ROOT / "models", ROOT / "results"
J = np.flatnonzero(g.IS_JUNCTION)
U = g.UNMONITORED[J]
GROUP_OF = np.where(g.ALWAYS[J], -1, g.SECTOR[J])
KEYS = ("time", "sensors", "level", "live", "since")
RUN = dict(CWS="continuous_CWS_noleak", S24="intermittent_S24_noleak", S48="intermittent_S48_noleak",
           S72="intermittent_S72_noleak")
_cache = {}


def load(regime):
    if regime not in _cache:
        _cache[regime] = g.load_run(RUN[regime], 2018, every=12)
    return _cache[regime]


def rows(run, a, b, step=1):
    t = run["time"]
    return np.arange(np.searchsorted(t, np.datetime64(a)), np.searchsorted(t, np.datetime64(b)))[::step]


def test_rows(run):
    return rows(run, "2018-11-01", "2019-01-01", 4)


def train_rows(run, months=0):
    return rows(run, "2018-01-29", "2018-11-01" if not months else f"2018-{2 + months:02d}-01")


def norm(tag):
    z = np.load(MODELS / f"norm_{tag}.npz"); return z["mu"], z["sd"]


def gnn_expected(tag, run, idx, aware=False, dead=None):
    MU, SD = norm(tag)
    model = Reconstructor(g.EDGE_INDEX, g.N, in_channels=13 if aware else 11)
    model.load_state_dict(torch.load(MODELS / f"reconstructor_{tag}.pt")); model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            sub = {k: run[k][idx[i:i + 64]] for k in KEYS}
            if dead is not None:
                sub["dead"] = dead
            x = torch.tensor(g.node_features(sub, MU, SD, aware))
            out.append((sum(model(g.hide(x, k)) for k in range(3)) / 3)[:, J].numpy())
    return np.concatenate(out)


def flat(run, idx, keep, MU, SD, aware, dead=None):
    ok = run["live"][idx][:, g.S_IDX]
    if dead is not None:
        ok = ok & ~dead
    z = np.where(ok, (run["sensors"][idx] - MU[g.S_IDX]) / SD[g.S_IDX], 0.0)
    cols = [z[:, keep], run["level"][idx] / 4.0, g.time_features(run["time"][idx]), np.ones(len(idx))]
    if aware:
        cols += [ok[:, keep].astype(float), run["state"][idx].astype(float)]
    return np.column_stack(cols)


def fit_ridge(run, idx, MU, SD, aware=False, drop=()):
    """One linear map per hidden group and per supply sector. drop: sensors left out of the fit."""
    Y = ((run["truth"][idx] - MU) / SD)[:, J]; L = run["live"][idx][:, J]
    maps = []
    for k in range(3):
        keep = np.setdiff1d(np.arange(33), np.concatenate([g.GROUPS[k], np.asarray(drop, int)]))
        A = flat(run, idx, keep, MU, SD, aware); W = np.zeros((A.shape[1], len(J)))
        for grp in np.unique(GROUP_OF):
            cols = np.flatnonzero(GROUP_OF == grp); Ar = A[L[:, cols[0]]]
            W[:, cols] = np.linalg.solve(Ar.T @ Ar + 1e-3 * np.eye(A.shape[1]), Ar.T @ Y[L[:, cols[0]]][:, cols])
        maps.append((keep, W))
    return maps


def ridge_expected(maps, run, idx, MU, SD, aware=False, dead=None):
    return sum(flat(run, idx, keep, MU, SD, aware, dead) @ W for keep, W in maps) / 3


def mlp_inputs(run, idx, MU, SD, hidden=None, dead=None):
    """Sensor readings, a flag per sensor, tank level and time: what the GNN sees, without the pipes."""
    ok = run["live"][idx][:, g.S_IDX]
    if dead is not None:
        ok = ok & ~dead
    if hidden is not None:
        ok = ok & ~hidden
    z = np.where(ok, (run["sensors"][idx] - MU[g.S_IDX]) / SD[g.S_IDX], 0.0)
    return np.column_stack([z, ok, run["level"][idx] / 4.0, g.time_features(run["time"][idx])]).astype(np.float32)


GROUP_MASK = np.zeros((3, 33), bool)
for _k in range(3):
    GROUP_MASK[_k, g.GROUPS[_k]] = True


def fit_mlp(run, idx, MU, SD, epochs=300, seed=0):
    """A multilayer perceptron of about the size of the GNN (three hidden layers of 128), trained like it:
    one sensor group hidden per sample, squared error at pressurised junctions."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = torch.nn.Sequential(torch.nn.Linear(71, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, len(J)))
    Y = torch.tensor(((run["truth"][idx] - MU) / SD)[:, J]); L = torch.tensor(run["live"][idx][:, J])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * (len(idx) // 64 + 1))
    for _ in range(epochs):
        X = torch.tensor(mlp_inputs(run, idx, MU, SD, hidden=GROUP_MASK[rng.integers(0, 3, len(idx))]))
        for b in torch.randperm(len(idx)).split(64):
            loss = ((net(X[b]) - Y[b])[L[b]] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


def mlp_expected(net, run, idx, MU, SD, dead=None):
    with torch.no_grad():
        return (sum(net(torch.tensor(mlp_inputs(run, idx, MU, SD, GROUP_MASK[k], dead))) for k in range(3)) / 3).numpy()


def mlp():
    out = dict(sensors={}, schedule={}, months={})
    for regime, tag in (("CWS", "cws_hide"), ("S48", "s48_hide")):
        run = load(regime); MU, SD = norm(tag); te = test_rows(run)
        net = fit_mlp(run, train_rows(run), MU, SD); res = {}
        for k in (0, 3, 6, 11, 16):
            v = []
            for draw in range(1 if k == 0 else 3):
                dead = np.zeros(33, bool); dead[np.random.default_rng(100 * k + draw).choice(33, k, replace=False)] = True
                v.append(mae(mlp_expected(net, run, te, MU, SD, dead), run, te, MU, SD))
            res[k] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))
        out["sensors"][regime] = res; print(regime, res, flush=True)
        if regime == "S48":
            for target in ("S24", "S48", "S72"):
                r = load(target); t2 = test_rows(r)
                out["schedule"][f"S48->{target}"] = mae(mlp_expected(net, r, t2, MU, SD), r, t2, MU, SD)
            out["months"][10] = out["schedule"]["S48->S48"]
            for m in (1, 3):
                if (MODELS / f"norm_s48_m{m}.npz").exists():
                    MUm, SDm = norm(f"s48_m{m}")
                    netm = fit_mlp(run, train_rows(run, m), MUm, SDm, epochs=300 * 10 // m // 2)
                    out["months"][m] = mae(mlp_expected(netm, run, te, MUm, SDm), run, te, MUm, SDm)
        save("mlp", out)


def mae(est, run, idx, MU, SD):
    err = np.abs(est - ((run["truth"][idx] - MU) / SD)[:, J]) * SD[J]
    return float(err[run["live"][idx][:, J] & U].mean())


def save(name, obj):
    (RES / f"robustness_{name}.json").write_text(json.dumps(obj, indent=1)); print(json.dumps(obj, indent=1))


def sensors():
    out = {}
    for regime, tag in (("CWS", "cws_hide"), ("S48", "s48_hide")):
        run = load(regime); MU, SD = norm(tag); te = test_rows(run); tr = train_rows(run)
        maps = fit_ridge(run, tr, MU, SD)
        res = {}
        for k in (0, 3, 6, 11, 16):
            acc = dict(GNN=[], Ridge=[], Ridge_refitted=[])
            for draw in range(1 if k == 0 else 3):
                failed = np.random.default_rng(100 * k + draw).choice(33, k, replace=False)
                dead = np.zeros(33, bool); dead[failed] = True
                acc["GNN"].append(mae(gnn_expected(tag, run, te, dead=dead), run, te, MU, SD))
                acc["Ridge"].append(mae(ridge_expected(maps, run, te, MU, SD, dead=dead), run, te, MU, SD))
                refit = fit_ridge(run, tr, MU, SD, drop=failed)
                acc["Ridge_refitted"].append(mae(ridge_expected(refit, run, te, MU, SD, dead=dead), run, te, MU, SD))
                print(regime, k, draw, {m: round(v[-1], 3) for m, v in acc.items()}, flush=True)
            res[k] = {m: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for m, v in acc.items()}
        out[regime] = res
        save("sensors", out)


def schedule():
    out = {}
    cases = [("S48", "s48_hide", False), ("S48", "s48_aware", True), ("S24", "s24_hide", False), ("S72", "s72_hide", False)]
    for trained_on, tag, aware in cases:
        if not (MODELS / f"reconstructor_{tag}.pt").exists():
            continue
        MU, SD = norm(tag); src = load(trained_on)
        maps = fit_ridge(src, train_rows(src), MU, SD, aware)
        for target in ("S24", "S48", "S72"):
            if trained_on != "S48" and target != trained_on:
                continue                                       # other schedules serve only as own-schedule reference
            run = load(target); te = test_rows(run)
            key = f"{trained_on}{'_aware' if aware else ''}->{target}"
            out[key] = dict(GNN=mae(gnn_expected(tag, run, te, aware), run, te, MU, SD),
                            Ridge=mae(ridge_expected(maps, run, te, MU, SD, aware), run, te, MU, SD))
            print(key, out[key], flush=True)
    save("schedule", out)


def months():
    out = {}
    for regime, tags in (("S48", {1: "s48_m1", 3: "s48_m3", 10: "s48_hide"}),):
        run = load(regime); te = test_rows(run); out[regime] = {}
        for m, tag in tags.items():
            if not (MODELS / f"reconstructor_{tag}.pt").exists():
                continue
            MU, SD = norm(tag)
            maps = fit_ridge(run, train_rows(run, 0 if m == 10 else m), MU, SD)
            out[regime][m] = dict(GNN=mae(gnn_expected(tag, run, te), run, te, MU, SD),
                                  Ridge=mae(ridge_expected(maps, run, te, MU, SD), run, te, MU, SD))
            print(regime, m, out[regime][m], flush=True)
    save("months", out)


if __name__ == "__main__":
    dict(sensors=sensors, schedule=schedule, months=months, mlp=mlp)[sys.argv[1]]()
