"""Step 18 - classification of anomalies from the residual of the final detector.

Each anomaly (a leak or an authorised consumption event, given its location and start
time) is classified as abrupt burst, incipient leak or authorised event from the way the
normal residual of the tuned GNN develops around it. The classifier is trained on the
2018 events and tested on the 2019 events; the location and start time are those of the
simulation, so this measures classification given detection, not the two together.

  python 18_classify.py            # S48 (tuned GNN) and continuous (tuned GNN)
"""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import gnn_data as g
import scoring
from common import *

MODELS, RES = ROOT / "models", ROOT / "results"
J = np.flatnonzero(g.IS_JUNCTION); JN = [g.NODES[i] for i in J]
JPOS = {n: i for i, n in enumerate(JN)}
CLASSES = ["abrupt", "incipient", "event"]


def how(t):
    return (t.dayofweek * 24 + t.hour).to_numpy()


def z_fields(tag, run_prefix):
    """Standardised normal residual at junctions for the leak runs of 2018 and 2019."""
    MU, SD = (lambda z: (z["mu"], z["sd"]))(np.load(MODELS / f"norm_{tag}.npz"))
    ref = g.load_run(f"{run_prefix}_noleak", 2018, every=12)
    rec = np.load(MODELS / f"recon_{tag}_{run_prefix}_noleak_2018_12.npy")
    warm = int(np.searchsorted(ref["time"], np.datetime64("2018-01-29")))
    split = int(np.searchsorted(ref["time"], np.datetime64("2018-11-01")))
    L = ref["live"][:, J]; h = how(ref["time"])
    N = np.zeros((168, len(J)), np.float32)
    for k in range(168):
        m = (h == k) & (np.arange(len(h)) >= warm) & (np.arange(len(h)) < split)
        N[k] = (rec[m] * L[m]).sum(0) / np.maximum(L[m].sum(0), 1)
    f_ref = (N[h] - rec)[split:]; l_ref = L[split:]
    n = np.maximum(l_ref.sum(0), 1); mu = (f_ref * l_ref).sum(0) / n
    sd = np.maximum(np.sqrt((((f_ref - mu) ** 2) * l_ref).sum(0) / n), 1e-4)
    out = {}
    for year in (2018, 2019):
        run = g.load_run(f"{run_prefix}_leaks", year, every=12)
        r = np.load(MODELS / f"recon_{tag}_{run_prefix}_leaks_{year}_12.npy")
        z = np.where(run["live"][:, J], np.clip((N[how(run["time"])] - r - mu) / sd, -50, 50), 0.0)
        out[year] = dict(z=z.astype(np.float32), live=run["live"][:, J], time=run["time"])
    return out


_near = {}


def near(node, radius=300.0):
    if node not in _near:
        _near[node] = np.array([i for i, n in enumerate(JN) if scoring._DIST[node].get(n, np.inf) <= radius])
    return _near[node]


def features(fld, node, start, end):
    """Residual around the anomaly, in supply hours, and the features derived from it."""
    t = fld["time"]
    t0 = int(np.searchsorted(t, np.datetime64(start))); t1 = min(int(np.searchsorted(t, np.datetime64(end))) + 1, len(t))
    t1 = min(t1, t0 + 14 * 24)                                   # at most two weeks after onset
    idx = near(node)
    live = fld["live"][t0:t1][:, idx]
    s = np.where(live, fld["z"][t0:t1][:, idx], np.nan)
    s = np.nanmean(s, axis=1)                                     # mean over the neighbourhood, NaN when dry
    sup = s[~np.isnan(s)]                                         # the series in supply hours only
    if len(sup) < 2:
        sup = np.array([0.0, 0.0])
    first = sup[:6].mean(); later = sup[6:48].mean() if len(sup) > 6 else first
    week2 = sup[48:].mean() if len(sup) > 48 else later
    days = pd.Series(s, index=t[t0:t1]).resample("D").mean()
    clock_hours_to_half = float(np.argmax(np.nan_to_num(s) >= 0.5 * np.nanmax(s))) if np.nanmax(s) > 0 else 0.0
    return dict(onset=first, rise=later - first, late=week2, peak=float(np.nanmax(s)),
                persist=float(np.mean(sup[6:] > 0.5)) if len(sup) > 6 else 0.0,
                n_supply_hours=float(len(sup)), hours_to_half=clock_hours_to_half,
                day_var=float(np.nanstd(days.to_numpy())) if len(days) > 1 else 0.0)


def anomalies(run_prefix, year):
    leaks = pd.read_csv(DATA / "leaks.csv")
    wn = scoring._wn
    rows = []
    for _, lk in leaks.iterrows():
        s, e = pd.Timestamp(lk["start"]), pd.Timestamp(lk["end"])
        if s.year != year:
            continue
        link = wn.get_link(lk["pipe"]); node = link.start_node_name
        rows.append(dict(id=lk["pipe"], node=node, start=s, end=e, label=lk["kind"]))
    ev = pd.read_csv(OUT / f"{run_prefix}_leaks" / "events.csv", parse_dates=["start", "end"])
    for _, r in ev.iterrows():
        if r["start"].year == year:
            rows.append(dict(id=f"event_{r['node']}_{r['start']:%m%d}", node=r["node"], start=r["start"],
                             end=r["end"] + pd.Timedelta(hours=6), label="event"))
    return rows


def run(tag, run_prefix, name):
    fld = z_fields(tag, run_prefix)
    sets = {}
    for year in (2018, 2019):
        rows = anomalies(run_prefix, year)
        X = pd.DataFrame([features(fld[year], r["node"], r["start"], r["end"]) for r in rows])
        y = np.array([r["label"] for r in rows])
        sets[year] = (X, y, rows)
    Xtr, ytr, _ = sets[2018]; Xte, yte, rows_te = sets[2019]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr.fillna(0), ytr)
    pred = clf.predict(Xte.fillna(0))
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    acc = float((pred == yte).mean())
    # the two leak classes only
    m = yte != "event"
    cm_leaks = confusion_matrix(yte[m], pred[m], labels=CLASSES[:2])
    detail = [dict(id=r["id"], true=t, predicted=p, **{k: round(float(v), 3) for k, v in Xte.iloc[i].items()})
              for i, (r, t, p) in enumerate(zip(rows_te, yte, pred))]
    out = dict(train_n=dict(zip(*np.unique(ytr, return_counts=True))), test_n=dict(zip(*np.unique(yte, return_counts=True))),
               accuracy=acc, confusion=dict(labels=CLASSES, matrix=cm.tolist()),
               confusion_leaks_only=dict(labels=CLASSES[:2], matrix=cm_leaks.tolist()),
               coefficients={c: dict(zip(Xtr.columns, np.round(clf[-1].coef_[i], 2))) for i, c in enumerate(clf[-1].classes_)},
               detail=detail)
    out["train_n"] = {k: int(v) for k, v in out["train_n"].items()}; out["test_n"] = {k: int(v) for k, v in out["test_n"].items()}
    (RES / f"classification_{name}.json").write_text(json.dumps(out, indent=1))
    print(name, "accuracy", round(acc, 3), "n test", out["test_n"])
    print(" confusion (rows true, cols predicted)", CLASSES); print(cm)
    print(" leaks only", CLASSES[:2]); print(cm_leaks)
    return out


if __name__ == "__main__":
    run("s48_tuned", "intermittent_S48", "S48_tuned")
    run("cws_tuned", "continuous_CWS", "CWS_tuned")
