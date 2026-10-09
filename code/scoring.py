"""BattLeDIM scoring (Vrachimis et al., 2022, Eqs. 5-9), reimplemented from the
published Scoring_Algorithm.m. A detection is a pipe and a time.

  score_detections(detections, leak_flows, start, end)  ->  dict
"""
import networkx as nx
import numpy as np
import pandas as pd
import wntr
from common import DATA, DT

X_MAX, COST_WATER, COST_CREW = 300.0, 0.80, 500.0     # m, EUR/m3, EUR  (Table 3)

_wn = wntr.network.WaterNetworkModel(str(DATA / "L-TOWN.inp"))
_G = nx.Graph()
for _name, _l in _wn.links():
    _G.add_edge(_l.start_node_name, _l.end_node_name, length=getattr(_l, "length", 0.0))
_DIST = dict(nx.all_pairs_dijkstra_path_length(_G, weight="length"))


def pipe_distance(a, b):
    """Distance along the pipes between the centres of links a and b [m]."""
    if a == b:
        return 0.0
    la, lb = _wn.get_link(a), _wn.get_link(b)
    ends = [(la.start_node_name, lb.start_node_name), (la.start_node_name, lb.end_node_name),
            (la.end_node_name, lb.start_node_name), (la.end_node_name, lb.end_node_name)]
    return min(_DIST[u][v] for u, v in ends) + getattr(la, "length", 0) / 2 + getattr(lb, "length", 0) / 2


def score_detections(detections, leak_flows, start, end):
    """detections: iterable of (pipe, timestamp). leak_flows: DataFrame of leak
    outflow [m3/h], one column per leaking pipe. Only [start, end] is scored."""
    lf = leak_flows.loc[start:end]
    active = {p: lf.index[lf[p].to_numpy() > 0] for p in lf.columns}
    leaks = {p: (t[0], t[-1]) for p, t in active.items() if len(t)}
    found, rows, total = set(), [], 0.0
    for pipe, t in sorted(((p, pd.Timestamp(t)) for p, t in detections), key=lambda d: d[1]):
        if not (pd.Timestamp(start) <= t <= pd.Timestamp(end)):
            continue
        cand = [(pipe_distance(pipe, lp), lp) for lp, (t0, t1) in leaks.items() if t0 <= t <= t1]
        cand = [c for c in cand if c[0] <= X_MAX]
        if not cand:
            rows.append(dict(pipe=pipe, time=t, leak=None, distance=np.nan, score=-COST_CREW, kind="FP"))
            total -= COST_CREW
            continue
        x, lp = min(cand)
        if lp in found:                                   # repeat detection: ignored
            rows.append(dict(pipe=pipe, time=t, leak=lp, distance=x, score=0.0, kind="repeat"))
            continue
        found.add(lp)
        saved = float(lf.loc[t:, lp].sum()) * DT / 3600   # m3 from detection to repair
        s = saved * COST_WATER - COST_CREW * x / X_MAX
        rows.append(dict(pipe=pipe, time=t, leak=lp, distance=x, score=s, kind="TP"))
        total += s
    perfect = sum(float(lf[p].sum()) * DT / 3600 * COST_WATER for p in leaks)
    tp = len(found); fp = sum(r["kind"] == "FP" for r in rows)
    return dict(score=total, max_score=perfect, relative=total / perfect if perfect else np.nan,
                TP=tp, FP=fp, FN=len(leaks) - tp, leaks=len(leaks),
                TPR=tp / len(leaks) if leaks else np.nan, detail=pd.DataFrame(rows))
