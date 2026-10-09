"""Step 1 - turn the L-Town model into a sectorised network ready for rationing.

Outputs (in data/):
  ltown_iws.inp   network with leak nodes inserted and demands neutralised
  nodes.csv       per-node attributes: sector, role, elevation, demand, tank size
  valves.csv      sector inlet pipes that act as rotation valves
  demand.npz      base demands and weekly patterns used by the simulator
"""
import numpy as np
import pandas as pd
import networkx as nx
import wntr
import yaml

from common import *

rng = np.random.default_rng(SEED)
wn = wntr.network.WaterNetworkModel(str(DATA / "L-TOWN.inp"))

# ---------------------------------------------------------------- leaks ------
cfg = yaml.safe_load(open(DATA / "battledim_configuration.yaml"))
leaks = []
for row in cfg["leakages"][1:]:
    pipe, t0, t1, dia, kind, tpeak = [s.strip() for s in row.split(",")]
    leaks.append(dict(pipe=pipe, start=t0, end=t1, diameter=float(dia),
                      kind=kind, peak=tpeak, node=f"leak_{pipe}"))
leaks = pd.DataFrame(leaks)
for _, lk in leaks.iterrows():            # a zero-demand node at each pipe midpoint
    wn = wntr.morph.split_pipe(wn, lk["pipe"], lk["pipe"] + "_B", lk["node"])
leaks.to_csv(DATA / "leaks.csv", index=False)

# ------------------------------------------------------------- demands -------
pat = {p: np.asarray(wn.get_pattern(p).multipliers) for p in wn.pattern_name_list}
juncs = wn.junction_name_list
base = {k: np.zeros(len(juncs)) for k in pat}
for i, n in enumerate(juncs):
    j = wn.get_node(n)
    for d in j.demand_timeseries_list:
        if d.pattern_name in base:
            base[d.pattern_name][i] += d.base_value      # m3/s
    j.demand_timeseries_list.clear()
    j.add_demand(0.0, None)                              # set step by step later
mean_demand = sum(base[k] * pat[k].mean() for k in pat)  # m3/s per node

# --------------------------------------------- transmission skeleton ---------
G = nx.Graph()
for name, l in wn.links():
    G.add_edge(l.start_node_name, l.end_node_name, name=name,
               length=getattr(l, "length", 1.0),
               trunk=(l.link_type != "Pipe") or l.diameter >= TRUNK_DIAMETER)
sources = wn.reservoir_name_list + wn.tank_name_list
for s in sources:                                        # source connections
    for u, v in G.edges(s):
        G[u][v]["trunk"] = True

T = nx.Graph([(u, v) for u, v, d in G.edges(data=True) if d["trunk"]])
# join isolated trunk pieces to the main one through the shortest pipe route
while nx.number_connected_components(T) > 1:
    comps = sorted(nx.connected_components(T), key=len, reverse=True)
    main, other = comps[0], comps[1]
    dist, path = nx.multi_source_dijkstra(G, other, weight="length")
    target = min((n for n in main), key=lambda n: dist[n])
    nx.add_path(T, path[target])
skeleton = set(T.nodes)

# --------------------------------------------------------- sectors -----------
H = G.subgraph([n for n in G if n not in skeleton])
comps = sorted(nx.connected_components(H), key=len, reverse=True)
idx = {n: i for i, n in enumerate(juncs)}
comp_dem = [sum(mean_demand[idx[n]] for n in c) for c in comps]

group_dem = np.zeros(N_SECTORS)
comp_group = {}
for ci in np.argsort(comp_dem)[::-1]:                    # greedy balancing by demand
    g = int(np.argmin(group_dem))
    comp_group[ci] = g
    group_dem[g] += comp_dem[ci]

sector = {}
for ci, c in enumerate(comps):
    for n in c:
        sector[n] = comp_group[ci]
role = {n: "distribution" for n in sector}
# service connections on the mains follow the schedule of the nearest sector
interior = set(sector)
dist, path = nx.multi_source_dijkstra(G, interior, weight="length")
for n in skeleton:
    if n in idx:
        sector[n] = sector[path[n][0]]
        role[n] = "transmission"

valves = [dict(pipe=d["name"], sector=sector[u if u in interior else v])
          for u, v, d in G.edges(data=True)
          if (u in interior) != (v in interior)]
pd.DataFrame(valves).to_csv(DATA / "valves.csv", index=False)

# ------------------------------------------------- household storage ---------
tank_days = rng.uniform(TANK_DAYS_MIN, TANK_DAYS_MAX, len(juncs))
tank_vol = tank_days * mean_demand * 86400               # m3 per node

nodes = pd.DataFrame({
    "node": juncs,
    "x": [wn.get_node(n).coordinates[0] for n in juncs],
    "y": [wn.get_node(n).coordinates[1] for n in juncs],
    "elevation": [wn.get_node(n).elevation for n in juncs],
    "sector": [sector[n] for n in juncs],
    "role": [role[n] for n in juncs],
    "mean_demand_m3d": mean_demand * 86400,
    "tank_m3": tank_vol,
})
nodes.to_csv(DATA / "nodes.csv", index=False)
np.savez(DATA / "demand.npz", nodes=np.array(juncs), **{f"base_{k}": v for k, v in base.items()},
         **{f"pat_{k}": v for k, v in pat.items()})

wn.options.time.hydraulic_timestep = DT
wn.options.time.report_timestep = DT
wn.options.time.pattern_timestep = DT
wntr.network.write_inpfile(wn, str(DATA / "ltown_iws.inp"), version=2.2)

print(f"junctions {len(juncs)}, skeleton nodes {len(skeleton)}, sub-zones {len(comps)}, "
      f"rotation valves {len(valves)}")
print(nodes.groupby("sector").agg(nodes=("node", "size"), demand_m3d=("mean_demand_m3d", "sum"),
                                  storage_m3=("tank_m3", "sum"), elev_min=("elevation", "min"),
                                  elev_max=("elevation", "max")).round(1))
print(nodes.role.value_counts())
