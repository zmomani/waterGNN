"""Graph, features and data loading shared by the models."""
import numpy as np
import pandas as pd
import torch
import wntr
import yaml
from common import *

P_SCALE = 50.0                                   # pressures are divided by this before entering a model

_wn = wntr.network.WaterNetworkModel(str(DATA / "L-TOWN.inp"))
_cfg = yaml.safe_load(open(DATA / "battledim_configuration.yaml"))
NODES = _wn.node_name_list                       # 782 junctions, 2 reservoirs, 1 tank = 785
N = len(NODES)
IDX = {n: i for i, n in enumerate(NODES)}
SENSORS = _cfg["pressure_sensors"]
S_IDX = np.array([IDX[s] for s in SENSORS])
IS_JUNCTION = np.array([n in set(_wn.junction_name_list) for n in NODES])
UNMONITORED = IS_JUNCTION.copy(); UNMONITORED[S_IDX] = False
XY = np.array([_wn.get_node(n).coordinates for n in NODES])
LINKS = [(l.start_node_name, l.end_node_name) for _, l in _wn.links()]

_e = np.array([[IDX[a], IDX[b]] for a, b in LINKS]).T
EDGE_INDEX = torch.tensor(np.concatenate([_e, _e[::-1]], axis=1), dtype=torch.long)   # both directions

_elev = np.array([getattr(_wn.get_node(n), "elevation", getattr(_wn.get_node(n), "base_head", 0.0)) for n in NODES])
_type = np.zeros((N, 3)); _type[IS_JUNCTION, 0] = 1
for r in _wn.reservoir_name_list: _type[IDX[r], 1] = 1
for t in _wn.tank_name_list: _type[IDX[t], 2] = 1
_flag = np.zeros(N); _flag[S_IDX] = 1
STATIC = np.column_stack([_elev / 100.0, _type, _flag]).astype(np.float32)             # 5 features
TANK = IDX["T1"]


_nd = pd.read_csv(DATA / "nodes.csv").set_index("node")
SECTOR = np.array([int(_nd.sector[n]) if n in _nd.index else 0 for n in NODES])
ALWAYS = np.array([(n not in _nd.index) or (_nd.role[n] == "transmission") for n in NODES])   # mains, sources


def load_run(name, year, every=1):
    """Sensor readings (with noise), exact pressure at every node, and timestamps."""
    sc = pd.read_csv(OUT / name / "scada.csv.gz", index_col=0, parse_dates=True).loc[str(year)].iloc[::every]
    z = np.load(OUT / name / "full_state.npz")
    k = ((sc.index - pd.Timestamp(START)) / pd.Timedelta(seconds=DT)).astype(int).to_numpy()
    names = list(z["nodes"]); col = [names.index(n) for n in NODES if n in names]
    truth = np.zeros((len(k), N), np.float32)
    truth[:, [IDX[n] for n in NODES if n in names]] = z["pressure"][k][:, col]
    full = z["state"]                                    # [steps, sectors], every 5 minutes
    run_len = np.zeros(full.shape, np.float32)           # hours since the sector's supply window opened
    for s in range(full.shape[1]):
        on = full[:, s]
        idx = np.arange(len(on)); last_off = np.maximum.accumulate(np.where(~on, idx, -1))
        run_len[:, s] = np.where(on, (idx - last_off) * DT / 3600.0, 0.0)
    state = full[k]
    live = state[:, SECTOR] | ALWAYS                     # [T, N]: the pipe at the node is pressurised
    since = np.where(ALWAYS, 0.0, run_len[k][:, SECTOR]).astype(np.float32)
    return dict(live=live, since=since, time=sc.index, sensors=sc[[f"P_{s}" for s in SENSORS]].to_numpy(np.float32),
                level=sc["L_T1"].to_numpy(np.float32), truth=truth, state=state,
                flows=sc[[c for c in sc.columns if c.startswith("F_")]].to_numpy(np.float32))


def time_features(index):
    h = (index.hour + index.minute / 60) / 24 * 2 * np.pi
    d = index.dayofweek / 7 * 2 * np.pi
    return np.column_stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d)]).astype(np.float32)


def node_features(run, mu=None, sd=None, aware=False):
    """[T, N, 11 or 13]: 5 static, sensor pressure, tank level, 4 time features, and
    for the supply-aware model the supply state and the hours since supply began.
    A sensor on an unpressurised pipe is treated as missing: reading and flag are zero."""
    T = len(run["time"])
    x = np.zeros((T, N, 13 if aware else 11), np.float32)
    x[:, :, :5] = STATIC
    ok = run["live"][:, S_IDX] if "live" in run else np.ones((T, len(S_IDX)), bool)
    if mu is None:
        x[:, S_IDX, 5] = run["sensors"] / P_SCALE
    else:
        x[:, S_IDX, 5] = np.where(ok, (run["sensors"] - mu[S_IDX]) / sd[S_IDX], 0.0)
    x[:, S_IDX, 4] = ok
    x[:, TANK, 6] = run["level"] / 4.0
    x[:, :, 7:11] = time_features(run["time"])[:, None, :]
    if aware:
        x[:, :, 11] = run["live"]
        x[:, :, 12] = np.clip(run["since"] / 24.0, 0, 3)
    return x


# Three fixed groups of 11 sensors. To obtain the expected reading of a sensor, its
# group is hidden and the model estimates it from the other 22 sensors. Groups are
# interleaved along the west-east axis so that every hidden sensor has neighbours left.
_order = np.argsort(XY[S_IDX, 0])
GROUPS = [np.sort(_order[i::3]) for i in range(3)]          # positions within SENSORS


def hide(x, group):
    """Remove the readings and sensor flags of one group from a feature array."""
    x = x.clone() if hasattr(x, "clone") else x.copy()
    x[..., S_IDX[GROUPS[group]], 5] = 0.0
    x[..., S_IDX[GROUPS[group]], 4] = 0.0
    return x


def batched_edges(batch):
    return torch.cat([EDGE_INDEX + i * N for i in range(batch)], dim=1)
