"""Step 2 - simulate L-Town under continuous or rationed (intermittent) supply.

  python 02_simulate.py --regime continuous   --weeks 12 --leaks
  python 02_simulate.py --regime intermittent --weeks 12 --leaks --schedule S48

EPANET 2.2 is advanced one 5-minute step at a time (through EPyT). At every step
the script (i) opens or closes the rotation valves, (ii) updates each node's
requested inflow from the state of its household tank, (iii) updates leak
emitters, (iv) solves the pressure-dependent hydraulics and (v) updates tanks.
"""
import argparse, json, time, zlib
import numpy as np
import pandas as pd
import yaml
from epyt import epanet

from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--regime", choices=["continuous", "intermittent"], required=True)
ap.add_argument("--schedule", default="S48", choices=list(SCHEDULES))
ap.add_argument("--weeks", type=int, default=12)
ap.add_argument("--days", type=int, default=None, help="run length in days (overrides --weeks)")
ap.add_argument("--events", action="store_true", help="add authorised consumption events")
ap.add_argument("--leaks", action="store_true")
ap.add_argument("--only-leak", default=None, help="simulate a single leak, e.g. p673")
ap.add_argument("--name", default=None)
args = ap.parse_args()
name = args.name or f"{args.regime}_{args.schedule if args.regime == 'intermittent' else 'CWS'}" \
                    f"_{'leaks' if args.leaks else 'noleak'}"
out = OUT / name
out.mkdir(parents=True, exist_ok=True)
IWS = args.regime == "intermittent"
n_steps = args.days * 86400 // DT if args.days else args.weeks * STEPS_PER_WEEK
n_weeks = -(-n_steps // STEPS_PER_WEEK)
t_index = pd.date_range(START, periods=n_steps, freq=f"{DT}s")

# ------------------------------------------------------------ inputs ---------
nodes = pd.read_csv(DATA / "nodes.csv").set_index("node")
valves = pd.read_csv(DATA / "valves.csv")
leaks = pd.read_csv(DATA / "leaks.csv", parse_dates=["start", "end", "peak"])
dem = np.load(DATA / "demand.npz", allow_pickle=True)
cfg = yaml.safe_load(open(DATA / "battledim_configuration.yaml"))

d = epanet(str(DATA / "ltown_iws.inp"))
d.setTimeSimulationDuration(n_steps * DT)
d.setTimeHydraulicStep(DT)
d.setDemandModel("PDA", P_MIN, P_REQ, P_EXP)
api = d.api
EN_BASEDEMAND, EN_EMITTER, EN_STATUS = 1, 3, 11

all_nodes = d.getNodeNameID()
all_links = d.getLinkNameID()
nidx = {n: i + 1 for i, n in enumerate(all_nodes)}
lidx = {l: i + 1 for i, l in enumerate(all_links)}
juncs = list(dem["nodes"])
jpos = np.array([nidx[n] - 1 for n in juncs])            # positions in EPANET arrays
nodes = nodes.loc[juncs]
sector = nodes.sector.to_numpy()
is_dist = (nodes.role == "distribution").to_numpy()
tank_max = nodes.tank_m3.to_numpy()
q_fill = tank_max / (TANK_FILL_HOURS * 3600)             # m3/s at full pressure

# --------------------------------------------------------- schedule ----------
def sector_state(hours_in_week, hours):
    """True where a sector is in supply. One window per week up to 48 h,
    two equal windows 84 h apart above that; sector starts are 28 h apart."""
    on = np.zeros(N_SECTORS, bool)
    for k, h in enumerate(hours):
        starts = [k * 28.0] if h <= 48 else [k * 28.0, k * 28.0 + 84.0]
        length = h / len(starts)
        on[k] = any(((hours_in_week - s) % 168.0) < length for s in starts)
    return on

hours = SCHEDULES[args.schedule]

# ---------------------------------------------------------- demand -----------
rng = np.random.default_rng(SEED)                        # same draws in every run
node_bias = rng.lognormal(0, 0.10, len(juncs))           # persistent household spread

def consumption(k):
    """Household consumption [m3/s] at step k: L-Town weekly patterns, a mild
    seasonal cycle and step-to-step noise."""
    w = k % STEPS_PER_WEEK
    c = (dem["base_P-Residential"] * dem["pat_P-Residential"][w]
         + dem["base_P-Commercial"] * dem["pat_P-Commercial"][w]
         + dem["base_P-Industrial"] * dem["pat_P-Industrial"][0])
    season = 1 + 0.10 * np.sin(2 * np.pi * (k * DT / 86400 - 105) / 365)
    noise = np.random.default_rng(SEED + 1 + k).lognormal(0, 0.05, len(juncs))
    return c * season * node_bias * noise

# ----------------------------------------------------------- leaks -----------
leak_pos = np.array([juncs.index(n) for n in leaks["node"]])
leak_area = np.pi / 4 * leaks.diameter.to_numpy() ** 2
t0 = (leaks.start - pd.Timestamp(START)).dt.total_seconds().to_numpy()
tp = (leaks.peak - pd.Timestamp(START)).dt.total_seconds().to_numpy()
t1 = (leaks.end - pd.Timestamp(START)).dt.total_seconds().to_numpy()

leak_used = np.full(len(leaks), args.leaks) & ((leaks["pipe"] == args.only_leak) | (args.only_leak is None)).to_numpy()

def leak_coeff(t):
    """Emitter coefficients [m3/s per m^0.5]; incipient leaks grow linearly."""
    ramp = np.where(tp > t0, np.clip((t - t0) / np.maximum(tp - t0, 1), 0, 1), 1.0)
    active = (t >= t0) & (t < t1)
    return LEAK_CD * leak_area * np.sqrt(2 * 9.81) * ramp * active * leak_used

# ------------------------------------------- authorised consumption ----------
# Unmetered but legitimate withdrawals (hydrant use, flushing): the false-alarm
# class of the classifier. The same events are used in every run that has them.
ev_rng = np.random.default_rng(SEED + 77)
cand = np.flatnonzero((nodes.mean_demand_m3d.to_numpy() > 0))
n_events = int(round(EVENTS_PER_YEAR * n_steps * DT / (365 * 86400)))
events = pd.DataFrame({
    "node": [juncs[i] for i in ev_rng.choice(cand, n_events)],
    "start_step": np.sort(ev_rng.integers(STEPS_PER_WEEK, n_steps - 80, n_events)),
    "duration_steps": ev_rng.integers(EVENT_HOURS[0] * 12, EVENT_HOURS[1] * 12 + 1, n_events),
    "flow_m3h": ev_rng.uniform(*EVENT_FLOW_M3H, n_events).round(2),
})
if IWS:   # a withdrawal needs water: redraw the start until the node's sector is in supply
    for e in range(n_events):
        i, k0 = juncs.index(events.node[e]), int(events.start_step[e])
        while is_dist[i] and not sector_state(k0 * DT / 3600 % 168, hours)[sector[i]]:
            k0 = int(ev_rng.integers(STEPS_PER_WEEK, n_steps - 80))
        events.loc[e, "start_step"] = k0
    events = events.sort_values("start_step").reset_index(drop=True)
events["start"] = t_index[events.start_step]
events["end"] = t_index[np.minimum(events.start_step + events.duration_steps, n_steps - 1)]
ev_pos = np.array([juncs.index(n) for n in events["node"]], dtype=int)
ev_a, ev_b = events.start_step.to_numpy(), (events.start_step + events.duration_steps).to_numpy()
ev_q = events.flow_m3h.to_numpy() / 3600 * args.events

# ---------------------------------------------------------- outputs ----------
P = np.zeros((n_steps, len(juncs)), np.float32)          # nodal pressure [m]
Q = np.zeros((n_steps, len(juncs)), np.float32)          # delivered inflow [m3/h]
TANK = np.zeros((n_steps, len(juncs)), np.float16)       # household tank fill [0-1]
UNMET = np.zeros((n_steps, N_SECTORS), np.float32)       # unmet consumption [m3/h]
STATE = np.zeros((n_steps, N_SECTORS), bool)
LEAKQ = np.zeros((n_steps, len(leaks)), np.float32)      # leak outflow [m3/h]
EVQ = np.zeros((n_steps, max(n_events, 1)), np.float32)   # authorised withdrawal [m3/h]
flow_ids = cfg["flow_sensors"]
FLOW = np.zeros((n_steps, len(flow_ids)), np.float32)
LEVEL = np.zeros(n_steps, np.float32)
fpos = [lidx[l] - 1 for l in flow_ids]
tpos = nidx["T1"] - 1

V = TANK_INIT_FRACTION * tank_max.copy()
prev_state = None
d.openHydraulicAnalysis()
d.initializeHydraulicAnalysis()
tic = time.time()
for k in range(n_steps):
    t = k * DT
    state = sector_state(t / 3600 % 168, hours) if IWS else np.ones(N_SECTORS, bool)
    if prev_state is None or (state != prev_state).any():
        for pipe, sec in zip(valves["pipe"], valves["sector"]):
            api.ENsetlinkvalue(lidx[pipe], EN_STATUS, float(state[sec]))
        prev_state = state.copy()
    on = state[sector]                                    # service valve open
    live = on | ~is_dist                                  # pipe at the node is pressurised

    c = consumption(k)
    if IWS:   # float valve: refill at the tank's rate until full, then follow use
        req = np.where(on, np.minimum(q_fill, (tank_max - V) / DT + c), 0.0)
    else:
        req = c
    extra = np.zeros(len(juncs))                          # authorised withdrawals, straight from the pipe
    act = np.flatnonzero((k >= ev_a) & (k < ev_b)) if args.events else []
    for e in act:
        extra[ev_pos[e]] += ev_q[e] * live[ev_pos[e]]
    req = req + extra
    for i, q in zip(jpos, req):
        api.ENsetnodevalue(int(i) + 1, EN_BASEDEMAND, float(q) * 3600)   # CMH
    coeff = leak_coeff(t) * live[leak_pos]
    for i, cf in zip(jpos[leak_pos], coeff):
        api.ENsetnodevalue(int(i) + 1, EN_EMITTER, float(cf) * 3600)

    d.runHydraulicAnalysis()
    p = np.asarray(d.getNodePressure())[jpos]
    # isolated pipes are treated as depressurised; EPANET's negative pressures
    # (high zone when tank T1 runs empty) are floored at zero for the same reason
    p = np.clip(np.where(live, p, 0.0), 0.0, None)
    LEAKQ[k] = coeff * np.sqrt(np.clip(p[leak_pos], 0, None)) * 3600
    wag = np.clip((p - P_MIN) / (P_REQ - P_MIN), 0, 1) ** P_EXP           # Wagner curve
    delivered = req * wag
    for e in act:
        EVQ[k, e] = extra[ev_pos[e]] * wag[ev_pos[e]] * 3600
    household = (req - extra) * wag                       # inflow that reaches household tanks

    if IWS:
        V = V + (household - c) * DT
        short = np.clip(-V, 0, None) / DT
        V = np.clip(V, 0, tank_max)
        TANK[k] = np.divide(V, tank_max, out=np.zeros_like(V), where=tank_max > 0)
    else:
        short = c - household
    UNMET[k] = np.bincount(sector, short, N_SECTORS) * 3600
    P[k], Q[k], STATE[k] = p, delivered * 3600, state
    flows = np.asarray(d.getLinkFlows())
    FLOW[k] = flows[fpos]
    LEVEL[k] = np.asarray(d.getNodePressure())[tpos]
    d.nextHydraulicAnalysisStep()
    if k % STEPS_PER_WEEK == 0:
        print(f"week {k // STEPS_PER_WEEK + 1}/{n_weeks}  {time.time() - tic:.0f}s", flush=True)
d.closeHydraulicAnalysis()
d.unload()

# ------------------------------------------------------------ save -----------
# SCADA table with the same sensor set as BattLeDIM. Sensor noise is added here,
# so full_state.npz keeps the exact values and scada.csv.gz the measured ones.
nz = np.random.default_rng(SEED + zlib.crc32(name.encode()))
def noisy_p(x):      # pressure and level sensors: additive noise, none on a dry pipe
    return np.where(x > 0, np.clip(x + nz.normal(0, NOISE_PRESSURE_M, x.shape), 0, None), 0.0)
def noisy_q(x):      # flow meters: proportional noise
    return x * (1 + nz.normal(0, NOISE_FLOW_REL, x.shape))
jcol = {n: i for i, n in enumerate(juncs)}
cols = {f"P_{n}": noisy_p(P[:, jcol[n]]) for n in cfg["pressure_sensors"]}
cols |= {f"F_{l}": noisy_q(FLOW[:, i]) for i, l in enumerate(flow_ids)}
cols["L_T1"] = noisy_p(LEVEL)
cols |= {f"D_{n}": noisy_q(Q[:, jcol[n]]) for n in cfg["amrs"]}
cols |= {f"supply_sector{s}": STATE[:, s].astype(int) for s in range(N_SECTORS)}
scada = pd.DataFrame(cols, index=t_index)
scada.index.name = "Timestamp"
scada.round(3).to_csv(out / "scada.csv.gz")
if args.events:
    events.assign(delivered_m3=(EVQ[:, :n_events].sum(0) / 12).round(2)).to_csv(out / "events.csv", index=False)

labels = pd.DataFrame(LEAKQ, index=t_index, columns=leaks["pipe"]).round(3)
labels.to_csv(out / "leak_flows.csv.gz")
np.savez_compressed(out / "full_state.npz", pressure=P, delivered=Q, tank=TANK, unmet=UNMET,
                    state=STATE, leak_flow=LEAKQ, event_flow=EVQ, flow=FLOW, level=LEVEL, nodes=np.array(juncs))
meta = dict(vars(args), name=name, steps=n_steps, start=START, dt=DT,
            supply_hours=hours if IWS else [168] * N_SECTORS, runtime_s=round(time.time() - tic))
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print("saved", out)
