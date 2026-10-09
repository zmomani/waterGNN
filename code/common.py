"""Shared parameters for the intermittent L-Town dataset.

Every assumption that imposes intermittent (rationed) supply on L-Town is
declared here, so that the paper's parameter table can be checked line by line.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIGS = ROOT / "figs"

SEED = 2026
DT = 300                       # hydraulic and reporting step [s], as in BattLeDIM
STEPS_PER_WEEK = 7 * 24 * 3600 // DT

# --- transmission / distribution split -------------------------------------
TRUNK_DIAMETER = 0.150         # pipes >= 150 mm are treated as transmission mains
N_SECTORS = 6                  # rotation groups ("supply sectors")

# --- rationing schedule -----------------------------------------------------
# Supply hours per week for each sector. Documented Amman figures: 12-72 h/week
# (Rosenberg et al., 2007) and a city-wide mean of about 48 h/week
# (Zozmann et al., 2019). The six values below have mean 48 h.
SUPPLY_HOURS = [24, 36, 48, 48, 60, 72]
# Alternative schedules for the sensitivity analysis (hours scaled per sector)
SCHEDULES = {
    "S48": [24, 36, 48, 48, 60, 72],     # baseline, mean 48 h/week
    "S24": [12, 18, 24, 24, 30, 36],     # severe rationing, mean 24 h/week
    "S72": [36, 54, 72, 72, 90, 108],    # relaxed rationing, mean 72 h/week
}

# --- household storage --------------------------------------------------------
# Roof tanks of 1-2 m3 per household (Rosenberg et al., 2007) and billed use of
# 39.6 m3 per connection per quarter (0.44 m3/day) give 2.3-4.5 days of storage.
TANK_DAYS_MIN, TANK_DAYS_MAX = 2.3, 4.5
TANK_FILL_HOURS = 8.0          # time to fill an empty tank at full pressure
TANK_INIT_FRACTION = 0.5       # initial fill level

# --- pressure-dependent delivery (EPANET 2.2 PDA, Wagner curve) ---------------
P_MIN = 6.0                    # m: below this no water reaches a roof tank
P_REQ = 15.0                   # m: at or above this the float valve passes full flow
P_EXP = 0.5

# --- leaks --------------------------------------------------------------------
LEAK_CD = 0.75                 # discharge coefficient, as in the BattLeDIM generator
START = "2018-01-01 00:00"

# --- sensor noise (assumed; typical accuracy of field instruments) -----------
NOISE_PRESSURE_M = 0.10        # standard deviation of pressure and level readings [m]
NOISE_FLOW_REL = 0.01          # standard deviation of flow readings, relative

# --- authorised consumption events (assumed) ---------------------------------
EVENTS_PER_YEAR = 20
EVENT_HOURS = (1, 6)           # duration range [h]
EVENT_FLOW_M3H = (5.0, 15.0)   # withdrawal range [m3/h]
