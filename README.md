# waterGNN

Intermittency-Robust Leak Detection and Localisation in Water Distribution Networks Using Graph Neural Networks

## Status

Work in progress. This branch contains the dataset generator. The detection models and experiments will follow.

## Dataset generator

The generator takes the public L-Town benchmark network (BattLeDIM) and simulates it under continuous
supply and under intermittent (rationed) supply, with the same leaks, consumption and sensors in both.

```
pip install -r requirements.txt
bash code/run_all.sh
```

`run_all.sh` simulates 2018 and 2019 as one two-year run for each dataset (about 15 minutes per run).

| File | Purpose |
|---|---|
| `code/common.py` | every parameter and assumption |
| `code/01_build_network.py` | supply sectors, isolation valves, household tanks, leak nodes |
| `code/02_simulate.py` | continuous or intermittent simulation, one 5-minute step at a time |
| `data/L-TOWN.inp` | nominal L-Town model, as distributed with the EPyT package |
| `data/battledim_configuration.yaml` | leak timetable and sensor list from the BattLeDIM generator |

Each run writes to `outputs/<run>/`:

| File | Content |
|---|---|
| `scada.csv.gz` | sensor readings with measurement noise: `P_` pressure (m), `F_` flow (m3/h), `L_` tank level (m), `D_` metered inflow (m3/h), `supply_sector` state |
| `leak_flows.csv.gz` | outflow of every leak at every step (m3/h): the labels |
| `events.csv` | authorised consumption events |
| `full_state.npz` | exact pressure, inflow and tank level at every node |

## Sources

- L-Town network and leak scenarios: Vrachimis et al. (2022), Battle of the Leakage Detection and Isolation Methods, JWRPM 148(12). https://github.com/KIOS-Research/BattLeDIM
- EPyT: Kyriakou et al. (2023), JOSS 8(92), 5947.
