# waterGNN dataset (v1)

Simulated sensor data for the L-Town network, 2018-2019 at 5-minute resolution, under continuous
supply and three intermittent supply schedules. The code that generates it is in the `code` folder
of the main line of this repository.

| Folder | Supply | Leaks |
|---|---|---|
| `continuous_CWS_leaks` | continuous | 33 BattLeDIM leaks, 40 authorised consumption events |
| `continuous_CWS_noleak` | continuous | none |
| `intermittent_S48_leaks` / `_noleak` | rationed, mean 48 h per week | as above / none |
| `intermittent_S24_leaks` / `_noleak` | rationed, mean 24 h per week | as above / none |
| `intermittent_S72_leaks` / `_noleak` | rationed, mean 72 h per week | as above / none |

Files in each folder, split by year:

| File | Content |
|---|---|
| `scada_<year>.csv.gz` | sensor readings with measurement noise: `P_` pressure (m) at 33 nodes, `F_` flow (m3/h), `L_T1` tank level (m), `D_` metered inflow (m3/h) at 82 nodes, `supply_sector0..5` (1 = sector in supply) |
| `leak_flows_<year>.csv.gz` | outflow of every leak at every step (m3/h): the labels |
| `events.csv` | authorised consumption events (node, start, end, flow) |
| `meta.json` | run settings |

`verification.csv` lists the checks made on every run. Intended use: train on the leak-free 2018
data, tune on the 2018 leaks, test on the 2019 leaks.

Load a file directly:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/zmomani/waterGNN/dataset/intermittent_S48_leaks/scada_2019.csv.gz"
scada = pd.read_csv(url, index_col=0, parse_dates=True)
```

The all-node state files (pressure at every node) are not stored here because of their size;
regenerate them with `code/run_all.sh`.

Network and leak scenarios: Vrachimis et al. (2022), Battle of the Leakage Detection and Isolation
Methods, JWRPM 148(12). https://github.com/KIOS-Research/BattLeDIM
