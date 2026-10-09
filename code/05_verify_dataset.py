"""Step 5 - check every generated run against the experimental design."""
import json
import numpy as np
import pandas as pd
import yaml
from common import *

leaks = pd.read_csv(DATA / "leaks.csv", parse_dates=["start", "end"])
cfg = yaml.safe_load(open(DATA / "battledim_configuration.yaml"))
EXPECT = 730 * 86400 // DT
rows = []
for run in sorted(p for p in OUT.iterdir() if (p / "meta.json").exists()):
    meta = json.loads((run / "meta.json").read_text())
    z = np.load(run / "full_state.npz")
    lf = pd.read_csv(run / "leak_flows.csv.gz", index_col=0, parse_dates=True)
    sc = pd.read_csv(run / "scada.csv.gz", index_col=0, parse_dates=True)
    P = z["pressure"]; names = list(z["nodes"])
    sens = cfg["pressure_sensors"][0]
    exact = P[:, names.index(sens)]
    live = exact > 0
    noise = float(np.std(sc[f"P_{sens}"].to_numpy()[live] - exact[live])) if live.any() else np.nan
    ev = pd.read_csv(run / "events.csv") if (run / "events.csv").exists() else pd.DataFrame()
    rows.append(dict(
        run=run.name, steps=len(sc), full_length=len(sc) == EXPECT,
        leaks_2018=int((lf.loc[:"2018-12-31"].max() > 0).sum()),
        leaks_2019=int((lf.loc["2019-01-01":].max() > 0).sum()),
        events=len(ev), events_delivered=int((ev.delivered_m3 > 0).sum()) if len(ev) else 0,
        noise_sd_m=round(noise, 3), nan=int(np.isnan(P).sum() + sc.isna().sum().sum()),
        supply_h_week=np.round(z["state"].mean(0) * 168, 1).tolist(),
        sensors=sum(c.startswith("P_") for c in sc.columns),
    ))
out = pd.DataFrame(rows)
out.to_csv(OUT / "verification.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(out.to_string(index=False))
