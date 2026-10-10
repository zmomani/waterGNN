"""Step 6 - train the reconstruction network and the no-graph baseline.

  python 06_train_reconstructor.py --run continuous_CWS_noleak --tag cws

Both learn the pressure at every node from the 33 pressure sensors, using
leak-free data of 2018: January-October for training, November-December for
validation. The baseline is a ridge regression with no knowledge of the pipes.
"""
import argparse, json, time
import numpy as np
import torch
import gnn_data as g
from gnn_model import Reconstructor
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="continuous_CWS_noleak")
ap.add_argument("--tag", default="cws")
ap.add_argument("--epochs", type=int, default=16)
ap.add_argument("--samples", type=int, default=2500, help="random time steps drawn per epoch")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--aware", action="store_true", help="give the model the supply state")
ap.add_argument("--hide", action="store_true", help="hide one sensor group per sample (for detection)")
args = ap.parse_args()
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)

run = g.load_run(args.run, 2018)
split = int(np.searchsorted(run["time"], np.datetime64("2018-11-01")))
warm = 4 * STEPS_PER_WEEK                             # first four weeks excluded as warm-up
train_idx = np.arange(warm, split)
val_idx = np.sort(rng.choice(np.arange(split, len(run["time"])), 600, replace=False))
mask = torch.tensor(g.IS_JUNCTION)
# Each node's normal pressure and its spread, from the training period. The network
# learns deviations from normal; without this it spends its capacity on the fixed
# pressure differences between nodes, which elevation already explains.
_t, _l = run["truth"][train_idx[::12]], run["live"][train_idx[::12]]      # statistics over pressurised steps only
MU = (_t * _l).sum(0) / np.maximum(_l.sum(0), 1)
SD = np.maximum(np.sqrt((((_t - MU) ** 2) * _l).sum(0) / np.maximum(_l.sum(0), 1)), 0.05)
np.savez(MODELS / f"norm_{args.tag}.npz", mu=MU, sd=SD)
SD_T = torch.tensor(SD)


def batch_of(idx):
    sub = {k: run[k][idx] for k in ("time", "sensors", "level", "live", "since")}
    x = torch.tensor(g.node_features(sub, MU, SD, args.aware))
    if args.hide:                                     # a different hidden group for each third of the batch
        for k in range(3):
            x[k::3] = g.hide(x[k::3], k)
    return x, torch.tensor((run["truth"][idx] - MU) / SD), torch.tensor(run["live"][idx] & g.IS_JUNCTION)


model = Reconstructor(g.EDGE_INDEX, g.N, in_channels=13 if args.aware else 11)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
steps = args.epochs * (args.samples // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)


def evaluate():
    model.eval(); err = []
    with torch.no_grad():
        for i in range(0, len(val_idx), 100):
            x, y, m = batch_of(val_idx[i:i + 100])
            e = ((model(x) - y).abs() * SD_T).numpy(); e[~m.numpy()] = np.nan      # pressurised junctions only
            err.append(e)
    model.train()
    return np.concatenate(err)                        # [samples, nodes], metres


history, tic = [], time.time()
for epoch in range(args.epochs):
    idx = rng.choice(train_idx, args.samples, replace=False)
    losses = []
    for i in range(0, len(idx) - args.batch + 1, args.batch):
        x, y, m = batch_of(np.sort(idx[i:i + args.batch]))
        loss = ((model(x) - y)[m] ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    err = evaluate()
    row = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)),
               val_mae_unmonitored_m=float(np.nanmean(err[:, g.UNMONITORED])),
               val_mae_sensors_m=float(np.nanmean(err[:, g.S_IDX])), minutes=(time.time() - tic) / 60)
    history.append(row); print(json.dumps(row), flush=True)
torch.save(model.state_dict(), MODELS / f"reconstructor_{args.tag}.pt")
gnn_err = evaluate()

summary = dict(run=args.run, epochs=args.epochs, samples_per_epoch=args.samples, aware=args.aware,
               training_minutes=round(history[-1]["minutes"], 1),
               parameters=sum(p.numel() for p in model.parameters()),
               gnn_mae_unmonitored_m=round(float(np.nanmean(gnn_err[:, g.UNMONITORED])), 3),
               gnn_p95_m=round(float(np.nanpercentile(gnn_err[:, g.UNMONITORED], 95)), 3))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / f"reconstruction_{args.tag}.json").write_text(json.dumps(dict(summary=summary, history=history), indent=1))
print(json.dumps(summary, indent=1))
