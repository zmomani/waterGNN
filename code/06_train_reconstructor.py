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
MU = run["truth"][train_idx[::12]].mean(0)
SD = np.maximum(run["truth"][train_idx[::12]].std(0), 0.05)
np.savez(MODELS / f"norm_{args.tag}.npz", mu=MU, sd=SD)
SD_T = torch.tensor(SD)


def batch_of(idx):
    sub = dict(time=run["time"][idx], sensors=run["sensors"][idx], level=run["level"][idx])
    return torch.tensor(g.node_features(sub, MU, SD)), torch.tensor((run["truth"][idx] - MU) / SD)


model = Reconstructor(g.EDGE_INDEX, g.N)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
steps = args.epochs * (args.samples // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)


def evaluate():
    model.eval(); err = []
    with torch.no_grad():
        for i in range(0, len(val_idx), 100):
            x, y = batch_of(val_idx[i:i + 100])
            err.append(((model(x) - y).abs() * SD_T).numpy())
    model.train()
    return np.concatenate(err)                        # [samples, nodes], metres


history, tic = [], time.time()
for epoch in range(args.epochs):
    idx = rng.choice(train_idx, args.samples, replace=False)
    losses = []
    for i in range(0, len(idx) - args.batch + 1, args.batch):
        x, y = batch_of(np.sort(idx[i:i + args.batch]))
        loss = ((model(x) - y)[:, mask] ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    err = evaluate()
    row = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)),
               val_mae_unmonitored_m=float(err[:, g.UNMONITORED].mean()),
               val_mae_sensors_m=float(err[:, g.S_IDX].mean()), minutes=(time.time() - tic) / 60)
    history.append(row); print(json.dumps(row), flush=True)
torch.save(model.state_dict(), MODELS / f"reconstructor_{args.tag}.pt")
gnn_err = evaluate()

# ---------------------------------------------------- baseline: ridge --------
def flat(idx):
    return np.column_stack([run["sensors"][idx] / g.P_SCALE, run["level"][idx] / 4.0,
                            g.time_features(run["time"][idx]), np.ones(len(idx))])

tr = train_idx[::6]                                   # every 30 minutes
A, Y = flat(tr), run["truth"][tr] / g.P_SCALE
W = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y)
ridge_err = np.abs(flat(val_idx) @ W - run["truth"][val_idx] / g.P_SCALE) * g.P_SCALE
np.save(MODELS / f"ridge_{args.tag}.npy", W)

summary = dict(run=args.run, epochs=args.epochs, samples_per_epoch=args.samples,
               training_minutes=round(history[-1]["minutes"], 1),
               parameters=sum(p.numel() for p in model.parameters()),
               gnn_mae_unmonitored_m=round(float(gnn_err[:, g.UNMONITORED].mean()), 3),
               ridge_mae_unmonitored_m=round(float(ridge_err[:, g.UNMONITORED].mean()), 3),
               gnn_p95_m=round(float(np.percentile(gnn_err[:, g.UNMONITORED], 95)), 3),
               ridge_p95_m=round(float(np.percentile(ridge_err[:, g.UNMONITORED], 95)), 3))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / f"reconstruction_{args.tag}.json").write_text(json.dumps(dict(summary=summary, history=history), indent=1))
np.savez_compressed(ROOT / "results" / f"reconstruction_{args.tag}_errors.npz",
                    gnn=gnn_err.mean(0), ridge=ridge_err.mean(0))
print(json.dumps(summary, indent=1))
