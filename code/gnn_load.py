"""Build a reconstructor with the architecture its training run recorded."""
import json
import torch
import gnn_data as g
from gnn_model import Reconstructor
from common import *


def load_model(tag, aware=False):
    MODELS = ROOT / "models"
    cfgf = MODELS / f"config_{tag}.json"
    cfg = {k: v for k, v in json.load(open(cfgf)).items() if k in ("in_channels", "hidden", "layers", "order", "embed")} if cfgf.exists() \
        else dict(in_channels=13 if aware else 11)
    model = Reconstructor(g.EDGE_INDEX, g.N, **cfg)
    model.load_state_dict(torch.load(MODELS / f"reconstructor_{tag}.pt")); return model.eval()
