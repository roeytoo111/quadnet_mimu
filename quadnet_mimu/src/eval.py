import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .datasets import QuadMIMUDataset
from .models import build_model
from .utils import Normalizer, get_device, load_json


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    err = pred - gt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    std = float(np.std(err))
    return {"rmse": rmse, "mae": mae, "max": max_abs, "std": std}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QuadNet MIMU")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--split", type=str, default=None, help="Override data.split")
    parser.add_argument("--mode", type=str, default=None, help="Override model.mode")
    parser.add_argument("--n_imus", type=int, default=None, help="Override data.n_imus")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(cfg: Dict, checkpoint_path: str) -> Dict[str, float]:
    device = get_device(cfg.get("device", "auto"))
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    results_dir = Path(cfg.get("evaluation", {}).get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    ds = QuadMIMUDataset(
        dataset_root=data_cfg["dataset_root"],
        split=data_cfg.get("split", "D1"),
        target=data_cfg.get("target", "distance"),
        window_length=int(data_cfg.get("window_length", 120)),
        stride=int(data_cfg.get("stride", 120)),
        n_imus=int(data_cfg.get("n_imus", 4)),
        augment={"enabled": False},
        mock=False,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    sample_x, _, _ = next(iter(loader))
    in_channels = sample_x.shape[1]
    out_dim = 2 if (data_cfg.get("target", "distance") == "both" or model_cfg.get("multi_target", False)) else 1
    n_imus = int(data_cfg.get("n_imus", max(1, in_channels // 6)))

    model = build_model(
        mode=(model_cfg.get("mode", "rda")),
        n_imus=n_imus,
        dropout=float(model_cfg.get("dropout", 0.2)),
        out_dim=out_dim,
        ara_shared_weights=bool(model_cfg.get("ara_shared_weights", True)),
        in_channels=in_channels,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load normalizer if present
    norm_path = data_cfg.get("save_normalizer_to", "./artifacts/normalizer.pkl")
    if os.path.exists(norm_path):
        normalizer = Normalizer.load(norm_path)
    else:
        normalizer = None

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            if normalizer is not None:
                xb = normalizer.transform(xb)
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)
            preds.append(yhat.detach().cpu().numpy())
            gts.append(yb.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).squeeze()
    gt = np.concatenate(gts, axis=0).squeeze()
    m = metrics(pred, gt)

    # Save CSV row
    tag = f"{data_cfg.get('split','D1')}_{model_cfg.get('mode','rda')}_{data_cfg.get('n_imus',4)}im"
    out_csv = results_dir / f"{tag}.csv"
    header = "split,mode,n_imus,rmse,mae,max,std\n"
    row = f"{data_cfg.get('split','D1')},{model_cfg.get('mode','rda')},{data_cfg.get('n_imus',4)},{m['rmse']:.6f},{m['mae']:.6f},{m['max']:.6f},{m['std']:.6f}\n"
    if not out_csv.exists():
        out_csv.write_text(header + row)
    else:
        with open(out_csv, "a") as f:
            f.write(row)
    print("Evaluation:", m)
    return m


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode:
        cfg["model"]["mode"] = args.mode
    if args.split:
        cfg["data"]["split"] = args.split
    if args.n_imus is not None:
        cfg["data"]["n_imus"] = int(args.n_imus)
    if args.device:
        cfg["device"] = args.device
    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()

