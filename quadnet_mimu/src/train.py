import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import QuadMIMUDataset
from .models import build_model
from .utils import Normalizer, compute_channel_stats, get_device, set_seed


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QuadNet MIMU")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default=None, help="Override model.mode")
    parser.add_argument("--split", type=str, default=None, help="Override data.split")
    parser.add_argument("--n_imus", type=int, default=None, help="Override data.n_imus")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_dataloaders(cfg: Dict, mock: bool) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    ds = QuadMIMUDataset(
        dataset_root=data_cfg["dataset_root"],
        split=data_cfg.get("split", "D1"),
        target=data_cfg.get("target", "distance"),
        window_length=int(data_cfg.get("window_length", 120)),
        stride=int(data_cfg.get("stride", 120)),
        n_imus=int(data_cfg.get("n_imus", 4)),
        per_imu_normalization=bool(data_cfg.get("per_imu_normalization", False)),
        augment=data_cfg.get("augment", {"enabled": False}),
        mock=mock,
    )
    # 80/20 split for train/val if no explicit split mapping
    val_fraction = 0.2
    val_len = max(1, int(len(ds) * val_fraction))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"], drop_last=False)
    return train_loader, val_loader


def fit(cfg: Dict, mock: bool = False) -> None:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    train_loader, val_loader = create_dataloaders(cfg, mock=mock)
    sample_x, _, _ = next(iter(train_loader))
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

    # Normalization: compute per-channel stats on a few training batches
    def stats_iter():
        for i, (xb, _, _) in enumerate(train_loader):
            if i >= 10:
                break
            yield xb
    stats = compute_channel_stats(stats_iter(), num_channels=in_channels)
    normalizer = Normalizer(stats)
    norm_path = data_cfg.get("save_normalizer_to", "./artifacts/normalizer.pkl")
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    normalizer.save(norm_path)

    if train_cfg.get("optimizer", "adam").lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    else:
        optimizer = optim.SGD(model.parameters(), lr=float(train_cfg["lr"]), momentum=0.9, weight_decay=float(train_cfg.get("weight_decay", 0.0)))

    scheduler_name = train_cfg.get("scheduler", {}).get("name", "reduce_on_plateau")
    if scheduler_name == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(train_cfg["scheduler"].get("patience", 10)), factor=float(train_cfg["scheduler"].get("factor", 0.5)))
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, int(train_cfg.get("epochs", 200))))
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda")
    writer = SummaryWriter(log_dir=train_cfg.get("log_dir", "./runs"))
    ckpt_dir = Path(train_cfg.get("ckpt_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    criterion = nn.MSELoss()

    for epoch in range(int(train_cfg.get("epochs", 200))):
        # Train
        model.train()
        train_loss_sum, train_rmse_sum, train_count = 0.0, 0.0, 0
        for xb, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} / {train_cfg.get('epochs', 200)}"):
            xb = normalizer.transform(xb).to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * xb.size(0)
            train_rmse_sum += rmse_loss(pred.detach(), yb).item() * xb.size(0)
            train_count += xb.size(0)
        train_loss = train_loss_sum / train_count
        train_rmse = train_rmse_sum / train_count

        # Validation
        model.eval()
        val_loss_sum, val_rmse_sum, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = normalizer.transform(xb).to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_rmse_sum += rmse_loss(pred, yb).item() * xb.size(0)
                val_count += xb.size(0)
        val_loss = val_loss_sum / val_count
        val_rmse = val_rmse_sum / val_count

        if scheduler is not None:
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_rmse)
            else:
                scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("RMSE/train", train_rmse, epoch)
        writer.add_scalar("RMSE/val", val_rmse, epoch)

        # Early stopping and checkpointing
        improved = val_rmse < best_val_rmse - 1e-6
        if improved:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_rmse": best_val_rmse,
                "config": cfg,
            }
            torch.save(ckpt, ckpt_dir / f"best_{data_cfg.get('split','D1')}_{model_cfg.get('mode','rda')}_{data_cfg.get('n_imus',4)}im.pt")
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}: train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")
        if epochs_no_improve >= int(train_cfg.get("early_stopping_patience", 20)):
            print("Early stopping triggered.")
            break

    writer.close()


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
    fit(cfg, mock=args.mock)


if __name__ == "__main__":
    main()

