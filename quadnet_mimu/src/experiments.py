import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, List

import yaml

# Placeholder CLI to demonstrate experiment grid setup.
# Users can extend this to enumerate IMU combinations and run train/eval subprocesses.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run QuadNet MIMU experiments sweep")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--splits", type=str, nargs="+", default=["D1"])
    p.add_argument("--modes", type=str, nargs="+", default=["rda", "ara"])
    p.add_argument("--n_imus", type=int, nargs="+", default=[1, 2, 3, 4])
    return p.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runs: List[Dict] = []
    for split, mode, n in itertools.product(args.splits, args.modes, args.n_imus):
        r = {
            "split": split,
            "mode": mode,
            "n_imus": n,
            "cmd_train": f"python -m quadnet_mimu.src.train --config {args.config} --split {split} --mode {mode} --n_imus {n}",
            "cmd_eval": f"python -m quadnet_mimu.src.eval --config {args.config} --split {split} --mode {mode} --n_imus {n} --checkpoint checkpoints/best_{split}_{mode}_{n}im.pt",
        }
        runs.append(r)
    print("Planned runs:")
    for r in runs:
        print(f"- {r['cmd_train']}")
        print(f"  {r['cmd_eval']}")


if __name__ == "__main__":
    main()

