from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_pred_vs_gt(pred: np.ndarray, gt: np.ndarray, title: Optional[str] = None) -> None:
    t = np.arange(len(pred))
    plt.figure(figsize=(10, 4))
    plt.plot(t, gt, label="GT")
    plt.plot(t, pred, label="Pred")
    plt.xlabel("Window Index")
    plt.ylabel("Delta")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

