"""Reusable grid search for PyTorch models.

The caller supplies a model builder, a hyperparameter grid and a training function, so nothing here
depends on notebook globals (data, scaling, device).
"""
import itertools
import time

import pandas as pd


def n_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grid_search(builder, grid, train_fn, lr_key="lr", **train_kwargs):
    """Train one model per grid point and rank them by validation loss.

    builder:      callable(**cfg) -> nn.Module, called with every grid key except `lr_key`.
    grid:         {name: [values]}; the Cartesian product is searched. `lr_key` is passed to train_fn instead.
    train_fn:     callable(model, lr=..., **train_kwargs) -> (model, info) where info has
                  "best_val_loss", "epochs" and "train_s".
    Returns (DataFrame sorted by val_loss, total search seconds, best config as a plain dict incl. `lr_key`).
    The frame keeps the trained model and its info in the "_model" / "_info" columns so the winner can be used
    without retraining. The best config is also printed.
    """
    rows, cfgs = [], []
    t0 = time.perf_counter()
    for values in itertools.product(*grid.values()):
        cfg = dict(zip(grid, values))
        lr = cfg.pop(lr_key)
        model, info = train_fn(builder(**cfg), lr=lr, **train_kwargs)
        cfgs.append({**cfg, lr_key: lr})
        rows.append({**cfg, lr_key: lr, "val_loss": info["best_val_loss"], "epochs": info["epochs"],
                     "train_s": info["train_s"], "params": n_params(model), "_model": model, "_info": info})
        print(f"{cfg} {lr_key}={lr:g} -> val {info['best_val_loss']:.4f} ({info['train_s']:.0f} s)")
    total_s = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    best = df["val_loss"].idxmin()                     # rows are in grid order, so this indexes cfgs too
    print(f"Best config: {cfgs[best]} -> val {df.loc[best, 'val_loss']:.4f}")
    return df.sort_values("val_loss"), total_s, cfgs[best]
