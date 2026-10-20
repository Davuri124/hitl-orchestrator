import argparse, json
from pathlib import Path
import numpy as np
from .config import Cfg
from .featurize import build_tables
from .ppo_lag import train_ppo_lag, save_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache"); ap.add_argument("--runs", default="runs")
    ap.add_argument("--iters", type=int, default=None); ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--cost_spec", type=float, default=None); ap.add_argument("--cost_rr", type=float, default=None)
    ap.add_argument("--human_acc", type=float, default=None)
    a = ap.parse_args(); cfg = Cfg()
    if a.cost_spec is not None: cfg.cost_specialist = a.cost_spec
    if a.cost_rr is not None: cfg.cost_reretrieve = a.cost_rr
    if a.human_acc is not None: cfg.human_acc = a.human_acc
    if a.iters: cfg.iters = a.iters
    if a.seeds: cfg.n_seeds = a.seeds
    runs = Path(a.runs); runs.mkdir(exist_ok=True, parents=True)
    meta = json.load(open(Path(a.cache) / "meta.json"))
    T = build_tables(np.load(Path(a.cache) / "cache.npz"), meta["idx"]["train"], meta["qids"])
    D = T.Q.shape[1] + 6 + T.R.shape[-1] + T.A.shape[-1]
    for b in cfg.budgets:
        for s in range(cfg.n_seeds):
            ac, lam, hist = train_ppo_lag(T, meta["idx"]["train"], b, cfg, seed=s, val_idx=meta["idx"]["val"])
            save_policy(runs / f"policy_b{b:.2f}_s{s}.pt", ac, lam, b, D)
            json.dump(hist, open(runs / f"hist_b{b:.2f}_s{s}.json", "w"))


if __name__ == "__main__":
    main()
