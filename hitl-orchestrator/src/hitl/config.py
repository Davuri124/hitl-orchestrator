from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cfg:
    cache_dir: Path = Path("cache")
    out_dir: Path = Path("runs")
    seed: int = 42
    # data (HotpotQA distractor: train -> train/val, official dev -> test)
    n_train: int = 3000
    n_val: int = 1000
    n_test: int = 2000
    dev_offset: int = 0            # >0 selects a DISJOINT slice of the shuffled dev set (fresh confirmatory queries)
    # models (all real, all open)
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    spec_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    n_samples: int = 5              # sampled answers per cell for semantic-agreement signals
    k_narrow: int = 2
    k_wide: int = 5
    # compute costs charged inside the reward (human cost lives in the CMDP constraint)
    cost_reretrieve: float = 0.02
    cost_specialist: float = 0.05
    human_acc: float = 1.0          # 1.0 = oracle reviewer (gold answer). Sweep <1 as a sensitivity study.
    # constraint: expected human reviews per query (0.10 == 100 per 1,000 queries)
    budgets: tuple = (0.02, 0.05, 0.10, 0.20, 0.30)
    # PPO-Lagrangian
    iters: int = 300
    episodes_per_iter: int = 1024
    hidden: int = 128
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 512
    ent_coef: float = 0.01
    gae_lambda: float = 0.95
    dual_lr: float = 1.0
    lam_init: float = 0.5
    lam_max: float = 20.0
    n_seeds: int = 5
    # stream-level budgeted policy (the production setting: hard cap of K reviews per window of W queries)
    s_iters: int = 250
    s_windows: int = 32             # parallel training windows per iteration
    s_W: int = 200                  # training window length (budget features are fractions, so it transfers to W=1000)
    s_epochs: int = 4
    s_minibatch: int = 2048
    s_gae_lambda: float = 0.98
    s_budget_lo: float = 0.02
    s_budget_hi: float = 0.30
    s_val_every: int = 20
    eval_W: int = 1000              # "per 1,000 queries"
    eval_orders: int = 3
    stack_risk: bool = False        # give the RL policy the out-of-fold supervised risk score as an input feature
