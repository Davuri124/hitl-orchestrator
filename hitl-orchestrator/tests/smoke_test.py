"""MECHANICS CHECK ONLY. Random tensors verify env/PPO-Lag/evaluation run end-to-end and the Lagrange multiplier
reacts to the budget. Nothing here is data or a result; real numbers come only from cache.npz built on real runs."""
import sys; sys.path.insert(0, "src")
import numpy as np
from hitl.config import Cfg
from hitl.featurize import Tables, NP
from hitl.ppo_lag import train_ppo_lag
from hitl.evaluate import run_stream, summarise
from hitl import baselines as B

rng = np.random.default_rng(0); N = 600
f1 = (rng.random((N, 3, 2)) < np.array([0.45, 0.6, 0.9])[None, :, None]).astype(np.float32)
Q = rng.normal(size=(N, NP)).astype(np.float32); R = rng.normal(size=(N, 3, 14)).astype(np.float32)
A = rng.normal(size=(N, 3, 2, 21)).astype(np.float32); A[..., 0] += 2 * f1       # one informative signal
T = Tables(f1, Q, R, A)
cfg = Cfg(iters=40, episodes_per_iter=256)
tr = list(range(300))
va = list(range(300, 400)); te = list(range(400, 600))
ac, lam, hist = train_ppo_lag(T, tr, 0.10, cfg, seed=0, val_idx=va)
print("final human/query", hist[-1]["human_rate"], "lambda", lam)
cap = 20
for name, fn in [("wide_only", B.fixed_escalation(True, False)), ("wide_spec_gbm", B.HeuristicRisk(T, tr, va, 0.1, True)), ("wide_gbm", B.HeuristicRisk(T, tr, va, 0.1, False)), ("auto", B.auto_only), ("spec", B.always_specialist), ("random", B.random_human(0.1))]:
    s, _ = summarise([run_stream(T, te, fn, cap, cfg, 0)]); print(name, {k: round(float(v), 3) for k, v in s.items()})
from hitl.ppo_lag import ActorCritic
s, _ = summarise([run_stream(T, te, B.EnsemblePolicy([ac]), cap, cfg, 0)]); print("ppo", {k: round(float(v), 3) for k, v in s.items()})
assert s["human_rate"] <= cap / len(te) + 1e-9, "hard cap violated"
print("SMOKE OK")

from hitl.baselines import calibrate_human_tau, ThresholdedPolicy
tau = calibrate_human_tau([ac], T, va, 0.10, cfg)
for name, fn in [("ppo_sample", B.EnsemblePolicy([ac], sample=True)), ("ppo_thresh", ThresholdedPolicy([ac], tau))]:
    s, _ = summarise([run_stream(T, te, fn, cap, cfg, 0)]); print(name, {k: round(float(v), 3) for k, v in s.items()})
    assert s["human_rate"] <= cap / len(te) + 1e-9
print("DECODE VARIANTS OK")
