"""Streaming evaluation under a HARD review cap, same cap for every method. Writes results.csv, frontier.png, audit.db."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from .config import Cfg
from .env import VecEnv, HUMAN, build_obs
from .featurize import build_tables, A_FEATS, OBS_NAMES
from . import baselines as B
from .ppo_lag import load_policy
from .audit import AuditLog

STAGE = {0: "retrieval", 1: "answer"}
ACTN = ["advance", "re_retrieve", "specialist", "human"]


def run_stream(T, idx, act_fn, cap, cfg, seed, audit=None, qids=None):
    rng = np.random.default_rng(seed); order = rng.permutation(np.asarray(idx))
    env = VecEnv(T, cfg, seed); used = 0; rec = {}
    for q in order:
        obs = env.reset([q]); step = 0
        while not env.done[0]:
            env.human_ok[:] = used < cap
            mask = env.mask(); stage = int(env.stage[0])
            a = int(act_fn(obs, mask)[0])
            nobs, r, c, _ = env.step([a]); used += int(c[0])
            if audit is not None:
                info = getattr(act_fn, "last", None)
                audit.log(qid=qids[q] if qids else str(q), step=step, stage=STAGE[stage], action=ACTN[a],
                          p_human=float(info["mean"][0, HUMAN]) if info else None,
                          p_human_lo=float(info["lo"][0, HUMAN]) if info else None,
                          p_human_hi=float(info["hi"][0, HUMAN]) if info else None,
                          budget_used=used, budget_cap=cap, reward=float(r[0]), cost=float(c[0]),
                          state={n: round(float(v), 4) for n, v in zip(OBS_NAMES, obs[0])})
            obs = nobs; step += 1
        rec[int(q)] = (env.net[0], env.final_f1[0], env.n_h[0], env.n_s[0], env.n_r[0])
    arr = np.array([rec[int(q)] for q in sorted(rec)])
    return arr


def summarise(arrs):
    m = np.mean(arrs, 0)            # [n_queries, 5], averaged across stream orders
    return dict(net_reward=m[:, 0].mean(), f1=m[:, 1].mean(), acc=(m[:, 1] >= 0.5).mean(), human_rate=m[:, 2].mean(),
                spec_rate=m[:, 3].mean(), rereq_rate=m[:, 4].mean()), m[:, 0]


def paired_ci(a, b, n=2000, seed=0):
    d = a - b; rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def signal_audit(T, test_idx, risk_model, rv=0, mv=0):
    """AUROC of each signal for predicting 'answer wrong' in evidence-variant rv / answerer mv."""
    n = len(test_idx); one = np.ones(n, int)
    y = (T.f1[test_idx, rv, mv] < 0.5).astype(int); rows = []
    for f in A_FEATS:
        s = T.A[test_idx, rv, mv, A_FEATS.index(f)]
        if np.std(s) < 1e-9:
            continue                                  # signal not available in this cell
        au = roc_auc_score(y, s)
        rows.append((f, max(au, 1 - au), "higher=err" if au > .5 else "lower=err"))
    rows.append(("risk_gbm (all signals)", roc_auc_score(y, risk_model.risk(build_obs(T, test_idx, one, rv * one, mv * one))), "supervised"))
    df = pd.DataFrame(rows, columns=["signal", "AUROC_for_predicting_error", "direction"])
    df.insert(0, "cell", f"ev={['narrow','wide','gold'][rv]},model={['base','spec'][mv]}")
    return df, float(y.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache"); ap.add_argument("--runs", default="runs")
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--cost_spec", type=float, default=None); ap.add_argument("--cost_rr", type=float, default=None)
    ap.add_argument("--human_acc", type=float, default=None)
    a = ap.parse_args(); cfg = Cfg(); runs = Path(a.runs)
    if a.cost_spec is not None: cfg.cost_specialist = a.cost_spec
    if a.cost_rr is not None: cfg.cost_reretrieve = a.cost_rr
    if a.human_acc is not None: cfg.human_acc = a.human_acc
    meta = json.load(open(Path(a.cache) / "meta.json")); idx = meta["idx"]
    T = build_tables(np.load(Path(a.cache) / "cache.npz"), idx["train"], meta["qids"])
    tr, va, te = idx["train"], idx["val"], idx["test"]
    audit = AuditLog(str(runs / "audit.db"), "eval"); rows, per_q, comps = [], {}, []
    sigs = []
    for rv, mv in [(0, 0), (1, 1)]:
        sg, err = signal_audit(T, te, B.RiskGBM(T, tr, va, 0.1, rv=rv, mv=mv), rv, mv)
        sigs.append(sg); print(f"\nerror rate on test (F1<0.5), {sg.cell.iloc[0]}: {err:.3f}\n{sg.drop(columns='cell').to_string(index=False)}")
    pd.concat(sigs).to_csv(runs / "signal_audit.csv", index=False); print()
    for b in cfg.budgets:
        cap = int(round(b * len(te)))
        nets = [load_policy(p, cfg.hidden) for p in sorted(runs.glob(f"policy_b{b:.2f}_s*.pt"))]
        methods = {"auto_only": B.auto_only, "always_specialist": B.always_specialist,
                   "random_human": B.random_human(b),
                   "conf_logprob": B.threshold_policy("mean_lp", B.calibrate_tau(T, va, "mean_lp", b)),
                   "critic_pyes": B.threshold_policy("critic_yes", B.calibrate_tau(T, va, "critic_yes", b)),
                   "conf_semantic": B.threshold_policy("sem_agree", B.calibrate_tau(T, va, "sem_agree", b)),
                   "risk_gbm": B.RiskGBM(T, tr, va, b),
                   "wide_only": B.fixed_escalation(True, False),
                   "wide_spec": B.fixed_escalation(True, True),
                   "wide_gbm": B.HeuristicRisk(T, tr, va, b, specialist=False),
                   "wide_spec_gbm": B.HeuristicRisk(T, tr, va, b, specialist=True)}
        if nets:
            methods["ppo_lag_ensemble"] = B.EnsemblePolicy(nets)                       # argmax decoding
            methods["ppo_lag_sample"] = B.EnsemblePolicy(nets, sample=True, seed=0)    # stochastic, as trained
            methods["ppo_lag_thresh"] = B.ThresholdedPolicy(nets, B.calibrate_human_tau(nets, T, va, b, cfg))
        for name, fn in methods.items():
            arrs = [run_stream(T, te, fn, cap, cfg, s, audit if (name == "ppo_lag_ensemble" and s == 0) else None, meta["qids"])
                    for s in range(a.orders)]
            s_, pq = summarise(arrs); per_q[(b, name)] = pq; rows.append(dict(budget=b, method=name, **s_))
        o = B.oracle_hindsight(T, te, cap, cfg)
        rows.append(dict(budget=b, method="oracle_hindsight", net_reward=o.mean(), f1=np.nan, acc=np.nan,
                         human_rate=cap / len(te), spec_rate=np.nan, rereq_rate=np.nan))
        for ppo in ["ppo_lag_ensemble", "ppo_lag_sample", "ppo_lag_thresh"]:
            if (b, ppo) in per_q:
                for other in ["wide_spec_gbm", "wide_gbm", "auto_only"]:
                    d, lo, hi = paired_ci(per_q[(b, ppo)], per_q[(b, other)])
                    comps.append(dict(budget=b, policy=ppo, baseline=other, diff=d, ci_lo=lo, ci_hi=hi))
                    print(f"b={b:.2f}  {ppo:17s} minus {other:13s}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    audit.close()
    pd.DataFrame(comps).to_csv(runs / "comparisons.csv", index=False)
    df = pd.DataFrame(rows); df.to_csv(runs / "results.csv", index=False); print("\n", df.round(4).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4.5))
    for m_, g in df.groupby("method"):
        g = g.sort_values("budget"); plt.plot(g.human_rate, g.net_reward, marker="o", label=m_,
                                              ls="--" if m_ == "oracle_hindsight" else "-")
    plt.xlabel("human reviews per query (hard cap)"); plt.ylabel("net reward (F1 - compute cost)")
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(runs / "frontier.png", dpi=160)


if __name__ == "__main__":
    main()
