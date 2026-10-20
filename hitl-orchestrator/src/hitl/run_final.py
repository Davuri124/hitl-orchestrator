"""ONE command, CPU only, reuses cache.npz (no GPU):
   python -m hitl.run_final --cache cache --out final
1) trains the budget-aware stream policy (several seeds, in parallel)   2) evaluates under a HARD cap per 1,000 queries
on three real-query streams (iid / type-blocks / hardest-first stress)   3) writes FINAL_REPORT.md, CSVs, plots, audit.db
"""
import argparse, json, re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from .config import Cfg
from .featurize import build_tables
from .env import build_obs
from . import baselines as B
from .stream import StreamRunner, make_stream, train_worker, stream_dim
from .ppo_lag import load_policy
from .audit import AuditLog
from .evaluate import paired_ci, signal_audit


# ------------------------------------------------------------------ real metadata (type / level) for drift streams
def load_types(cache: Path, meta):
    f = cache / "types.json"
    if not f.exists():
        try:
            from .data import all_examples
            sz = {k: int(v) for k, v in re.findall(r"--(n_\w+|dev_offset) (\d+)", (cache / "size.txt").read_text())}
            ex, _ = all_examples(Cfg(n_train=sz["n_train"], n_val=sz["n_val"], n_test=sz["n_test"], dev_offset=sz.get("dev_offset", 0)))
            assert [e.qid for e in ex] == meta["qids"], "query order mismatch between dataset and cache"
            json.dump({"type": [e.qtype for e in ex], "level": [e.level for e in ex]}, open(f, "w"))
        except Exception as e:
            print(f"[warn] could not build types.json ({e}); drift streams limited to 'iid' and 'hard_first'")
            return None
    d = json.load(open(f))
    code = lambda xs: np.unique(xs, return_inverse=True)[1]
    return {"type": np.array(d["type"]), "type_code": code(d["type"]), "level_code": code(d["level"])}


# ------------------------------------------------------------------ closed-form heuristics under the same hard cap
def simulate(windows, budgets, f1cell, risk, nonhuman_cost, tau_fn, h=1.0):
    Bn, W = windows.shape; K = np.maximum(1, np.round(budgets * W)).astype(int)
    net, f1, hum = np.zeros((Bn, W)), np.zeros((Bn, W)), np.zeros((Bn, W))
    for i in range(Bn):
        rem = K[i]
        for t in range(W):
            q = windows[i, t]; base = f1cell[q]
            if rem > 0 and risk[q] > tau_fn(rem, W - t, budgets[i]):
                f = h * 1.0 + (1 - h) * base; rem -= 1; hum[i, t] = 1
            else:
                f = base
            f1[i, t], net[i, t] = f, f - nonhuman_cost
    return dict(net=net, f1=f1, h=hum, K=K)


def oracle(windows, budgets, T, cfg):
    f = T.f1
    auto = np.stack([f[:, 0, 0], f[:, 1, 0] - cfg.cost_reretrieve, f[:, 0, 1] - cfg.cost_specialist,
                     f[:, 1, 1] - cfg.cost_reretrieve - cfg.cost_specialist], 1).max(1)
    Bn, W = windows.shape; K = np.maximum(1, np.round(budgets * W)).astype(int)
    net = np.zeros((Bn, W)); hum = np.zeros((Bn, W))
    for i in range(Bn):
        a = auto[windows[i]].copy(); top = np.argsort(-(1 - a))[:K[i]]; a[top] = 1.0; hum[i, top] = 1; net[i] = a
    return dict(net=net, f1=net, h=hum, K=K)


def per_query(windows, arr, n):
    s, c = np.zeros(n), np.zeros(n)
    np.add.at(s, windows.ravel(), arr.ravel()); np.add.at(c, windows.ravel(), 1)
    return s / np.maximum(c, 1)


def mdt(df, **kw):
    try:
        return df.to_markdown(**kw)
    except Exception:                      # tabulate missing -> plain text table
        return "```\n" + df.to_string(**{k: v for k, v in kw.items() if k == "index"}) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="final")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--orders", type=int, default=None); ap.add_argument("--cost_spec", type=float, default=None)
    ap.add_argument("--human_acc", type=float, default=None)
    ap.add_argument("--stack_risk", action="store_true", help="feed out-of-fold supervised risk to the RL policy")
    ap.add_argument("--prev", default=None, help="results_stream.csv of an earlier run, for a side-by-side table")
    ap.add_argument("--budgets", default="0.02,0.05,0.10,0.20,0.30")
    a = ap.parse_args(); cfg = Cfg(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cache = Path(a.cache)
    if a.iters: cfg.s_iters = a.iters
    if a.W: cfg.eval_W = a.W
    if a.orders: cfg.eval_orders = a.orders
    if a.cost_spec is not None: cfg.cost_specialist = a.cost_spec
    if a.human_acc is not None: cfg.human_acc = a.human_acc
    cfg.stack_risk = bool(a.stack_risk)
    budgets = [float(x) for x in a.budgets.split(",")]
    meta = json.load(open(cache / "meta.json")); idx = meta["idx"]; tr, va, te = idx["train"], idx["val"], idx["test"]
    T = build_tables(np.load(cache / "cache.npz"), tr, meta["qids"]); N = T.f1.shape[0]
    ty = load_types(cache, meta)
    Tp = T                                                    # state seen by the RL policy
    if cfg.stack_risk:
        from .stacked import risk_table, to_feature, augment
        rf = out / "risk_feature.npy"
        if not rf.exists():
            print("fitting out-of-fold risk models (6 cells x 6 fits) ...", flush=True)
            np.save(rf, to_feature(risk_table(T, tr), tr))
        Tp = augment(T, np.load(rf))

    # ---------------- 1) train budget-aware stream policies
    todo = [s for s in range(a.seeds) if not (out / f"stream_s{s}.pt").exists()]
    if todo:
        print(f"training stream policies for seeds {todo} ({cfg.s_iters} iters each) ...", flush=True)
        jobs = [(str(cache), str(out), s, cfg) for s in todo]
        if a.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
                for s in ex.map(train_worker, jobs): print(f"  seed {s} done", flush=True)
        else:
            for j in jobs: train_worker(j)
    nets = [load_policy(out / f"stream_s{s}.pt", cfg.hidden) for s in range(a.seeds)]

    # ---------------- 2) baselines' risk models (same signals as the policy sees)
    g11 = B.RiskGBM(T, tr, va, 0.1, rv=1, mv=1); g10 = B.RiskGBM(T, tr, va, 0.1, rv=1, mv=0)
    allq = np.arange(N); one = np.ones(N, int)
    risk11 = g11.risk(build_obs(T, allq, one, one, one)); risk10 = g10.risk(build_obs(T, allq, one, one, 0 * one))
    v11 = np.sort(risk11[va]); v10 = np.sort(risk10[va])
    qtau = lambda srt, rate: -np.inf if rate >= 1 else srt[int(np.clip(np.floor((1 - rate) * len(srt)), 0, len(srt) - 1))]
    static = lambda srt: (lambda rem, left, b: qtau(srt, b))
    paced = lambda srt: (lambda rem, left, b: qtau(srt, rem / left))
    c_rr, c_sp, h = cfg.cost_reretrieve, cfg.cost_specialist, cfg.human_acc

    # ---------------- 3) evaluation streams
    W = min(cfg.eval_W, len(te)); kinds = ["iid", "hard_first"] + (["type_blocks"] if ty else [])
    runner = StreamRunner(Tp, cfg, 0); audit = AuditLog(str(out / "audit.db"), "final-stream")
    rows, comps = [], []; rng = np.random.default_rng(123)
    for kind in kinds:
        n_ord = cfg.eval_orders if kind != "hard_first" else 1
        wins = np.concatenate([make_stream(kind, te, T.f1[:, 0, 0], ty["type_code"] if ty else None, W, rng) for _ in range(n_ord)])
        for b in budgets:
            bs = np.full(len(wins), b); perq = {}
            def rec(name, r, extra=None):
                perq[name] = per_query(wins, r["net"], N)[te]
                viol = int((r["h"].sum(1) > r["K"]).sum())
                rows.append(dict(stream=kind, budget=b, method=name, net_reward=r["net"].mean(), f1=r["f1"].mean(),
                                 human_rate=r["h"].mean(), spec_rate=(r["s"].mean() if "s" in r else (1.0 if extra == "spec" else 0.0)),
                                 rereq_rate=(r["r"].mean() if "r" in r else 1.0), windows_over_budget=viol))
            # no-human references
            rec("auto_only", dict(net=T.f1[wins, 0, 0], f1=T.f1[wins, 0, 0], h=np.zeros(wins.shape), K=np.ones(len(wins)), s=np.zeros(wins.shape), r=np.zeros(wins.shape)))
            rec("wide_spec_no_human", dict(net=T.f1[wins, 1, 1] - c_rr - c_sp, f1=T.f1[wins, 1, 1], h=np.zeros(wins.shape), K=np.ones(len(wins)), s=np.ones(wins.shape), r=np.ones(wins.shape)))
            # heuristics (same action set as the policy: re-retrieve, specialist, humans by risk)
            rec("static_wide_gbm", {**simulate(wins, bs, T.f1[:, 1, 0], risk10, c_rr, static(v10), h), "s": np.zeros(wins.shape), "r": np.ones(wins.shape)})
            rec("paced_wide_gbm", {**simulate(wins, bs, T.f1[:, 1, 0], risk10, c_rr, paced(v10), h), "s": np.zeros(wins.shape), "r": np.ones(wins.shape)})
            rec("static_wide_spec_gbm", {**simulate(wins, bs, T.f1[:, 1, 1], risk11, c_rr + c_sp, static(v11), h), "s": np.ones(wins.shape), "r": np.ones(wins.shape)})
            rec("paced_wide_spec_gbm", {**simulate(wins, bs, T.f1[:, 1, 1], risk11, c_rr + c_sp, paced(v11), h), "s": np.ones(wins.shape), "r": np.ones(wins.shape)})
            # learned budget-aware policy
            do_audit = (kind == "iid" and abs(b - 0.10) < 1e-9)
            ens = B.EnsemblePolicy(nets, sample=True, seed=0)
            r, _ = runner.run(wins, bs, act_fn=ens, audit=audit if do_audit else None, qids=meta["qids"]); rec("ppo_stream", r)
            r, _ = runner.run(wins, bs, act_fn=B.EnsemblePolicy(nets, sample=False)); rec("ppo_stream_argmax", r)
            o = oracle(wins, bs, T, cfg); rec("oracle_hindsight", {**o, "s": np.zeros(wins.shape), "r": np.zeros(wins.shape)})
            heur = ["static_wide_gbm", "paced_wide_gbm", "static_wide_spec_gbm", "paced_wide_spec_gbm"]
            best = max(heur, key=lambda m: perq[m].mean())
            for other in dict.fromkeys([best, "paced_wide_spec_gbm", "static_wide_spec_gbm", "wide_spec_no_human"]):
                d, lo, hi = paired_ci(perq["ppo_stream"], perq[other])
                comps.append(dict(stream=kind, budget=b, policy="ppo_stream", baseline=other + (" (best heuristic)" if other == best else ""),
                                  diff=d, ci_lo=lo, ci_hi=hi))
            print(f"[{kind:11s} b={b:.2f}] ppo {perq['ppo_stream'].mean():.4f} | best heuristic {best} {perq[best].mean():.4f} | oracle {perq['oracle_hindsight'].mean():.4f}", flush=True)
    audit.close()
    R = pd.DataFrame(rows); C = pd.DataFrame(comps); R.to_csv(out / "results_stream.csv", index=False); C.to_csv(out / "comparisons_stream.csv", index=False)

    # ---------------- 4) signal audit + plots + report
    sig = pd.concat([signal_audit(T, te, g11, 1, 1)[0], signal_audit(T, te, B.RiskGBM(T, tr, va, 0.1, rv=0, mv=0), 0, 0)[0]]); sig.to_csv(out / "signal_audit.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4), squeeze=False)
    for ax, kind in zip(axs[0], kinds):
        for m, g in R[R.stream == kind].groupby("method"):
            if m in ("auto_only", "ppo_stream_argmax", "static_wide_gbm"): continue
            g = g.sort_values("budget"); ax.plot(g.budget, g.net_reward, marker="o", ls="--" if m == "oracle_hindsight" else "-", label=m)
        ax.set_title(kind); ax.set_xlabel("human reviews per query (hard cap per window)"); ax.set_ylabel("net reward"); ax.grid(alpha=.3)
    axs[0][0].legend(fontsize=6); plt.tight_layout(); plt.savefig(out / "frontier_streams.png", dpi=150)

    main_c = C[C.baseline.str.contains("best heuristic")].copy()
    verdict = lambda r: "WIN" if r.ci_lo > 0 else ("LOSS" if r.ci_hi < 0 else "TIE")
    main_c["verdict"] = main_c.apply(verdict, axis=1)
    cnt = main_c.verdict.value_counts().to_dict()
    piv = R[R.method.isin(["auto_only", "wide_spec_no_human", "static_wide_spec_gbm", "paced_wide_spec_gbm", "paced_wide_gbm", "ppo_stream", "oracle_hindsight"])]
    md = ["# Final report (auto-generated)\n",
          f"Data: HotpotQA distractor, real runs. train/val/test = {len(tr)}/{len(va)}/{len(te)} queries. Eval window W={W}, "
          f"hard cap = round(b*W) human reviews per window. Costs: re-retrieve {c_rr}, specialist {c_sp}; human accuracy {h}.\n",
          f"Policy input includes out-of-fold supervised risk (stacked): {cfg.stack_risk}.\n",
          f"Budget compliance: windows over budget = {int(R.windows_over_budget.sum())} (must be 0).\n",
          "## Learned policy vs best heuristic (same action set, same cap)\n", f"Cells: {cnt}\n",
          mdt(main_c[["stream", "budget", "baseline", "diff", "ci_lo", "ci_hi", "verdict"]].round(4), index=False),
          "\n## Net reward by stream and budget\n"]
    for kind in kinds:
        md += [f"### {kind}\n", mdt(piv[piv.stream == kind].pivot(index="method", columns="budget", values="net_reward").round(4)), ""]
    if a.prev and Path(a.prev).exists():
        P = pd.read_csv(a.prev); P = P[P.method == "ppo_stream"][["stream", "budget", "net_reward"]].rename(columns={"net_reward": "ppo_prev"})
        Q = R[R.method == "ppo_stream"][["stream", "budget", "net_reward"]].rename(columns={"net_reward": "ppo_stacked"})
        M = P.merge(Q, on=["stream", "budget"]); M["delta"] = M.ppo_stacked - M.ppo_prev
        md += ["## PPO without vs with stacked risk (net reward)\n", mdt(M.round(4), index=False), ""]
    md += ["## Signal audit (AUROC for predicting a wrong answer, test set)\n", mdt(sig.round(3), index=False),
           "\n## How to read this\n- WIN/LOSS = paired-bootstrap 95% CI vs the best heuristic excludes zero; TIE otherwise. 15 cells are tested, so isolated borderline wins are expected by chance.",
           "- `hard_first` is a stress test (queries ordered by ground-truth difficulty), not a deployment estimate.",
           "- Human reviewer is an oracle (gold answer) unless `--human_acc` < 1.\n- Results are offline replay of real pipeline runs, not live serving."]
    (out / "FINAL_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md[:8])); print(f"\nDONE -> {out}/FINAL_REPORT.md")


if __name__ == "__main__":
    main()
