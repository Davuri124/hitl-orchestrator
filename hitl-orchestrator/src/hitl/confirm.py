"""Confirmatory evaluation on FRESH queries. Everything is frozen: policies, risk models, thresholds, decoding, comparators.

  python -m hitl.confirm prereg   --main_cache cache --runs final_stacked --out confirm      (BEFORE building the fresh cache)
  python -m hitl.build_cache ...  --n_train 0 --n_val 0 --n_test 2000 --dev_offset 2000 --cache_dir cache_fresh   (GPU)
  python -m hitl.confirm evaluate --main_cache cache --fresh_cache cache_fresh --runs final_stacked --out confirm

Fresh = official HotpotQA dev (distractor) questions at shuffled positions [2000, 4000); the main test used [0, 2000).
"""
import argparse, hashlib, json, datetime
from pathlib import Path
import numpy as np, pandas as pd
from .config import Cfg
from .featurize import fit_transforms, apply_transforms
from .env import build_obs
from . import baselines as B
from .stacked import risk_table, fit_models, predict_models, risk_stats, to_feature_with, to_feature, augment
from .stream import StreamRunner, make_stream
from .ppo_lag import load_policy
from .run_final import simulate, oracle, per_query, load_types, mdt

PRIMARY_BUDGETS = [0.02, 0.05, 0.10]
BUDGETS = [0.02, 0.05, 0.10, 0.20, 0.30]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")): h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def cmd_prereg(a):
    main, runs, out = Path(a.main_cache), Path(a.runs), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = {"main/cache.npz": sha(main / "cache.npz"), "main/meta.json": sha(main / "meta.json"), "runs/risk_feature.npy": sha(runs / "risk_feature.npy")}
    pols = sorted(runs.glob("stream_s*.pt")); assert pols, f"no stream_s*.pt in {runs}"
    files.update({f"runs/{p.name}": sha(p) for p in pols})
    pre = {"created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "code_sha256": code_hash(), "files": files,
           "fresh_split": "HotpotQA distractor dev, shuffled with seed 42, positions [2000, 4000); main test used [0, 2000)",
           "primary_budgets": PRIMARY_BUDGETS, "alpha": 0.05, "resamples": 10000}
    json.dump(pre, open(out / "prereg.json", "w"), indent=1)
    md = f"""# Pre-registration (self-registered before the fresh cache is built)
Created (UTC): {pre['created_utc']}
Code hash (sha256 of src/hitl/*.py): `{pre['code_sha256']}`

## Frozen artefacts (hashes in prereg.json)
{chr(10).join(f"- {k}: `{v[:16]}...`" for k, v in files.items())}

## Data
Fresh queries: {pre['fresh_split']}. No fresh query was used for training, validation, model selection or any design decision.

## Primary hypotheses (confirmatory)
For each budget b in {PRIMARY_BUDGETS} on the **iid** stream (window W=1000, 3 random orders, hard cap round(b*W) reviews per window):
mean per-query net reward of `ppo_stream` (risk-stacked policy, 3-seed ensemble, sampling decoding) is **greater than** that of
`paced_wide_spec_gbm`. One-sided paired bootstrap over queries (10,000 resamples), **Holm correction over these 3 tests, alpha = 0.05**.
Hypothesis is CONFIRMED only if its Holm-adjusted p < 0.05.

## Everything else is exploratory
Other budgets (0.20, 0.30), type_blocks, hard_first, per-seed results, the no-risk ablation: descriptive only, no inferential claims.

## Rules
No retraining, re-tuning, threshold changes, or re-selection of decoding / comparator after seeing fresh results.
The evaluation refuses to run if code or artefact hashes differ from this file. Negative results are reported as such.
Note: self-registration is weaker than an external registry; commit this file to a public repo / OSF before building the fresh cache.
"""
    (out / "PREREGISTRATION.md").write_text(md, encoding="utf-8"); print(md)


def boot(d, B_=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(d)
    m = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B_)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((np.sum(m <= 0) + 1) / (B_ + 1))


def holm(ps):
    order = np.argsort(ps); adj = np.zeros(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def cmd_evaluate(a):
    cfg = Cfg(); out = Path(a.out); main, fresh, runs = Path(a.main_cache), Path(a.fresh_cache), Path(a.runs)
    pre = json.load(open(out / "prereg.json")); checks = []
    assert pre["code_sha256"] == code_hash(), "CODE CHANGED since pre-registration: results would not be confirmatory"
    cur = {"main/cache.npz": main / "cache.npz", "main/meta.json": main / "meta.json", "runs/risk_feature.npy": runs / "risk_feature.npy"}
    cur.update({f"runs/{p.name}": p for p in runs.glob("stream_s*.pt")})
    assert set(cur) == set(pre["files"]) and all(sha(p) == pre["files"][k] for k, p in cur.items()), "FROZEN ARTEFACT CHANGED since pre-registration"
    checks.append("code hash and all frozen artefact hashes match the pre-registration")

    meta = json.load(open(main / "meta.json")); idx = meta["idx"]; tr, va = idx["train"], idx["val"]
    cm = np.load(main / "cache.npz"); tf = fit_transforms(cm, tr); T = apply_transforms(cm, tf, meta["qids"])
    mf = json.load(open(fresh / "meta.json")); cf = np.load(fresh / "cache.npz"); Tf = apply_transforms(cf, tf, mf["qids"])
    assert not (set(mf["qids"]) & set(meta["qids"])), "fresh queries overlap with main queries"
    checks.append(f"fresh queries ({len(mf['qids'])}) are disjoint from all {len(meta['qids'])} main queries")

    # frozen risk machinery (refit deterministically from main train; verified against the saved training feature)
    rt = risk_table(T, tr, log=lambda *x, **k: None); mu, sd = risk_stats(rt, tr)
    same = np.allclose(to_feature_with(rt, mu, sd), np.load(runs / "risk_feature.npy"), atol=1e-3)
    checks.append("recomputed risk feature matches the one the policies were trained on" if same else
                  "WARNING: recomputed risk feature differs slightly from the training one (non-determinism); interpret with care")
    models = fit_models(T, tr); Tfp = augment(Tf, to_feature_with(predict_models(models, Tf), mu, sd))
    g11 = B.RiskGBM(T, tr, va, 0.1, rv=1, mv=1); g10 = B.RiskGBM(T, tr, va, 0.1, rv=1, mv=0)
    Nf = Tf.f1.shape[0]; allf = np.arange(Nf); one = np.ones(Nf, int)
    risk11 = g11.risk(build_obs(Tf, allf, one, one, one)); risk10 = g10.risk(build_obs(Tf, allf, one, one, 0 * one))
    nv = len(va); ov = np.ones(nv, int)
    v11 = np.sort(g11.risk(build_obs(T, va, ov, ov, ov))); v10 = np.sort(g10.risk(build_obs(T, va, ov, ov, 0 * ov)))
    qtau = lambda srt, rate: -np.inf if rate >= 1 else srt[int(np.clip(np.floor((1 - rate) * len(srt)), 0, len(srt) - 1))]
    static = lambda srt: (lambda rem, left, b: qtau(srt, b)); paced = lambda srt: (lambda rem, left, b: qtau(srt, rem / left))
    c_rr, c_sp, h = cfg.cost_reretrieve, cfg.cost_specialist, cfg.human_acc
    nets = [load_policy(p, cfg.hidden) for p in sorted(runs.glob("stream_s*.pt"))]
    nr = [load_policy(p, cfg.hidden) for p in sorted(Path(a.norisk_dir).glob("stream_s*.pt"))] if a.norisk_dir and Path(a.norisk_dir).exists() else []
    ty = load_types(fresh, mf); W = min(cfg.eval_W, Nf)
    run_p, run_n = StreamRunner(Tfp, cfg, 0), StreamRunner(Tf, cfg, 0)
    rng = np.random.default_rng(321); rows, comps, seedrows = [], [], []
    for kind in ["iid", "type_blocks", "hard_first"]:
        if kind == "type_blocks" and not ty: continue
        n_ord = cfg.eval_orders if kind != "hard_first" else 1
        wins = np.concatenate([make_stream(kind, allf, Tf.f1[:, 0, 0], ty["type_code"] if ty else None, W, rng) for _ in range(n_ord)])
        for b in BUDGETS:
            bs = np.full(len(wins), b); perq = {}; z, o1 = np.zeros(wins.shape), np.ones(wins.shape)
            def rec(name, r):
                perq[name] = per_query(wins, r["net"], Nf)
                rows.append(dict(stream=kind, budget=b, method=name, net_reward=r["net"].mean(), f1=r["f1"].mean(), human_rate=r["h"].mean(),
                                 windows_over_budget=int((r["h"].sum(1) > r["K"]).sum())))
            rec("auto_only", dict(net=Tf.f1[wins, 0, 0], f1=Tf.f1[wins, 0, 0], h=z, K=np.ones(len(wins))))
            rec("wide_spec_no_human", dict(net=Tf.f1[wins, 1, 1] - c_rr - c_sp, f1=Tf.f1[wins, 1, 1], h=z, K=np.ones(len(wins))))
            rec("static_wide_gbm", simulate(wins, bs, Tf.f1[:, 1, 0], risk10, c_rr, static(v10), h))
            rec("paced_wide_gbm", simulate(wins, bs, Tf.f1[:, 1, 0], risk10, c_rr, paced(v10), h))
            rec("static_wide_spec_gbm", simulate(wins, bs, Tf.f1[:, 1, 1], risk11, c_rr + c_sp, static(v11), h))
            rec("paced_wide_spec_gbm", simulate(wins, bs, Tf.f1[:, 1, 1], risk11, c_rr + c_sp, paced(v11), h))
            r, _ = run_p.run(wins, bs, act_fn=B.EnsemblePolicy(nets, sample=True, seed=0)); rec("ppo_stream", r)
            if kind == "iid":
                for si, net in enumerate(nets):
                    r1, _ = run_p.run(wins, bs, act_fn=B.EnsemblePolicy([net], sample=True, seed=si)); seedrows.append(dict(budget=b, seed=si, net_reward=r1["net"].mean()))
            if nr:
                r2, _ = run_n.run(wins, bs, act_fn=B.EnsemblePolicy(nr, sample=True, seed=0)); rec("ppo_stream_no_risk", r2)
            rec("oracle_hindsight", oracle(wins, bs, Tf, cfg))
            for other in ["paced_wide_spec_gbm", "paced_wide_gbm"]:
                d, lo, hi, p1 = boot(perq["ppo_stream"] - perq[other])
                comps.append(dict(stream=kind, budget=b, baseline=other, diff=d, ci_lo=lo, ci_hi=hi, p_one_sided=p1))
            print(f"[{kind:11s} b={b:.2f}] ppo {perq['ppo_stream'].mean():.4f} | paced_wide_spec_gbm {perq['paced_wide_spec_gbm'].mean():.4f}", flush=True)
    R, C = pd.DataFrame(rows), pd.DataFrame(comps); R.to_csv(out / "confirm_results.csv", index=False); C.to_csv(out / "confirm_comparisons.csv", index=False)
    S = pd.DataFrame(seedrows); S.to_csv(out / "confirm_per_seed.csv", index=False)

    prim = C[(C.stream == "iid") & (C.baseline == "paced_wide_spec_gbm") & C.budget.isin(PRIMARY_BUDGETS)].copy().sort_values("budget")
    prim["p_holm"] = holm(prim.p_one_sided.values); prim["confirmed"] = prim.p_holm < pre["alpha"]
    nconf = int(prim.confirmed.sum())
    expl = C[~((C.stream == "iid") & (C.baseline == "paced_wide_spec_gbm") & C.budget.isin(PRIMARY_BUDGETS))].copy()
    expl["note"] = np.where(expl.ci_lo > 0, "ppo > comparator (exploratory)", np.where(expl.ci_hi < 0, "ppo < comparator (exploratory)", "no clear difference"))
    md = ["# Confirmatory report (fresh queries, everything frozen)\n", "## Integrity checks\n"] + [f"- {c}" for c in checks] + [
          f"- budget violations: {int(R.windows_over_budget.sum())} windows over budget (must be 0)\n",
          f"Fresh queries: {Nf}. Window W={W}. Costs: re-retrieve {c_rr}, specialist {c_sp}; human accuracy {h}.\n",
          f"## PRIMARY RESULT: {nconf}/3 hypotheses confirmed\n",
          mdt(prim[["budget", "diff", "ci_lo", "ci_hi", "p_one_sided", "p_holm", "confirmed"]].round(4), index=False),
          "\nDiff = ppo_stream minus paced_wide_spec_gbm (net reward per query), iid stream. Confirmed = Holm-adjusted one-sided p < 0.05.\n",
          "## Exploratory: all other cells (no inferential claims)\n", mdt(expl[["stream", "budget", "baseline", "diff", "ci_lo", "ci_hi", "note"]].round(4), index=False),
          "\n## Net reward on fresh queries\n"]
    for kind in R.stream.unique():
        md += [f"### {kind}\n", mdt(R[R.stream == kind].pivot(index="method", columns="budget", values="net_reward").round(4)), ""]
    md += ["## Per-seed (iid) net reward, single-seed policies\n", mdt(S.pivot(index="seed", columns="budget", values="net_reward").round(4)),
           "\nLimits: HotpotQA only, Qwen 3B/7B, oracle human, specialist cost is an assumption, offline replay (not live serving)."]
    (out / "CONFIRM_REPORT.md").write_text("\n".join(md), encoding="utf-8"); print("\n" + "\n".join(md[:14]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prereg", "evaluate"])
    ap.add_argument("--main_cache", default="cache"); ap.add_argument("--fresh_cache", default="cache_fresh")
    ap.add_argument("--runs", default="final_stacked"); ap.add_argument("--out", default="confirm")
    ap.add_argument("--norisk_dir", default=None, help="optional: stream_s*.pt trained WITHOUT the risk feature (ablation row)")
    a = ap.parse_args(); {"prereg": cmd_prereg, "evaluate": cmd_evaluate}[a.cmd](a)
