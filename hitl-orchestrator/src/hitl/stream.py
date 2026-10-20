"""Stream-level budgeted orchestration (the actual topic: a HARD cap of K human reviews per window of W queries).

The orchestrator sees, besides the per-query signals, how much budget is left and how far into the window it is
(rem_frac, pos_frac, pace, target rate). Because the cap is enforced by action masking, the constraint is satisfied by
construction; PPO only has to learn WHERE to spend (budget pacing under non-stationary difficulty).
Budget-augmented MDP == CMDP with a hard per-window constraint (no Lagrange multiplier to tune).
"""
import numpy as np, torch, torch.nn.functional as F
from .env import VecEnv, HUMAN
from .ppo_lag import ActorCritic, gae, save_policy
from .featurize import OBS_NAMES

EXTRA = 4
EXTRA_NAMES = ["rem_frac", "pos_frac", "pace", "rate_x10"]


def _names(d):
    n_extra_a = d - len(OBS_NAMES) - EXTRA            # 1 if the risk feature is stacked into the answer-stage state
    return OBS_NAMES + ["a_risk_oof"] * n_extra_a + EXTRA_NAMES


def stream_dim(T):
    return T.Q.shape[1] + 6 + T.R.shape[-1] + T.A.shape[-1] + EXTRA


def extra_feats(rem, K, t, W, budgets):
    rf = rem / K
    return np.stack([rf, np.full(len(K), t / W), rf - (1.0 - t / W), budgets * 10.0], 1).astype(np.float32)


class StreamRunner:
    def __init__(self, T, cfg, seed=0):
        self.T, self.cfg, self.env = T, cfg, VecEnv(T, cfg, seed)

    def run(self, windows, budgets, act_fn=None, ac=None, record=False, rscale=1.0, audit=None, qids=None, audit_win=0):
        windows = np.asarray(windows); B, W = windows.shape
        budgets = np.asarray(budgets, float)
        K = np.maximum(1, np.round(budgets * W)).astype(int); rem = K.copy(); env = self.env
        res = {k: np.zeros((B, W)) for k in ("net", "f1", "h", "s", "r")}
        buf = {k: [] for k in ("o", "m", "a", "lp", "v", "r", "act")} if record else None
        for t in range(W):
            obs = env.reset(windows[:, t]); sub = 0
            while not env.done.all() and sub < 8:
                env.human_ok[:] = rem > 0                       # HARD per-window budget governor
                mask, active, stage = env.mask(), ~env.done, env.stage.copy()
                o = np.concatenate([obs, extra_feats(rem, K, t, W, budgets)], 1)
                ot, mt = torch.as_tensor(o), torch.as_tensor(mask)
                if ac is not None:
                    with torch.no_grad():
                        d = ac.dist(ot, mt); a = d.sample(); lp = d.log_prob(a); v = ac.vr(ot).squeeze(-1)
                    a_np = a.numpy()
                else:
                    a_np = np.asarray(act_fn(o, mask))
                nobs, r, c, _ = env.step(a_np)
                rem = rem - c.astype(int)
                if audit is not None and active[audit_win]:
                    info = getattr(act_fn, "last", None)
                    audit.log(qid=qids[windows[audit_win, t]] if qids else str(windows[audit_win, t]), step=sub,
                              stage=["retrieval", "answer"][int(stage[audit_win])],
                              action=["advance", "re_retrieve", "specialist", "human"][int(a_np[audit_win])],
                              p_human=float(info["mean"][audit_win, HUMAN]) if info else None,
                              p_human_lo=float(info["lo"][audit_win, HUMAN]) if info else None,
                              p_human_hi=float(info["hi"][audit_win, HUMAN]) if info else None,
                              budget_used=int(K[audit_win] - rem[audit_win]), budget_cap=int(K[audit_win]),
                              reward=float(r[audit_win]), cost=float(c[audit_win]),
                              state={n: round(float(x), 4) for n, x in zip(_names(o.shape[1]), o[audit_win])})
                if record:
                    for k, x in zip(buf, [ot, mt, torch.as_tensor(a_np), lp, v,
                                          torch.as_tensor(r * rscale, dtype=torch.float32), torch.as_tensor(active)]):
                        buf[k].append(x)
                obs = nobs; sub += 1
            res["net"][:, t], res["f1"][:, t], res["h"][:, t] = env.net, env.final_f1, env.n_h
            res["s"][:, t], res["r"][:, t] = env.n_s, env.n_r
        res["K"] = K; res["rem_end"] = rem
        return res, buf


# ------------------------------------------------------------------ windows / streams (real queries, re-ordered)
def build_windows(rng, pool, B, W, attrs=None, p_drift=0.5):
    out = np.zeros((B, W), int)
    for i in range(B):
        w = rng.choice(pool, W, replace=False)
        if attrs and rng.random() < p_drift:                    # real-metadata drift: homogeneous blocks (type / level)
            cat = attrs[rng.integers(len(attrs))][w]
            groups = [w[cat == c] for c in np.unique(cat)]; rng.shuffle(groups); w = np.concatenate(groups)
        out[i] = w
    return out


def make_stream(kind, idx, f1_base, types, W, rng):
    idx = np.asarray(idx); n = (len(idx) // W) * W
    if kind == "iid":
        order = rng.permutation(idx)
    elif kind == "type_blocks":                                  # blocks of 100 same-type questions, random block order
        cats = types[idx]; groups = []
        for c in np.unique(cats):
            g = rng.permutation(idx[cats == c]); groups += [g[i:i + 100] for i in range(0, len(g), 100)]
        rng.shuffle(groups); order = np.concatenate(groups)
    elif kind == "hard_first":                                   # STRESS TEST: hardest (by base-pipeline F1) first
        order = idx[np.argsort(f1_base[idx] + rng.random(len(idx)) * 1e-3)]
    else:
        raise ValueError(kind)
    return order[:n].reshape(-1, W)


# ------------------------------------------------------------------ PPO training of the budget-aware policy
def _compact(buf, rscale=None):
    O, A, LP, V = (torch.stack(buf[k]) for k in ("o", "a", "lp", "v"))
    M = torch.stack(buf["m"]).to(torch.uint8)
    R = torch.stack(buf["r"]); ACT = torch.stack(buf["act"])
    order = torch.argsort((~ACT).to(torch.int8), dim=0, stable=True)     # active steps first, time order preserved
    def g(X):
        o = order.view(*order.shape, *([1] * (X.dim() - 2))).expand_as(X)
        return torch.gather(X, 0, o)
    O, M, A, LP, V, R, ACT = g(O), g(M).bool(), g(A), g(LP), g(V), g(R), g(ACT)
    return O, M, A, LP, V, R, ACT


def val_score(runner, ac, val_idx, cfg, seed=0):
    W = min(cfg.s_W, len(val_idx)); rng = np.random.default_rng(seed)
    wins = rng.permutation(np.asarray(val_idx))[: (len(val_idx) // W) * W].reshape(-1, W)
    bs = np.repeat([0.05, 0.10, 0.20], len(wins)); ws = np.tile(wins, (3, 1))
    res, _ = runner.run(ws, bs, ac=ac)
    return float(res["net"].mean())


def train_stream(T, train_idx, val_idx, attrs, cfg, seed=0, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    runner = StreamRunner(T, cfg, seed); D = stream_dim(T)
    ac = ActorCritic(D, 4, cfg.hidden); opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    W = min(cfg.s_W, len(train_idx)); rscale = 10.0 / W
    best, best_state = -1e9, None
    for it in range(cfg.s_iters):
        wins = build_windows(rng, np.asarray(train_idx), cfg.s_windows, W, attrs)
        lg = np.exp(rng.uniform(np.log(cfg.s_budget_lo), np.log(cfg.s_budget_hi), cfg.s_windows))
        un = rng.uniform(cfg.s_budget_lo, cfg.s_budget_hi, cfg.s_windows)              # better coverage of high budgets
        budgets = np.where(rng.random(cfg.s_windows) < 0.5, lg, un)
        res, buf = runner.run(wins, budgets, ac=ac, record=True, rscale=rscale)
        O, M, A, LP, V, R, ACT = _compact(buf)
        Adv, Ret = gae(R, V, ACT.float(), cfg.s_gae_lambda)
        sel = ACT
        o, m, a, lp0, adv, ret = O[sel], M[sel], A[sel], LP[sel], Adv[sel], Ret[sel]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8); n = len(a)
        for _ in range(cfg.s_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, cfg.s_minibatch):
                j = perm[s:s + cfg.s_minibatch]
                d = ac.dist(o[j], m[j]); ratio = (d.log_prob(a[j]) - lp0[j]).exp()
                pl = -torch.min(ratio * adv[j], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[j]).mean()
                vl = F.mse_loss(ac.vr(o[j]).squeeze(-1), ret[j])
                loss = pl + 0.5 * vl - cfg.ent_coef * d.entropy().mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        if it % cfg.s_val_every == cfg.s_val_every - 1 or it == cfg.s_iters - 1:
            vs = val_score(runner, ac, val_idx, cfg, seed)
            if vs > best: best, best_state = vs, {k: v.clone() for k, v in ac.state_dict().items()}
            log(f"[stream s={seed}] it {it+1:3d} train net {res['net'].mean():.3f} | val net {vs:.3f} (best {best:.3f})")
    ac.load_state_dict(best_state); return ac


def train_worker(args):
    """Process-pool entry point: one seed per process."""
    import json
    from pathlib import Path
    import numpy as np, torch
    torch.set_num_threads(1)
    cache, out, seed, cfg = args
    from .featurize import build_tables
    meta = json.load(open(Path(cache) / "meta.json")); idx = meta["idx"]
    T = build_tables(np.load(Path(cache) / "cache.npz"), idx["train"], meta["qids"])
    from .run_final import load_types
    if cfg.stack_risk:
        from .stacked import augment
        T = augment(T, np.load(Path(out) / "risk_feature.npy"))
    ty = load_types(Path(cache), meta)
    attrs = [ty["type_code"], ty["level_code"]] if ty else None
    ac = train_stream(T, idx["train"], idx["val"], attrs, cfg, seed)
    save_policy(Path(out) / f"stream_s{seed}.pt", ac, 0.0, -1.0, stream_dim(T))
    return seed
