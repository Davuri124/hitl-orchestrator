"""PPO-Lagrangian for the budget-constrained MDP:  max E[R]  s.t.  E[#human reviews per query] <= b.
Plain PyTorch (no Ray/RLlib): two critics (reward, cost), masked categorical actor, dual ascent on lambda."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .env import VecEnv


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, d, na=4, h=128):
        super().__init__()
        self.pi, self.vr, self.vc = mlp(d, na, h), mlp(d, 1, h), mlp(d, 1, h)

    def dist(self, obs, mask):
        return torch.distributions.Categorical(logits=self.pi(obs).masked_fill(~mask, -1e9))


def collect(env, ac, idx, T_max=8):
    obs = env.reset(idx)
    buf = {k: [] for k in ["o", "m", "a", "lp", "vr", "vc", "r", "c", "act"]}
    for _ in range(T_max):
        mask, active = env.mask(), ~env.done
        o, m = torch.as_tensor(obs), torch.as_tensor(mask)
        with torch.no_grad():
            d = ac.dist(o, m); a = d.sample()
            lp, vr, vc = d.log_prob(a), ac.vr(o).squeeze(-1), ac.vc(o).squeeze(-1)
        obs, r, c, _ = env.step(a.numpy())
        for k, v in zip(buf, [o, m, a, lp, vr, vc, torch.as_tensor(r, dtype=torch.float32),
                              torch.as_tensor(c, dtype=torch.float32), torch.as_tensor(active)]):
            buf[k].append(v)
        if env.done.all(): break
    assert env.done.all(), "episode did not terminate"
    return {k: torch.stack(v) for k, v in buf.items()}


def gae(rew, val, act, lam):
    T, B = rew.shape; adv = torch.zeros_like(rew); last = torch.zeros(B)
    for t in reversed(range(T)):
        nxt = act[t + 1] if t + 1 < T else torch.zeros(B)
        nv = val[t + 1] * nxt if t + 1 < T else torch.zeros(B)
        last = rew[t] + nv - val[t] + lam * nxt * last
        adv[t] = last
    return adv * act, (adv + val) * act


@torch.no_grad()
def greedy_eval(T, idx, ac, cfg):
    env = VecEnv(T, cfg, 0); obs = env.reset(idx)
    for _ in range(8):
        if env.done.all(): break
        a = ac.dist(torch.as_tensor(obs), torch.as_tensor(env.mask())).probs.argmax(-1).numpy()
        obs, _, _, _ = env.step(a)
    return float(env.net.mean()), float(env.n_h.mean())


def train_ppo_lag(T, train_idx, budget, cfg, seed=0, log=print, val_idx=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env = VecEnv(T, cfg, seed)
    D = T.Q.shape[1] + 6 + T.R.shape[-1] + T.A.shape[-1]
    ac = ActorCritic(D, 4, cfg.hidden); opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    lam, hist = cfg.lam_init, []
    best_score, best_state = -1e9, None
    for it in range(cfg.iters):
        b = collect(env, ac, rng.choice(train_idx, cfg.episodes_per_iter))
        act = b["act"].float()
        Ar, Rr = gae(b["r"], b["vr"], act, cfg.gae_lambda)
        Ac, Rc = gae(b["c"], b["vc"], act, cfg.gae_lambda)
        sel = act.bool()
        o, m, a, lp0 = b["o"][sel], b["m"][sel], b["a"][sel], b["lp"][sel]
        adv = ((Ar - lam * Ac) / (1 + lam))[sel]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        Rr, Rc, n = Rr[sel], Rc[sel], len(a)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, cfg.minibatch):
                j = perm[s:s + cfg.minibatch]
                d = ac.dist(o[j], m[j]); ratio = (d.log_prob(a[j]) - lp0[j]).exp()
                pl = -torch.min(ratio * adv[j], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[j]).mean()
                vl = F.mse_loss(ac.vr(o[j]).squeeze(-1), Rr[j]) + F.mse_loss(ac.vc(o[j]).squeeze(-1), Rc[j])
                loss = pl + 0.5 * vl - cfg.ent_coef * d.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        Jc = float(b["c"].sum(0).mean())
        lam = float(np.clip(lam + cfg.dual_lr * (Jc - budget), 0.0, cfg.lam_max))     # dual ascent
        hist.append(dict(it=it, ret=float(b["r"].sum(0).mean()), human_rate=Jc, lam=lam))
        if val_idx is not None and it % 10 == 9:
            vr, vh = greedy_eval(T, val_idx, ac, cfg)
            score = vr - 10.0 * max(0.0, vh - budget - 0.01)        # infeasible policies are penalised
            hist[-1].update(val_ret=vr, val_human=vh)
            if score > best_score:
                best_score, best_state = score, {k: v.clone() for k, v in ac.state_dict().items()}
        if it % 25 == 0 or it == cfg.iters - 1:
            extra = f" | val {hist[-1].get('val_ret', float('nan')):.3f} (human {hist[-1].get('val_human', float('nan')):.3f})" if val_idx is not None else ""
            log(f"[b={budget:.2f} s={seed}] it {it:3d} train return {hist[-1]['ret']:.3f} human/query {Jc:.3f} lambda {lam:.3f}{extra}")
    if best_state is not None: ac.load_state_dict(best_state)
    return ac, lam, hist


def save_policy(path, ac, lam, budget, D):
    torch.save({"state": ac.state_dict(), "lam": lam, "budget": budget, "D": D}, path)


def load_policy(path, hidden=128):
    ck = torch.load(path, map_location="cpu"); ac = ActorCritic(ck["D"], 4, hidden)
    ac.load_state_dict(ck["state"]); ac.eval(); return ac
