import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from .env import ADV, RERET, SPEC, HUMAN, build_obs
from .featurize import COL, A_FEATS


def _stageA(obs): return obs[:, COL["stage_A"]] > 0.5


def auto_only(obs, mask): return np.zeros(len(obs), int)


def always_specialist(obs, mask):
    a = np.zeros(len(obs), int); a[_stageA(obs) & mask[:, SPEC]] = SPEC; return a


def random_human(b, seed=0):
    rng = np.random.default_rng(seed)
    def f(obs, mask):
        a = np.zeros(len(obs), int); a[_stageA(obs) & mask[:, HUMAN] & (rng.random(len(obs)) < b)] = HUMAN; return a
    return f


def threshold_policy(feat, tau, low_is_risky=True):
    j = COL[f"a_{feat}"]
    def f(obs, mask):
        a = np.zeros(len(obs), int)
        risky = obs[:, j] < tau if low_is_risky else obs[:, j] > tau
        a[_stageA(obs) & risky & mask[:, HUMAN]] = HUMAN; return a
    return f


def calibrate_tau(T, val_idx, feat, b, low_is_risky=True):
    s = T.A[val_idx, 0, 0, A_FEATS.index(feat)]
    return float(np.quantile(s, b if low_is_risky else 1 - b))


class RiskGBM:
    """Strong supervised baseline: calibrated error-risk model on the same signals + budget-quantile threshold."""
    def __init__(self, T, train_idx, val_idx, b, rv=0, mv=0):
        tr = np.asarray(train_idx); n = len(tr); one = np.ones(n, int)
        X = build_obs(T, tr, one, rv * one, mv * one); y = (T.f1[tr, rv, mv] < 0.5).astype(int)
        self.m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0).fit(X, y)
        nv = len(val_idx); ov = np.ones(nv, int)
        self.tau = float(np.quantile(self.risk(build_obs(T, val_idx, ov, rv * ov, mv * ov)), 1 - b))

    def risk(self, obs): return self.m.predict_proba(obs)[:, 1]

    def __call__(self, obs, mask):
        a = np.zeros(len(obs), int)
        a[_stageA(obs) & (self.risk(obs) > self.tau) & mask[:, HUMAN]] = HUMAN; return a


class EnsemblePolicy:
    """Mean action distribution over K independently trained PPO-Lag policies; exposes min/max interval for audit.
    sample=False -> argmax (this UNDER-USES low budgets: a budget-feasible CMDP policy is stochastic, p(human) is
    typically < 0.5 everywhere, so argmax never picks human). sample=True -> act as trained."""
    def __init__(self, nets, sample=False, seed=0):
        self.nets, self.last, self.sample, self.rng = nets, None, sample, np.random.default_rng(seed)

    @torch.no_grad()
    def __call__(self, obs, mask):
        o, m = torch.as_tensor(obs), torch.as_tensor(mask)
        P = torch.stack([torch.softmax(n.pi(o).masked_fill(~m, -1e9), -1) for n in self.nets])
        mean = P.mean(0)
        self.last = dict(mean=mean.numpy(), lo=P.min(0).values.numpy(), hi=P.max(0).values.numpy())
        if not self.sample: return mean.argmax(-1).numpy()
        p = mean.numpy().astype(np.float64); p /= p.sum(1, keepdims=True)
        return np.array([self.rng.choice(4, p=row) for row in p])


@torch.no_grad()
def calibrate_human_tau(nets, T, idx, b, cfg):
    """Use the policy's p(HUMAN) as a learned risk score: threshold = (1-b) quantile of p(human) at the final
    answer-stage state on validation queries, following the policy's best non-human actions."""
    from .env import VecEnv
    env = VecEnv(T, cfg, 0); obs = env.reset(idx); ph = np.full(len(idx), np.nan)
    for _ in range(8):
        if env.done.all(): break
        o, m = torch.as_tensor(obs), torch.as_tensor(env.mask())
        P = torch.stack([torch.softmax(n.pi(o).masked_fill(~m, -1e9), -1) for n in nets]).mean(0).numpy()
        live = (env.stage == 1) & ~env.done
        ph[live] = P[live, HUMAN]
        nh = P.copy(); nh[:, HUMAN] = -1.0
        obs, _, _, _ = env.step(nh.argmax(1))
    return float(np.nanquantile(ph, 1 - b))


class ThresholdedPolicy:
    """PPO-Lag for re-retrieve/specialist decisions (argmax over non-human actions); human only when
    p(HUMAN) exceeds a validation-calibrated threshold at the answer stage."""
    def __init__(self, nets, tau): self.nets, self.tau, self.last = nets, tau, None

    @torch.no_grad()
    def __call__(self, obs, mask):
        o, m = torch.as_tensor(obs), torch.as_tensor(mask)
        P = torch.stack([torch.softmax(n.pi(o).masked_fill(~m, -1e9), -1) for n in self.nets]).mean(0).numpy()
        nh = P.copy(); nh[:, HUMAN] = -1.0; act = nh.argmax(1)
        stageA = obs[:, COL["stage_A"]] > 0.5
        act[stageA & mask[:, HUMAN] & (P[:, HUMAN] > self.tau)] = HUMAN
        return act


def oracle_hindsight(T, idx, cap, cfg):
    """Unachievable ceiling: best automated branch per query, humans spent on the largest hindsight gains."""
    idx = np.asarray(idx); f = T.f1[idx]
    auto = np.stack([f[:, 0, 0], f[:, 1, 0] - cfg.cost_reretrieve,
                     f[:, 0, 1] - cfg.cost_specialist, f[:, 1, 1] - cfg.cost_reretrieve - cfg.cost_specialist], 1).max(1)
    out = auto.copy(); top = np.argsort(-(1.0 - auto))[:cap]; out[top] = 1.0
    return out


def fixed_escalation(reretrieve=True, specialist=False):
    """No learning: always upgrade retrieval (and optionally the answerer), never call a human."""
    def f(obs, mask):
        a = np.zeros(len(obs), int); A = _stageA(obs)
        if reretrieve: a[~A & mask[:, RERET]] = RERET
        if specialist: a[A & mask[:, SPEC]] = SPEC
        return a
    return f


class HeuristicRisk:
    """Strong non-RL baseline with the SAME action set as the RL policy: always re-retrieve, (optionally) always
    call the specialist, then spend the human budget on the highest-risk answers via a supervised risk model
    trained on that exact post-escalation cell. If PPO-Lag cannot beat this, the RL adds nothing."""
    def __init__(self, T, train_idx, val_idx, b, specialist):
        self.spec = specialist
        self.g = RiskGBM(T, train_idx, val_idx, b, rv=1, mv=1 if specialist else 0)

    def __call__(self, obs, mask):
        a = np.zeros(len(obs), int); A = _stageA(obs)
        a[~A & mask[:, RERET]] = RERET
        if self.spec: a[A & mask[:, SPEC]] = SPEC
        ready = A & ~mask[:, SPEC] if self.spec else A
        a[ready & (self.g.risk(obs) > self.g.tau) & mask[:, HUMAN]] = HUMAN
        return a
