"""Vectorised replay environment over the cached REAL pipeline outputs.
Stage R (after retrieval): ADVANCE | RE_RETRIEVE (narrow->wide) | HUMAN (human supplies gold evidence)
Stage A (after summarise->answer->critic->verify): RELEASE | SPECIALIST (re-run with the 7B agent) | HUMAN (gold answer)
Reward = answer F1 - compute costs. Constraint cost = 1 per human review.
"""
import numpy as np
from .featurize import OBS_DIM

ADV, RERET, SPEC, HUMAN = 0, 1, 2, 3
R_STAGE, A_STAGE = 0, 1


def build_obs(T, idx, stage, rv, mv):
    idx, stage, rv, mv = map(np.asarray, (idx, stage, rv, mv))
    a_ok = (stage == A_STAGE)[:, None]
    return np.concatenate([T.Q[idx], np.eye(2)[stage], np.eye(3)[rv], mv[:, None].astype(np.float32),
                           T.R[idx, rv], T.A[idx, rv, mv] * a_ok], 1).astype(np.float32)


class VecEnv:
    def __init__(self, T, cfg, seed=0):
        self.T, self.cfg, self.rng = T, cfg, np.random.default_rng(seed)

    def reset(self, idx):
        self.idx = np.asarray(idx, dtype=int); B = len(self.idx)
        z = lambda dt=int: np.zeros(B, dt)
        self.stage, self.rv, self.mv = z(), z(), z()
        self.done, self.human_ok = z(bool), np.ones(B, bool)
        self.final_f1, self.net = z(float), z(float)
        self.n_h, self.n_s, self.n_r = z(), z(), z()
        return self.obs()

    def obs(self):
        return build_obs(self.T, self.idx, self.stage, self.rv, self.mv)

    def mask(self):
        isR = self.stage == R_STAGE
        m = np.zeros((len(self.idx), 4), bool)
        m[:, ADV] = True
        m[:, RERET] = isR & (self.rv == 0)
        m[:, SPEC] = ~isR & (self.mv == 0)
        m[:, HUMAN] = self.human_ok & np.where(isR, self.rv != 2, True)
        return m

    def step(self, a):
        a = np.asarray(a).copy(); B = len(a)
        a[~self.mask()[np.arange(B), a]] = ADV            # invalid -> advance
        act, isR = ~self.done, self.stage == R_STAGE
        isA = ~isR
        cell = self.T.f1[self.idx, self.rv, self.mv]
        succ = self.rng.random(B) < self.cfg.human_acc
        rew, cost = np.zeros(B), np.zeros(B)
        m = act & isR & (a == RERET); self.rv[m] = 1; rew[m] -= self.cfg.cost_reretrieve; self.n_r[m] += 1
        m = act & isR & (a == HUMAN); cost[m] = 1; self.rv[m & succ] = 2; self.n_h[m] += 1
        self.stage[act & isR & ((a == ADV) | (a == HUMAN))] = A_STAGE
        m = act & isA & (a == SPEC); self.mv[m] = 1; rew[m] -= self.cfg.cost_specialist; self.n_s[m] += 1
        m = act & isA & (a == ADV); rew[m] += cell[m]; self.final_f1[m] = cell[m]; self.done[m] = True
        m = act & isA & (a == HUMAN)
        f = np.where(succ, 1.0, cell); cost[m] = 1; self.n_h[m] += 1
        rew[m] += f[m]; self.final_f1[m] = f[m]; self.done[m] = True
        self.net += rew
        return self.obs(), rew, cost, self.done.copy()
