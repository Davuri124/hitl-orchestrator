"""cache.npz (raw real outputs) -> standardised state tables. PCA / z-scoring are fit on TRAIN queries only."""
import numpy as np
from sklearn.decomposition import PCA

NP = 8
A_FEATS = ["mean_lp", "min_lp", "mean_ent", "max_ent", "n_tok", "critic_yes", "nli_ent", "nli_con", "xgap", "xent", "sem_agree", "sem_ent", "xmodel_agree"]
R_FEATS = ["top1", "margin", "ent", "sel_mean", "sel_min", "k"]
OBS_NAMES = ([f"q_pc{j}" for j in range(NP)] + ["stage_R", "stage_A", "rv_narrow", "rv_wide", "rv_gold", "model_spec"]
             + [f"r_{n}" for n in R_FEATS] + [f"r_pc{j}" for j in range(NP)]
             + [f"a_{n}" for n in A_FEATS] + [f"a_pc{j}" for j in range(NP)])
COL = {n: i for i, n in enumerate(OBS_NAMES)}
OBS_DIM = len(OBS_NAMES)


class Tables:
    def __init__(self, f1, Q, R, A, qids=None):
        self.f1, self.Q, self.R, self.A, self.qids = f1, Q, R, A, qids


def _zfit(x, tr):
    flat = x[tr].reshape(-1, x.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6


def fit_transforms(cache, train_idx):
    """Fit PCA / z-score on TRAIN rows only. The returned dict can be re-applied to new (fresh) queries unchanged."""
    tr = np.asarray(train_idx); tf = {}
    for k in ("rfeat", "afeat"):
        tf[k] = _zfit(cache[k], tr)
    for k in ("qemb", "remb", "aemb"):
        x = cache[k]
        p = PCA(NP, random_state=0).fit(x[tr].reshape(-1, x.shape[-1]))
        y = p.transform(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], NP)
        tf[k] = (p, *_zfit(y, tr))
    return tf


def apply_transforms(cache, tf, qids=None):
    def pca(k):
        p, mu, sd = tf[k]; x = cache[k]
        y = p.transform(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], NP)
        return (y - mu) / sd
    z = lambda k: (cache[k] - tf[k][0]) / tf[k][1]
    Q = pca("qemb")
    R = np.concatenate([z("rfeat"), pca("remb")], -1)
    A = np.concatenate([z("afeat"), pca("aemb")], -1)
    return Tables(cache["f1"].astype(np.float32), Q.astype(np.float32), R.astype(np.float32), A.astype(np.float32), qids)


def build_tables(cache, train_idx, qids=None):
    return apply_transforms(cache, fit_transforms(cache, train_idx), qids)
