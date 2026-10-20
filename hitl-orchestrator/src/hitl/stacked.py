"""Risk-stacked state: a supervised error-risk score (GBM on the same signals) is given to the RL policy as an input feature,
so PPO does not have to learn risk RANKING from scratch and can focus on pacing / escalation choices.
Leakage control: train queries get OUT-OF-FOLD risk (K-fold); val/test queries get the model fit on all train queries
(exactly what deployment would see)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from .env import build_obs
from .featurize import Tables


def _fit(X, y):
    if len(np.unique(y)) < 2:
        return float(y.mean()), None
    return None, HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0).fit(X, y)


def _pred(const, model, X):
    return np.full(len(X), const, np.float32) if model is None else model.predict_proba(X)[:, 1]


def risk_table(T, train_idx, n_folds=5, seed=0, log=print):
    """risk[i, rv, mv] = P(answer wrong) for every query and every (evidence variant, answerer) cell."""
    N = T.f1.shape[0]; tr = np.asarray(train_idx); allq = np.arange(N); others = np.setdiff1d(allq, tr)
    out = np.zeros((N, 3, 2), np.float32); one = np.ones(N, int)
    for rv in range(3):
        for mv in range(2):
            X = build_obs(T, allq, one, rv * one, mv * one); y = (T.f1[:, rv, mv] < 0.5).astype(int)
            c, m = _fit(X[tr], y[tr]); out[others, rv, mv] = _pred(c, m, X[others])
            for a, b in KFold(n_folds, shuffle=True, random_state=seed).split(tr):
                c, m = _fit(X[tr[a]], y[tr[a]]); out[tr[b], rv, mv] = _pred(c, m, X[tr[b]])
            log(f"  risk model cell (evidence={rv}, model={mv}) done", flush=True)
    return out


def _logit(risk):
    r = np.clip(risk, 1e-4, 1 - 1e-4)
    return np.log(r / (1 - r))


def risk_stats(risk, train_idx):
    z = _logit(risk)[np.asarray(train_idx)]
    return float(z.mean()), float(z.std() + 1e-6)


def to_feature_with(risk, mu, sd):
    return ((_logit(risk) - mu) / sd).astype(np.float32)


def to_feature(risk, train_idx):
    return to_feature_with(risk, *risk_stats(risk, train_idx))


def fit_models(T, train_idx):
    """One GBM per (evidence, answerer) cell, fit on ALL train queries. Used to score fresh queries (frozen)."""
    tr = np.asarray(train_idx); one = np.ones(len(tr), int); models = {}
    for rv in range(3):
        for mv in range(2):
            models[(rv, mv)] = _fit(build_obs(T, tr, one, rv * one, mv * one), (T.f1[tr, rv, mv] < 0.5).astype(int))
    return models


def predict_models(models, T):
    N = T.f1.shape[0]; allq = np.arange(N); one = np.ones(N, int); out = np.zeros((N, 3, 2), np.float32)
    for (rv, mv), (c, m) in models.items():
        out[:, rv, mv] = _pred(c, m, build_obs(T, allq, one, rv * one, mv * one))
    return out


def augment(T, feat):
    """Append the risk feature as one extra channel of the answer-stage state (zeroed automatically at the retrieval stage)."""
    return Tables(T.f1, T.Q, T.R, np.concatenate([T.A, feat[..., None]], -1).astype(np.float32), T.qids)
