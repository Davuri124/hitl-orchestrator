# Budget-Constrained RL for Human-in-the-Loop Agent Orchestration (final)

**Topic implemented:** multi-agent pipeline (retrieval -> summarise -> answer -> critic -> NLI verifier) executed for real on
HotpotQA; an orchestration agent that sees measured uncertainty signals (token logprob/entropy, critic P(Yes), NLI, retrieval
margin, cross-model agreement, sampled-answer agreement, embeddings) and chooses ADVANCE / RE_RETRIEVE / SPECIALIST / HUMAN
under a **hard budget of K human reviews per 1,000 queries**; audit trail of every transition (SQLite, optional Kafka).
No synthetic data anywhere. The human reviewer is an oracle (gold answer) unless `--human_acc` < 1.

## Two stages
1. GPU, once (~5-6 h on Kaggle T4): `python -m hitl.build_cache ...` -> `cache/cache.npz` (real outputs for every branch).
2. CPU, ~15-25 min: `python -m hitl.run_final --cache cache --out final` -> trains the budget-aware policy (3 seeds, parallel),
   evaluates on iid / type-blocked / hardest-first streams under a hard per-window cap against static and paced heuristics,
   writes `final/FINAL_REPORT.md`, CSVs, plot, `audit.db`.

Use `hitl_FINAL.ipynb` (Kaggle) - it only needs your existing `cache/` (no GPU).

## What is and is not claimed
- Offline replay of real pipeline runs, not live serving. A live-agent backend is not implemented or validated.
- Policy vs heuristics is reported as WIN/TIE/LOSS from paired-bootstrap CIs; the report generator never rewrites ties as wins.
- `hard_first` is a stress test (ordering by ground-truth difficulty), not a deployment estimate.
- Earlier per-query-average CMDP (PPO-Lagrangian) code remains in `train.py` / `evaluate.py` as an ablation.

## Risk-stacked variant (`--stack_risk`, version 1.1-stacked)
`python -m hitl.run_final --cache cache --out final_stacked --stack_risk --prev final/results_stream.csv`
Gives the RL policy an out-of-fold supervised risk score (per evidence/answerer cell) as an input feature, so it does not have
to learn risk ranking itself. Train queries use K-fold out-of-fold risk (no leakage); val/test use the model fit on all train.
Baselines are unchanged and use the same underlying risk models, so the comparison stays like-for-like.

## Confirmatory run on fresh queries (version 1.2-confirm)
`hitl_CONFIRM.ipynb`: (1) pre-register hypotheses + hashes of frozen policies/code, (2) build a cache for 2000 NEW dev queries
(shuffled positions 2000-4000; the main test used 0-2000), (3) evaluate with everything frozen. Primary test: PPO(stacked) > paced_wide_spec_gbm
on the iid stream at b in {0.02, 0.05, 0.10}, one-sided paired bootstrap, Holm-corrected, alpha 0.05. Everything else is exploratory.
The evaluation refuses to run if code or frozen artefacts changed after pre-registration.
