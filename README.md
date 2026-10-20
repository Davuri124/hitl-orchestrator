# HITL Orchestrator

**Budget-constrained reinforcement learning for human-in-the-loop agent orchestration.**

An orchestration agent decides, step by step, whether to **advance automatically**, **re-retrieve with modified constraints**, **route to a stronger specialist model**, or **interrupt for human review**, while strictly staying inside a human-review budget of **K reviews per 1,000 queries**. Every decision is written to an audit trail.

> **No synthetic data.** The system runs a real multi-agent pipeline on HotpotQA (distractor setting) with real questions, real gold answers and real gold supporting paragraphs. Policies are trained and evaluated by replaying cached real outputs of every action the orchestrator could take.

**Status:** code version `1.2-confirm` | main study complete | confirmatory evaluation on fresh queries **prepared but not yet completed** (see [Confirmatory evaluation](#confirmatory-evaluation))

---

## Table of contents

1. [Motivation](#motivation)
2. [Headline findings](#headline-findings)
3. [System design](#system-design)
4. [Learning formulations](#learning-formulations)
5. [Baselines and evaluation protocol](#baselines-and-evaluation-protocol)
6. [Results](#results)
7. [What we learned](#what-we-learned)
8. [What can and cannot be claimed](#what-can-and-cannot-be-claimed)
9. [Limitations](#limitations)
10. [Confirmatory evaluation](#confirmatory-evaluation)
11. [Reproducibility and artefacts](#reproducibility-and-artefacts)
12. [Engineering notes](#engineering-notes)
13. [Hyperparameters](#hyperparameters)
14. [Version history](#version-history)
15. [Author and citation](#author-and-citation)

---

## Motivation

Enterprise workflows chain several LLM agents (retrieval, summarisation, critique, verification). Two failure modes dominate:

- **Full autonomy** lets a wrong intermediate step propagate into a confident, wrong final output.
- **Reviewing too many steps** overwhelms human reviewers and defeats the purpose of automation.

Human attention is a hard, scarce resource: the realistic constraint is *at most K reviews per 1,000 queries*. A fixed threshold on an LLM's self-reported confidence fails because verbalised confidence is badly calibrated and the right threshold depends on the budget and on how difficult the current stretch of queries is.

**Research question:** can a learned orchestration policy, trained on real pipeline behaviour, allocate a limited human-review budget (and cheaper automatic escalations) better than strong non-learned rules?

This project follows the earlier RLHIL framework (a budget-constrained CMDP for human-in-the-loop oversight evaluated on simulated environments). Here the requirement is stricter: no synthetic data.

| Consequence of "no synthetic data" | How it is handled |
|---|---|
| Uncertainty must be measured, not simulated | Run real LLM agents; read token log-probabilities, critic Yes/No probabilities, NLI scores, retrieval scores, sampled-answer agreement |
| Correctness needs ground truth | HotpotQA gold answers and gold supporting paragraphs |
| Actions not taken have unknown outcomes | **Cached-counterfactual replay:** execute every branch once (3 evidence variants x 2 answerers = 6 cells per query) |
| Humans cannot be simulated with invented behaviour | The reviewer is an oracle supplying the gold answer/evidence (the single explicit modelling assumption; `human_acc` is a sensitivity switch) |
| Easy baselines would inflate results | Compare against strong, matched-action-set heuristics, not only "never/always intervene" |

---

## Headline findings

- **Most benefit comes from cheap automatic upgrades and from spending human reviews on the highest-risk answers.** Wide re-retrieval adds **+0.080** net reward over the cheap default, the 7B specialist **+0.019** alone (**+0.083** together), and human reviews add **+0.016** (b=0.02) to **+0.195** (b=0.30) on top.
- **A plain PPO policy does not clearly beat a strong non-RL baseline** (always re-retrieve, supervised risk model, budget pacing) with the same actions and the same hard cap: **0 wins, 12 ties, 3 losses** in 15 comparisons.
- **Giving the policy a supervised risk score helps.** The risk-stacked policy improved over the plain stream policy in **all 15 cells** (+0.005 to +0.021 net reward) and scored **4 wins, 10 ties, 1 loss** against the best heuristic. All 12 cells with budget up to 0.20 had a positive point estimate (+0.001 to +0.017); the 3 cells at b=0.30 were negative (-0.004 to -0.014).
- **Error-prediction signals are the real bottleneck.** The best single signal reaches about 0.68 AUROC, the combined supervised model about 0.77, and the hindsight oracle is **0.15 to 0.23** net reward above the learned policy.
- **Budget compliance was perfect** in all stream evaluations (0 windows over budget).

### Honest reading

The stacked-policy gains are small (about 0.5 to 1.7 points of net reward), individually borderline (lower confidence bounds +0.0007 to +0.0042), and were found after design choices were guided by test-set numbers. A pre-registered confirmatory evaluation on 2,000 fresh queries is therefore prepared; **its cache is built but the evaluation has not been completed**. Until it is, the correct claim is *"consistent small improvement, not yet independently confirmed"*, not *"RL beats heuristics"*.

---

## System design

```
Query (HotpotQA question + 10 paragraphs)
  -> Retrieval agent (narrow k=2 / wide k=5 2-hop / gold = human)
  -> DECISION d0: ADVANCE | RE_RETRIEVE | HUMAN (evidence)
  -> Summariser (Qwen2.5-3B)
  -> Answerer (base 3B or specialist 7B)
  -> Critic (P(Yes)) + NLI verifier + 5 sampled answers
  -> DECISION d1: RELEASE | SPECIALIST | HUMAN (answer)

Orchestration agent (budget-aware PPO policy, risk-stacked)  <->  Budget governor (hard cap of K reviews per 1,000 queries)
Audit trail: every transition -> SQLite (+ optional Kafka topic)
```

### Data and splits

HotpotQA distractor: each question has 10 paragraphs (2 gold, 8 distractors), a gold answer and gold supporting facts. Fixed seed 42.

| Split | Size | Source |
|---|---|---|
| Train | 3,000 | official train split |
| Validation | 1,000 | official train split (checkpoint selection, threshold calibration) |
| Test | 2,000 | official dev split, shuffled positions 0 to 2000 (final comparisons) |
| Confirmatory | 2,000 | official dev split, shuffled positions 2000 to 4000 (disjoint, checked by id) |

### Agents and models

| Agent | Implementation | Output used by the orchestrator |
|---|---|---|
| Retrieval | `bge-small-en-v1.5` dense scores. **narrow:** top-2. **wide** (the "re-retrieve with modified constraints" action): top-5 with a second-hop query expanded by the top paragraph. **gold:** what a human would supply | Top-1 score, margin, softmax entropy over the 10 paragraphs, mean/min selected score, k, mean paragraph embedding |
| Summariser | Qwen2.5-3B-Instruct, at most 5 factual bullets | Notes passed to the answerer |
| Answerer | Base: Qwen2.5-3B-Instruct. Specialist: Qwen2.5-7B-Instruct-AWQ ("route to specialist"). Greedy decoding plus 5 sampled answers (T=0.7) | Answer, token log-prob / entropy statistics, sampled-answer agreement |
| Critic | Qwen2.5-3B-Instruct, one-word Yes/No verdict; confidence read from next-token probabilities (not self-reported text) | P(Yes) |
| Verifier | DeBERTa-v3 NLI (premise: evidence, hypothesis: question + answer) | P(entailment), P(contradiction) |

The pipeline is served with vLLM on a Kaggle T4; policy training and evaluation run on CPU.

### Measured uncertainty signals (the orchestrator's state)

| Signal | Meaning |
|---|---|
| mean/min token log-prob, mean/max token entropy, answer length | Answerer's own token-level uncertainty (top-20 entropy approximation) |
| critic P(Yes), NLI entailment/contradiction | Independent judgement that the answer is supported by the evidence |
| `xgap`, `xent` | Critic-vs-NLI disagreement and entropy (cross-agent) |
| `sem_agree`, `sem_ent` | Mean F1 of 5 sampled answers vs the greedy answer; entropy over answer clusters (semantic-entropy style) |
| `xmodel_agree` | F1 between base and specialist answer; observable only after the specialist has been paid for |
| PCA embeddings (8-d each) | Question, retrieved evidence, answer |
| supervised risk score (stacked variant) | Out-of-fold GBM probability that the answer is wrong, per evidence/answerer cell |
| budget state (stream variants) | Remaining budget fraction, position in the window, pace (remaining minus remaining queries), target rate |

### Decision process, actions and reward

- **Stage R** (after retrieval): `ADVANCE`, `RE_RETRIEVE` (narrow to wide; once), `HUMAN` (human supplies gold evidence; pipeline continues).
- **Stage A** (after summarise, answer, critique, verify): `RELEASE`, `SPECIALIST` (re-run with the 7B answerer; once), `HUMAN` (human supplies the gold answer; episode ends). After a specialist re-run there is one more release/human decision.
- **Reward** = answer F1 minus compute costs (re-retrieve 0.02, specialist 0.05; both are assumptions and were swept). A human review yields F1 = 1.0 (oracle).
- **Constraint:** each `HUMAN` action costs one review. Per-query form: expected reviews per query at most *b*. Stream form: at most `K = round(b x W)` reviews per window of *W* queries, enforced by masking `HUMAN` once the budget is exhausted.

### Cached-counterfactual replay

For every query, all six cells (narrow/wide/gold evidence x base/specialist answerer) are computed with all signals. The environment simply looks up the outcome of whichever path the policy chooses. Everything stays real (real model outputs, uncertainty and correctness) while RL training takes minutes instead of GPU-days.

*Caveat:* re-running the GPU stages gives slightly different cells (batched greedy decoding in vLLM is not bit-reproducible; differences of about 0.002 to 0.005 in net reward), so **one cache file is treated as the dataset**.

### Audit trail

Every transition is written to a SQLite table: timestamp, run id, query id, step, stage, action, the policy's mean probability of choosing `HUMAN` with a min/max interval across ensemble members, budget used and cap, reward, cost, and the full state vector. An optional Kafka producer mirrors rows to a topic. *Live Kafka streaming was implemented but not exercised in these experiments.*

---

## Learning formulations

### 1. Per-query constrained MDP: PPO-Lagrangian

Maximise expected net reward subject to expected human reviews per query at most *b*. Plain PyTorch (no Ray/RLlib): masked categorical actor, two critics (reward, cost), advantage `A_r - lambda * A_c`, dual ascent on lambda using the batch's mean cost. One policy per budget (0.02, 0.05, 0.10, 0.20, 0.30), 5 seeds, ensemble by mean action probability. Validation-based checkpoint selection (net reward penalised if the review rate exceeds the budget) was added after the first small run showed overfitting.

### 2. Stream-level hard budget (the topic as stated)

"Per 1,000 queries" is a window constraint, not a per-query average. A budget-augmented MDP processes a window of *W* queries sequentially; the state gains four features (remaining budget fraction, position in window, pace, target rate); `HUMAN` is masked when no budget remains, so **the constraint holds by construction** and no Lagrange multiplier is needed.

- One policy is trained for all budgets: budgets are sampled per window in [0.02, 0.30] (log-uniform in v1.0; half log-uniform, half uniform from v1.1, for better coverage of high budgets).
- Training windows use *W* = 200 (features are fractions, so the policy transfers to *W* = 1,000).
- Half of the training windows are re-ordered into homogeneous blocks by real metadata (question type or difficulty level) to expose the policy to drift.
- PPO with GAE (lambda 0.98), 3 seeds trained in parallel, validation-based checkpoint selection.

### 3. Risk-stacked state

The plain stream policy had to learn risk ranking from sparse rewards, which is exactly what a supervised model does well. In the stacked variant a gradient-boosted classifier (one per evidence/answerer cell) predicts *P(answer wrong)* from the same signals, and its z-scored logit is appended to the state. To avoid leakage, **training queries receive 5-fold out-of-fold predictions**; validation, test and fresh queries receive the model fit on all training queries, exactly as at deployment. The policy then only needs to learn pacing and escalation choices on top of calibrated risk.

### 4. Decoding

A budget-feasible stochastic policy usually has *p(HUMAN)* below 0.5 everywhere, so argmax decoding never chooses a human at low budgets (observed human rate of exactly 0.000 at b=0.02 and 0.05). The primary evaluation therefore **samples from the ensemble-mean action distribution**, as trained, fixed before the main and confirmatory runs. Argmax and a calibrated *p(HUMAN)*-threshold decoding are kept as diagnostics.

---

## Baselines and evaluation protocol

| Baseline | Description |
|---|---|
| `auto_only` | Cheap default (narrow evidence, base answerer), no humans |
| `always_specialist`, `wide_only`, `wide_spec_no_human` | Fixed automatic upgrades, no humans |
| `random_human`, `conf_logprob`, `critic_pyes`, `conf_semantic` | Spend the budget on random queries or on the lowest single-signal confidence (threshold calibrated on validation to the budget) |
| `risk_gbm` | Supervised risk model + budget-quantile threshold on the cheap default path (human-only action set) |
| `wide_gbm`, `wide_spec_gbm` | **Same action set as the policy:** always re-retrieve (and always call the specialist), then spend humans on the highest GBM risk |
| static / paced variants (stream) | Static: fixed validation-calibrated threshold. Paced: threshold re-computed from remaining budget / remaining queries (online budget pacing) |
| `oracle_hindsight` | Unachievable ceiling: best automatic path per query; humans spent on the largest hindsight gains |

**Metrics:** net reward (F1 minus compute cost; a human review counts as F1 = 1), F1, human/specialist/re-retrieve rates, budget violations.

**Statistics:** paired bootstrap over test queries (per-query values averaged over stream orderings), 95% CIs. Verdict is WIN/LOSS only if the CI excludes zero, otherwise TIE.

**Streams** (all real queries, only re-ordered): **iid**; **type-blocks** (blocks of 100 same-type questions); **hard-first** (ordered by ground-truth difficulty; a stress test, not a deployment estimate). The hard cap is identical for every method. Evaluation windows use *W* = 1,000 (2 per ordering on 2,000 queries), 3 random orderings for iid and type-blocks.

---

## Results

Results are listed in the order they were obtained, because early mistakes shaped the later design.

### 1. Small validation run (300 train / 100 val / 200 test)

Purpose: validate the whole chain before spending GPU hours (about 35 to 40 minutes). Mean F1 per evidence variant: narrow 0.52, wide 0.63, gold 0.67 (base model); the 7B specialist adds roughly 0.06 to 0.11. So there is genuine headroom for deferral. Findings:

- PPO-Lagrangian initially appeared to win, but it was compared with human-only baselines. It re-retrieved on 54 to 57% of queries, and re-retrieval alone gives about +0.11 F1 at tiny cost, which a trivial "always re-retrieve" rule also gets. **The comparison was not like-for-like.**
- Training return (about 0.96) far exceeded test net reward (0.56 to 0.69): overfitting on 300 queries. Validation-based checkpoint selection was added.
- With 200 test queries the CI half-width is about +/-0.05; no small difference could be resolved.

### 2. Matched baselines and added signals

Adding `wide_only` / `wide_spec` / `wide_gbm` / `wide_spec_gbm` (same action set as the policy) removed the apparent advantage: differences were within about +/-0.02 with all CIs spanning zero. Sampling-based semantic signals did not help. Argmax decoding left low budgets unused and produced a significant loss to `wide_gbm` at b=0.05 (-0.0485); sampling decoding spent the budget and removed the loss.

### 3. Full run (3,000 / 1,000 / 2,000), per-query average budget

Net reward on 2,000 test queries (PPO-Lagrangian, sampled decoding, 5-seed ensemble):

| Method | b=0.02 | b=0.05 | b=0.10 | b=0.20 | b=0.30 |
|---|---|---|---|---|---|
| `wide_gbm` | 0.5552 | 0.5809 | 0.6165 | 0.6787 | 0.7343 |
| `wide_spec_gbm` | 0.5618 | 0.5880 | 0.6273 | 0.6850 | 0.7269 |
| PPO-Lagrangian (sampled) | 0.5712 | 0.5954 | 0.6336 | 0.6899 | 0.7343 |
| oracle (hindsight) | 0.7296 | 0.7596 | 0.8096 | 0.9086 | 0.9649 |

Without humans: `auto_only` 0.4594, `always_specialist` 0.4779, `wide_only` 0.5394, wide + specialist 0.5428.

- **PPO vs `wide_gbm`:** +0.011 to +0.017 at b up to 0.10 with CIs excluding zero (lower bounds about +0.001 to +0.004); a tie at b=0.20 (+0.011) and b=0.30 (0.000).
- **PPO vs `wide_spec_gbm`:** +0.005 to +0.010, every CI spans zero. No significant win over the stronger heuristic. PPO's F1 is lower (0.606 vs 0.632 at b=0.02); it wins on net reward by calling the specialist on about 36 to 39% of queries instead of all.
- **Gain decomposition:** re-retrieval +0.080, specialist +0.019, both +0.083 over `auto_only`; humans add +0.016 (b=0.02) up to +0.195 (b=0.30) over `wide_only`.
- **Multiplicity:** about 30 CIs were computed; the b=0.05 lower bound (+0.0012) would not survive correction.

### 4. Signal audit and headroom

| Signal / model | AUROC for predicting a wrong answer |
|---|---|
| token entropy (best single signal) | about 0.68 (0.713 in the small run) |
| supervised GBM on all signals | about 0.77 (0.74 to 0.78 in the post-escalation cell) |
| cross-model agreement (`xmodel_agree`), post-escalation cell | 0.652 (0.612 in the small run) |
| sampled-answer agreement / entropy | 0.58 to 0.59 |
| critic P(Yes) | about 0.62 (small run) |
| critic-vs-NLI gap / entropy | about 0.53, i.e. chance (small run) |

The oracle is **0.158 to 0.231** net reward above the learned policy at every budget. No policy can close that gap without better error-prediction signals (a stronger verifier, larger judge models, or calibrated ensembles).

### 5. Specialist-cost sensitivity

The specialist cost (0.05) is an assumption, so policies were retrained at other costs. At cost 0.0 and 0.02, always calling the specialist is nearly optimal: PPO was above `wide_gbm` by +0.015 to +0.052 but below `wide_spec_gbm` by 0.005 to 0.014 (several CIs excluding zero in the negative direction). At cost 0.05 everything ties. (The cost-0.10 block was lost to notebook output truncation and was not analysed.) **The ranking depends on the cost assumption**; the policy roughly tracks the better of the two fixed rules.

### 6. Stream-level hard-budget policy (v1.0-final)

15 cells (3 streams x 5 budgets) vs the best of the static and paced heuristics. **0 windows over budget.** Result: **0 wins, 12 ties, 3 losses.** The losses were at b=0.30 on iid (-0.032), at b=0.30 on type-blocks (-0.035), and at b=0.20 on type-blocks (-0.015). At budgets up to 0.10 the policy matched the paced heuristic (within +/-0.01). Budget pacing mattered less than expected: on the hard-first stress stream the paced heuristic beat the static one by only +0.005 to +0.012.

### 7. Risk-stacked stream policy (v1.1-stacked)

**4 wins, 10 ties, 1 loss** over 15 cells against the best heuristic.

| Cell (stream, budget) | PPO minus comparator (net reward) | 95% CI | Verdict |
|---|---|---|---|
| iid, b=0.02 | +0.0126 | [+0.0007, +0.0247] | WIN |
| hard-first, b=0.02 | +0.0170 | [+0.0042, +0.0305] | WIN |
| hard-first, b=0.10 | +0.0156 | [+0.0017, +0.0299] | WIN |
| type-blocks, b=0.02 | +0.0133 | [+0.0018, +0.0248] | WIN |
| type-blocks, b=0.30 | -0.0139 | (excludes 0) | LOSS |

- Stacking raised net reward over the plain stream policy in **all 15 cells**, by +0.005 to +0.021.
- Only 4 of 15 CIs exclude zero and their lower bounds are close to zero. With correction for 15 comparisons, at most one or two would survive. Cells are not independent (same test queries and policies), so a sign test would be misleading.
- The likely mechanism (untested): the policy uses per-cell risk to decide when the specialist or re-retrieval is worth its cost, and uses the budget state to pace reviews.

### Summary of the study

| Stage | Question | Outcome |
|---|---|---|
| Small run | Does the chain work; is there headroom? | Yes. Re-retrieval explains most early gains; overfitting and under-powered test set |
| Matched baselines | Does RL beat heuristics with the same actions? | No (ties); argmax decoding under-spent the budget |
| Full run, per-query CMDP | Same, at 2,000 test queries | Small gain over `wide_gbm` at b up to 0.10; tie with `wide_spec_gbm`; cost-dependent |
| Stream, hard cap (1.0-final) | Does budget-aware RL beat paced heuristics? | 0 wins / 12 ties / 3 losses |
| Stream, risk-stacked (1.1-stacked) | Does a risk feature help the policy? | 4 wins / 10 ties / 1 loss; +0.005 to +0.021 over unstacked in all cells |
| Confirmatory (1.2-confirm) | Does it replicate on fresh queries? | Prepared, not completed |

---

## What we learned

- **The problem is nearly one-shot.** Each query involves two to four decisions and a terminal reward. When the best action is mostly a function of the final-state features, a supervised risk model plus fixed escalation is close to optimal. RL should matter more in longer-horizon settings (critique-and-revise loops, drifting streams with a shared budget).
- **Information, not optimisation, limits the result.** AUROC of 0.68 to 0.77 caps what any deferral rule can achieve; the oracle gap (0.16 to 0.23) is an order of magnitude larger than any policy difference measured.
- **Most reward comes from cheap automatic actions.** Wide re-retrieval (+0.080) dwarfs everything a policy adds on top, which is why early comparisons against human-only baselines were misleading.
- **Cost assumptions drive rankings.** The value of the specialist is almost exactly its assumed cost at 0.05.
- **Evaluation details matter as much as algorithms.** Argmax decoding silently left the budget unused; matched-action-set baselines changed the conclusion; the per-query average constraint was a relaxation of the "per 1,000 queries" requirement.
- **Stacking separates concerns.** Supervised learning ranks risk well; RL decides pacing and escalation. The learned policy is best described as a decision layer on top of a calibrated risk model, not a replacement for it.

---

## What can and cannot be claimed

| Supportable now | Not supportable |
|---|---|
| A real-execution, no-synthetic-data replay environment for budgeted human-in-the-loop orchestration with ground truth at every step, a hard budget per 1,000 queries (0 violations) and a full audit trail | "RL beats strong heuristics." Against a plain policy the result is parity; the stacked policy shows small, individually borderline gains not yet independently confirmed |
| A decomposition of where reward comes from (re-retrieval, specialist, human reviews) and an AUROC audit of the signals | "Production-ready." All results are offline replay; live agent serving and live Kafka streaming were not validated |
| Supervised risk + fixed escalation + budget pacing matches a plain RL policy; stacking the risk score into the RL state gives a small, consistent improvement at budgets up to 0.20 | Generalisation to legal or enterprise documents, other model families, or real human reviewers |

---

## Limitations

- **Researcher degrees of freedom.** Decoding, baselines and the stacking idea were chosen after looking at test results; the reported improvement may be optimistic. This is why the fresh-split confirmation exists.
- **Multiple comparisons.** 15 CIs per policy variant, no correction in the main report.
- **Single dataset and models.** HotpotQA (Wikipedia multi-hop QA), Qwen2.5-3B/7B only.
- **Oracle human.** The reviewer always supplies the gold answer; the `human_acc` switch was not swept in the main results.
- **Assumed costs.** Re-retrieve 0.02 and specialist 0.05 are our choices.
- **Seed variance.** Policies are 3-seed ensembles; per-seed variance is reported only in the confirmatory protocol.
- **Replay, not live.** No live-agent serving; cache rebuilds differ slightly (about 0.002 to 0.005).
- **Stress stream.** Hard-first ordering uses ground-truth difficulty and is not a deployment estimate.

---

## Confirmatory evaluation

**Design.** 2,000 new questions from the official dev split (shuffled positions 2000 to 4000; the main test used 0 to 2000), disjoint from all earlier queries (checked by id). Everything is frozen: the three stacked policies, the risk models (refit deterministically from the main training set and verified against the saved training feature), PCA/z-score transforms, validation-derived thresholds, sampling decoding and the comparator.

**Pre-registered primary hypotheses (3).** On the iid stream (*W* = 1,000, 3 orderings) at b = 0.02, 0.05, 0.10: the mean per-query net reward of the stacked policy is greater than that of `paced_wide_spec_gbm`. One-sided paired bootstrap (10,000 resamples), Holm correction over the 3 tests, alpha = 0.05. Everything else (b = 0.20 and 0.30, other streams, per-seed results, the no-risk ablation) is exploratory. The evaluation refuses to run if the code hash or any frozen-artefact hash differs from the registration.

**Status.** The pre-registration was generated on 2026-10-18 (14:52:51 UTC). The fresh cache was built successfully (mean F1 per cell, evidence x answerer, base / specialist: narrow 0.478 / 0.543, wide 0.561 / 0.616, gold 0.633 / 0.724), consistent with the main cache. The evaluation step then failed because `prereg.json` was not found in the working directory (most likely lost with a session change; the exact cause was not established). The registration is to be regenerated and its hashes compared with the printed ones before evaluating. **No policy has been evaluated on the fresh queries yet, so no confirmatory result exists.**

**How to read the eventual outcome.**

- **2 or 3 of 3 hypotheses confirmed:** the small improvement replicates.
- **0 or 1 confirmed:** the main-test gains were largely noise or selection effects and should be reported that way; the contribution then rests on the environment, the signal audit and the parity finding.

Either outcome is reportable.

---

## Reproducibility and artefacts

| Artefact | Purpose |
|---|---|
| `hitl-orchestrator.zip` (`src/hitl/*.py`, tests, README) | Complete code: pipeline stages, replay environment, PPO-Lagrangian, stream trainer, baselines, evaluation, audit, confirmatory protocol |
| `hitl_kaggle_notebook.ipynb` | Original all-in-Kaggle notebook (cache build + per-query training/evaluation) |
| `hitl_FINAL.ipynb` | CPU-only final run: stream policy (optionally risk-stacked) on the existing cache; generates `FINAL_REPORT.md` |
| `hitl_CONFIRM.ipynb` | Pre-registration, fresh cache build (GPU), frozen confirmatory evaluation |
| `hitl_run_outputs.zip`, `hitl_final_outputs.zip`, `hitl_final_stacked_outputs.zip` | Main cache, unstacked policies and results, stacked policies, risk feature and results. **Keep them:** the confirmatory run needs the stacked policies and the main cache |

**Compute:** Kaggle. T4 GPU for the real pipeline runs via vLLM (cache building); CPU for policy training and evaluation.

**Suggested run order**

1. Build the cache (GPU) with `hitl_kaggle_notebook.ipynb`.
2. Train and evaluate the stream policy (CPU) with `hitl_FINAL.ipynb`; enable risk stacking for the v1.1 variant.
3. For confirmation, run `hitl_CONFIRM.ipynb` with persistence on: register, compare printed hashes, build the fresh cache, then evaluate.

---

## Engineering notes

| Problem | Cause | Fix |
|---|---|---|
| `torchaudio` CUDA mismatch crashed sentence-transformers | pip install of vLLM brought a newer PyTorch (CUDA 13) while Kaggle's torchaudio targeted CUDA 12.8 | Uninstall `torchaudio` (unused). Do **not** remove `torchvision`: vLLM requires it |
| vLLM failed at startup ("No module named torchvision") | `torchvision` had been removed as a precaution | Reinstall `torchvision` without touching other packages |
| Project not found / old code loaded | Kaggle auto-extracts uploaded zips and pins dataset versions | Cell scans all attached inputs, reads each copy's version tag, uses the matching one; version guard |
| `meta.json` / cache not found | Stage cells not run in the current session, or cache wiped | Run stages in order; resumable stores; size guard against mixed sizes |
| Notebook outputs too long to read | Per-iteration training logs | Logs redirected to files; only summaries printed; `comparisons.csv` |
| Policy never chose HUMAN at low budgets | Argmax decoding of a stochastic budget-feasible policy | Sampling decoding; threshold decoding as diagnostic |
| Unfair first comparison | Baselines lacked re-retrieve / specialist actions | Matched-action-set baselines (`wide_gbm`, `wide_spec_gbm`, paced variants) |
| Overfitting on 300 queries | Policy could memorise question embeddings | Validation-based checkpoint selection; full-size training data |
| Confirmatory evaluation could not find `prereg.json` | Working directory changed between registration and evaluation | Re-run registration, compare printed hashes; keep outputs with persistence on |

---

## Hyperparameters

| Item | Value |
|---|---|
| PPO network | 2 hidden layers of 128 tanh units; separate reward and cost critics (cost critic unused in stream mode); Adam, lr 3e-4; clip 0.2; entropy coefficient 0.01 |
| Per-query PPO-Lagrangian | 300 iterations x 1,024 episodes; 6 epochs; minibatch 512; GAE lambda 0.95; dual lr 1.0, lambda init 0.5, max 20; 5 seeds per budget |
| Stream PPO | 250 iterations (200 in 1.0-final); 32 windows of *W* = 200 per iteration; 4 epochs; minibatch 2,048; GAE lambda 0.98; budgets sampled in [0.02, 0.30]; 3 seeds; reward scaled by 10 / *W* |
| Costs | re-retrieve 0.02; specialist 0.05; human review = constraint, F1 = 1 (oracle) |
| Evaluation | *W* = 1,000; orderings: 3 (iid, type-blocks), 1 (hard-first); bootstrap 2,000 (main) / 10,000 (confirmatory) resamples |
| Sampling signals | 5 samples per cell, temperature 0.7, max 24 tokens |
| Risk models | HistGradientBoosting, depth 4, lr 0.05, 200 iterations; one per evidence/answerer cell; 5-fold out-of-fold for training queries |

---

## Version history

| Version | Change |
|---|---|
| 0.1 to 0.5 | Pipeline stages, replay environment, PPO-Lagrangian; matched baselines; validation checkpoint selection; sampled-answer signals; cross-model agreement |
| 0.6-ppo-decode | Sampled and thresholded PPO decodings (fix for unused budgets under argmax) |
| 0.7-comparisons-csv | All CIs saved to a CSV; quieter cost sweep |
| 1.0-final | Stream-level hard budget, drift streams, paced baselines, auto-generated report, one-command run |
| 1.1-stacked | Out-of-fold supervised risk score as a policy input; better coverage of high training budgets |
| 1.2-confirm | Fresh-split support, frozen-artefact pre-registration, Holm-corrected confirmatory evaluation |

---

## Author and citation

**Agnes Teresa Davuri** ([@Davuri124](https://github.com/Davuri124)), B.Tech CSE, National Institute of Technology, Warangal.

```bibtex
@misc{davuri2026hitl,
  author = {Davuri, Agnes Teresa},
  title  = {Budget-Constrained Reinforcement Learning for Human-in-the-Loop Agent Orchestration},
  year   = {2026},
  note   = {Version 1.2-confirm; confirmatory evaluation pending}
}
```

## Data and model sources

- HotpotQA (distractor setting)
- Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct-AWQ
- `BAAI/bge-small-en-v1.5`
- DeBERTa-v3 NLI
- vLLM, PyTorch, scikit-learn (HistGradientBoosting)
