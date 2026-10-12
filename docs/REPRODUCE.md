# Reproducing the submitted chain

Everything needed to rebuild `outputs_stage2_r382_20260911/submission_S_stage2_r382DTSP.csv`
(public 0.89336) from the raw competition data, plus the pre-registered probe built on top of it.
Written 2026-09-12.

Every path below is relative to `C:/Data/Projects/Money Making/work/poker` (call it `$ROOT`);
`$CV = $ROOT/Claude Version`. The original tree is read-only: everything this project writes lives
under `$CV/src`, `$CV/experiments` and `$CV/outputs_*`.

---

## 1. Environment

| | |
|---|---|
| Python | 3.12.14 (MSC v.1944, 64-bit), at `$ROOT/.venv/Scripts/python.exe` |
| Lock files | `$ROOT/requirements.lock.txt` (sha256 `a7b3adc2a84cf77ab0f0e6f8289d2cf82ffda528f9c5255d7f563a8190b856f1`), `$ROOT/requirements-v2.lock.txt` (`72e986fd2c393883085e7a471272ca2d274f558164ac3979ee8232cba79f7f92`) |
| Key versions | polars 1.44.1, numpy 2.5.3, lightgbm 4.7.0, scikit-learn 1.9.0, joblib 1.6.0. **No pandas.** |
| Threads | every script pins `POLARS_MAX_THREADS=2` and `OMP_NUM_THREADS=2`; LightGBM uses `n_jobs=2`. Results are seed-fixed and thread-count independent, but keep these to reproduce the timings. |
| Encoding | `PYTHONIOENCODING=utf-8`; every file is read and written with `encoding='utf-8'`. |

Command template used for every step:

```bash
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m <module>
```

## 2. Inputs and their hashes

| role | path | sha256 |
|---|---|---|
| hands | `data/hands.parquet` | `82c9ad1c01cae6a68b28e194b9dd0bad9ca09f94628b429c0e58e970cc29eecd` |
| seats | `data/seats.parquet` | `347a68268d277079751368b7d3169793da5c547f7421e1851731e557b67ec8e7` |
| actions | `data/actions.parquet` | `ef9ef88d65e88e001f80bda69e84cc56290aa33f31530549b4611f5785a08166` |
| evaluation pairs | `data/evaluation_pairs.csv` | `8986e4c99bf2578d7b6a089982dba4ff60d995b152b97a20f9d41d5e11b14b22` |
| base submission (public 0.87625) | `outputs_v13_coverage/submission_conditional_coverage.csv` | `418084a433e9f251f01b9e7d69028586fb2be7d818ff15bc0a6a856121a21581` |
| incumbent evidence ranker | `outputs_v11_conditional/evidence-model.joblib` | `ef1b9384bad26e5de0463a67c383750fb940e8816e9703f6a4781870b587081d` |

`data/hands.parquet` carries a `phase` column. **Every one of the 400 tables is exactly 60 %
`development` hands followed by 40 % `evaluation` hands, with no interleaving at all** (0 of 400
tables have a development hand at or after any evaluation hand). Development labels may be used;
the evaluation phase is label-free and stays that way.

Seven files in the original tree are protected and must never change. Every script that writes under
`outputs_*` calls `evidence_aug.paths.check_protected()` before and after; it asserts the full sha256
of all seven against `experiments/bet_size_utility_20260910/study-plan.json`.

## 3. The chain

Scoring is `0.7 * PairAP + 0.2 * EvidenceMAP@5 + 0.1 * BehaviourMAP`. The chain changes the pair
order (PairAP) and the evidence lists (EvidenceMAP@5); the behaviour column is inherited unchanged
from the base at every step.

```
outputs_v13_coverage/submission_conditional_coverage.csv          public .87625
        |
        |  (A) pu_pairs.step59_blend_candidate --build            pair order only
        v
outputs_blend_20260911/submission_blend_w0.5_plus_R1.csv    = S   public .89055 *
        |
        |  (B) evidence_aug.step13_build_chronology_candidates    stage-1 evidence ranker
        |  (C) evidence_aug.step65_structural_evidence            structural features
        |  (D) evidence_aug.step65k_r382_candidate                stage-2 evidence, hybrid routing
        v
outputs_stage2_r382_20260911/submission_S_stage2_r382DTSP.csv     public .89336
        |
        |  (E) evidence_aug.step69_ci_revert_candidate            probe: revert the CI branch
        v
outputs_ci_revert_20260912/submission_S_r382DTSP_CIstage1.csv     NOT YET UPLOADED
```

\* **Unconfirmed.** The 0.89055 reading is assumed to be S; that has not been verified against the
leaderboard history, and every arithmetic decomposition in `OBSERVATIONS.md` that uses it inherits
the assumption. See "Open" at the end.

### (A) Pair order — candidate S

```bash
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m pu_pairs.step59_blend_candidate --w 0.5 --build
```

S is a **rank permutation of the base's non-R1 block**: the block is reordered by
`logit(FULL) + 0.5 * logit(RAW)`, and the sorted multiset of the base's own block risk values is
re-assigned in that order, so the value distribution and the block membership are unchanged. R1 pairs
receive `[.49, .4949]` by `risk_noT`. Every other row is byte-identical to the base. The blend weight
`w = 0.5` was chosen on the development-label 60 %-prefix testbed of step 58, never on a leaderboard
outcome.

Output: `outputs_blend_20260911/submission_blend_w0.5_plus_R1.csv`
sha256 `aa7c297974bd94dfada8f897f7d22f192dc2c7c5c7adae846334d170b71675d2`

### (B) Stage-1 evidence ranker

```bash
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m evidence_aug.step13_build_chronology_candidates
```

A LambdaRank ranker (`rank_evidence.PARAMS`) on **382 conditional features + `chrono_frac`**, fitted
on all 372 annotated development targets. `chrono_frac` is the hand's position within the pair's own
eligible history, as a ratio — deliberately scale-free, because evaluation histories are shorter than
development ones.

Two reproduction controls run before anything is written: a refit of the incumbent ranker must
reproduce `outputs_v11_conditional/evidence-model.joblib` predictions bit-for-bit, and per evaluation
batch the incumbent's ranks must reproduce the v13 evidence lists exactly for all 16,060 selected
pairs.

Outputs: `outputs_chronology_20260911/` — the fitted ranker
`evidence-chronology_frac.joblib` (`c2827fa96e4d62658de75e9d8d22d92459552cc86253687e749afe3ffee25e72`)
and `submission_chronology_frac.csv`
(`2c092cf5ac0b35de97a42998c215ee1aa419e8c22289f982a892579b75679f2f`).

### (C) Structural features

```bash
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m evidence_aug.step65_structural_evidence   # development / audit side
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m evidence_aug.step65e_eval_structural       # evaluation side, 21 batches
```

`step65` builds `experiments/label_process_20260911/audit-structural-features.parquet` (the training
and audit side); `step65e` applies the same `hand_features` to every hand of the 16,060 selected
evaluation pairs, batch by batch and label-free, into
`experiments/family_stage2_20260911/eval_structural/batch_XXX.parquet`. Both are the same ~70 columns.

~70 per-hand relational-action columns describing what the two members did *to each other* and to the
table: preflop shape (`pf_open_pair`, `pf_iso_raise`, `pf_n_raises`, `pf_others_fold`, …), postflop
interaction (`post_pair_aggr`, `post_pair_calls`, `post_victim_fold_to_pair_bet`, …), and outcome
(`pair_net_bb`, `pair_net_gap`, `pair_sd`, `pair_won`, `hu_pair_only`, `opener_net_bb`, …). In these
columns **`A = player_1` and `B = player_2`** of `evaluation_pairs.csv`.

### (D) Stage-2 evidence re-ranker — the shipped candidate

```bash
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m evidence_aug.step65k_r382_candidate
```

Stage 2 re-ranks only the **top 20** stage-1 candidates of each pair. One
`LGBMClassifier` per family, three seeds averaged:

```python
CLF = dict(n_estimators=150, num_leaves=7, max_depth=3, min_child_samples=40,
           learning_rate=.05, n_jobs=2, verbosity=-1)   # random_state in (11, 12, 13)
TOP = 20
```

Training rows are the top-20 out-of-fold candidates of the 372 development targets, under a 5-way
pool-hash split (`sha256('audit-trunc-v1:' + table_id) % 5`) so that no pair is scored by a model that
saw its own table. Two feature designs:

* `struct` = ~70 structural + `chrono_rank, chrono_frac, n_hands` + stage-1 score and rank;
* `r382` = the same plus the 378 ranker columns.

**The shipped model is the hybrid**: at inference each pair is routed by its own
`predicted_behavior` — `r382` for predicted directed transfer and soft play, `struct` for predicted
coordinated isolation. The hybrid is a **post-hoc** selection made after seeing the per-family results
of the single held read of folds 0–2 (step 65j), and is labelled post-hoc everywhere it appears.

Reproduction controls before writing: stage-1 ranks must reproduce the incumbent lists, and the
step-65f `struct` stage-2 lists must be reproduced from the step-65f bundle.

Output: `outputs_stage2_r382_20260911/submission_S_stage2_r382DTSP.csv`
sha256 `437b812a0fc39aeb2af2a480ea81edede10ad8c374b2bfd7d9b894305430302a` — **public 0.89336**.

### (E) The CI-revert probe — built, pre-registered, not uploaded

```bash
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m evidence_aug.step69_ci_revert_candidate
```

Identical to (D) except that pairs with `predicted_behavior == coordinated_isolation` keep their
**stage-1** evidence lists. Because nothing else differs, the public difference is an exact
attribution of the isolation branch. Pre-registered as amendment_1: point estimate **+0.0010**,
interval **[−0.0060, +0.0050]**.

Output: `outputs_ci_revert_20260912/submission_S_r382DTSP_CIstage1.csv`
sha256 `a392d511fa7347d0a954a8d35e7a7bf86585be9c5885048c4022a2b887e79044`

## 4. Other submitted / built candidates

| file | sha256 | note |
|---|---|---|
| `outputs_stage2_20260911/submission_M3_stage2_chrono3.csv` | `3679ebd0f5dd4f58536a05aec1e3a0c2bdeea29b3ce25a552d3a58807a673964` | struct stage 2 on the M3 order |
| `outputs_stage2_20260911/submission_S_stage2_chrono3.csv` | `b23335b9e6b342a8cad6442d51f63792414da1759fb55fc31f27ae84c87cc852` | struct stage 2 on S |
| `outputs_multiview_20260911/submission_M4_stage2_chrono3.csv` | `2a114a952124b81b03ceb568a572c8c4b6f47cad97c7a40622b34b36616b9121` | |
| `outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv` | `6f064147875b0c448355666eb6448a2f5105f19cd89bcaf4a5308d995b7f678b` | r382 hybrid on the M3 order |

## 5. How selection was kept honest

* **Two evaluation surfaces, used differently.** Model selection runs on the *training-only audit set*
  — the 162 positive pairs in the 172 tables outside every held fold, 19,325 hand rows, 5-way
  pool-hash out-of-fold. Confirmation runs at most once per study on held folds 0–2, against a gate
  written into a hashed `freeze.json` **before** the read. Folds 0–2 have now been read four times in
  total (steps 11, 65d, 65j, 72b).
* **No leaderboard tuning.** No feature, threshold, blend weight or candidate was chosen using a
  leaderboard score. Public readings are used only to *attribute* differences between already-frozen
  candidates, and each such reading is pre-registered with a point estimate and an interval before the
  file is uploaded.
* **Public-split resolution.** The pair-side delta has a standard deviation of about .007 composite,
  the evidence-only delta about .003. Two candidates are never ranked by a smaller public difference
  than that.
* **Evidence is never routed by true family.** Routing at inference uses `predicted_behavior` only.
* **Post-hoc stays labelled.** The hybrid routing (D) and the E6 hybrid reconstruction in OBS 18.14
  are post-hoc selections and are never promoted above a cleanly confirmed candidate.
* **Failures are preserved as failures.** `OBSERVATIONS.md` keeps every refuted hypothesis, including
  the two cases where a pre-registered rule fired on a statistic that turned out not to measure what
  it claimed (P3 in 18.15, E4 in 18.17).

## 6. Worked cases

`research/case_reviews_20260912/` contains five descriptive case reviews of evaluation pairs from the
shipped chain's high-risk block — two predicted directed transfer, two soft play, one coordinated
isolation, chosen by a rule fixed before looking (highest-risk pair per family, plus the family
median for the two-case families). Each shows the five submitted evidence hands, their position in
the pair's own eligible chronology, the structural columns stage 2 sees, and the action. There are no
evaluation labels, so these are readings of what the model selected, not claims about the pairs.

Rebuild with:

```bash
cd "$CV/src" && PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/Scripts/python.exe" -m evidence_aug.step75_case_reviews
```

(The `Reading` section of each file and the synthesis in `README.md` are written by hand and are
overwritten by a re-run; restore them from git or re-write them.)

## 7. Open

* **Which file scored 0.89055 has never been confirmed.** The chain diagram above and the public
  arithmetic in OBS 18.x assume it was S. If it was some other file, the two public readings that
  decompose 0.89336 change, and with them the conclusion that the struct stage 2 did not transfer
  beyond r382.
* `outputs_ci_revert_20260912/submission_S_r382DTSP_CIstage1.csv` is built, validated and
  pre-registered but **not uploaded**. Uploading is done by the project owner, never by the agent.
