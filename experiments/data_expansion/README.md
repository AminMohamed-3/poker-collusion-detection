# Poker data expansion experiment

The original data is organizer-generated; its simulator was not included in the eight downloaded files. We are testing an independently implemented approximation with explicit policies and causal action-change traces. It is not a reconstruction of the organizer's hidden generator.

## Completed data checks

The first 96,000-hand hand-rule pilot is preserved in `pilot/`. It failed mechanism diversity and domain checks and must not be used as accepted augmentation. One transfer pool had no essential donor action despite its provisional positive assignment. The improved writer requires essential mechanism actions when emitting positive evidence.

The repaired 96,000-hand pilot is in `learned/pilot/`: 32 separate player pools, 960 players, 157 supervised pairs, and 21 positives. Three assigned soft-play targets had no qualifying intervention and are excluded, not relabelled negative. There are no duplicate full hand payloads. All serialized card/chip checks pass, and replay of 320 sampled hands reproduces 3,067 actions and every final net result exactly. The online policy adapter matches 1,748 established offline feature vectors exactly across 64 additional generated hands.

Hand-level source-discrimination AUC fell from .9626 to .6379. Preflop endings moved from 34.65% to 41.45%, versus 47.25% in the same original-training sample. **A subsequent benign-pair feature audit still distinguishes sources with AUC .9879**, chiefly using stack variation and shared-hand counts. That audit uses original training pools only. Coarse hand realism is therefore insufficient, and this version is not ready for large-scale expansion. The earlier audit's `scale_up_admitted` field means only its initial legal/diversity/hand-domain gate; it does not override the deeper pair-domain finding or the utility/realism requirements.

Four ordinary policy models are trained on 217,751 distinct eligible action observations in total (up to 150,000 per fit), with confirmed non-target pair labels and outer-held pools excluded. They have related training data and differ in inner pool exclusions/model size; they are not four independent external sources. Assigned public style and fixed action-temperature choices vary between agents/pools. The legacy tilt/tightness/aggression/noise fields are currently inactive under the learned action function, and **active tilt is not claimed**. That gap must be addressed before larger-scale generation. Coordination-controller access to the partner's current cards is explicit; ordinary actors never receive those cards or future boards.

The engine originally excluded uncalled winning bets from its final collected pot. The organizer's records include gross contributions. The repaired serializer preserves engine net winnings while matching the original displayed-pot convention. This is independently checked in `replay-audit.json`.

## Completed matched utility trial

The `history_repair/` pilot is a further training-only sampler audit: empirical pooled stack quantiles, variable attendance and active tilt. It generates 158 eligible pairs, but benign-pair source AUC remains .9802, so it also fails the deeper realism check. The final bounded `profile_repair/` pilot conditions stacks and public style jointly on 8,910 original-training player prototypes. It generates 143 eligible pairs, including 23 trace-supported positives, and passes actual emitted-label coverage and pool-weight limits. Its hand-domain AUC improves to .6190 and preflop endings to 46.88% versus 47.25% original. However, benign-pair source AUC is still .9228 (ceiling .8), mainly from stack extrema. It also fails admission. Independent replay verifies 320 hands and 2,981 actions exactly. Active tilt changes action distributions in all three prechosen test cases. These are improvements to simulation checks, not model or leaderboard gains.

All four 96,000-hand pilots are preserved separately. They share policy lineage and sampling design and must not be pooled or counted as 384,000 independent new examples. The final two pilots were never substituted into the frozen utility trial. No more generator versions or held-out trials were launched this session. See [final generation audit](GENERATOR_RESULTS.md).

`augmentation/plan.json` freezes two arms with 4,680 rows each: 4,209 original training views plus 471 added views. The control resamples 157 original training pairs with class counts matched exactly to the 157 synthetic pairs; each pair retains full/70%/40% views. Treatment replaces only those extra views with the synthetic histories. Both arms therefore have identical sample count, class weighting and optimizer-step budgets. Neural scaling uses the same original-only data.

The fixed endpoints are matched-budget residual, mHC and Single-Pass mHC (three seeds each), and the incumbent 897-feature and unsubmitted 1,217-feature tree recipes. All 18 neural and four tree fits completed, and all five endpoints failed their preset data-improvement gate. Single-Pass had a small positive mean loss improvement (.000583), below the required .001; the other four endpoints increased mean loss. All original 1,217-feature held predictions reproduced exactly before fitting. The worker archives contain no held labels. The frozen `evaluate_augmentation.py` scored every outcome once locally after completion. See [all utility results](RESULTS.md).

No larger trial is justified by these results. Further omitted-family experiments must exclude the omitted synthetic mechanisms too. Other original validation splits require generator/reference models trained without their held pools; **this fold0 generator must not be reused across arbitrary new held splits**. More rows do not repair that leakage boundary.

## Provenance and reproduction

- `pilot-plan.json`, `pilot-generation.json`, `pilot-audit.json`: original rule-based pilot and failed audit.
- `learned/pilot-plan.json`, `learned/fit.json`, `learned/pilot-audit.json`: repaired generator recipe, ordinary-policy fit provenance and audit.
- `learned/adapter-parity.json`, `learned/replay-audit.json`: feature and serialized game replay checks.
- `learned-returned/`: remote receipts and exact remote source files. The JSON scalar-writing repair preserves all 192 previously written raw payload files byte for byte. Local/remote repaired Python hashes differ by platform line endings; originals and both repair receipts are retained.
- `pilot-results.tar.gz`, `learned-results.tar.gz`: locally backed-up remote datasets. Their individual members were verified before use.
- `augmentation/metadata.json`: pair mapping, class-matched resampling indices, source hashes and limitations.

The small tree fits ran locally. Neural fits and generation ran on the supplied RTX 3080 VM. All remote fits, generated data and receipts are backed up locally with member hashes verified. No Kaggle submission has used these examples, and no competition data or model has been published.

The user's reinforcement-learning fallback is saved in [project memory](../../../../research/POKER_RESEARCH_MEMORY.md), with a [feasibility design](../../../../research/poker-rl-generator-feasibility-20260910.md). It remains deferred because ordinary-history realism still fails. RL could exploit these simulation artifacts instead of adding useful difficulty.
