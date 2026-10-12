# Submission 56368246 — TabLDM risk-source addition

The September 19, 2026 submission scored 0.91898 on the public leaderboard, compared with 0.91772 for submission 56364176. The authenticated leaderboard placed it twelfth at submission time; fifth place was 0.93048. This is an observed public improvement, not a private-score claim or completion of the top-five goal.

The submitted file is `foundation_pair/tabldm_full_candidate/submission_tabldm_equal_rank80_20_on_pre_fast.csv`, SHA-256 `d32161a41aaffaaa83a9d06335480968f02056d2cf33e93fa53aa7b05726272a`. It contains 112,540 unique, sorted pair IDs. Behavior and evidence fields are byte-for-byte identical to the preceding best submission, so its five previously documented evidence cases remain applicable without modification.

Only the risk ranking changed. Let `r2` denote twice the global average rank, retaining integer half-tie ranks, and let N be the number of evaluation pairs. The risk score is:

`(4*r2(pre-Fast risk) + r2(r2(Fast risk) + r2(TabLDM risk))) / (10*N)`.

The outer base is the pre-Fast 0.91411 submission, SHA-256 `0493a807d81b7f3cc5ec54a74c86d08c362b72305fceee8e6af9044740990fc9`. It is not the already fused 0.91772 submission. Both learned risk sources use `1 - p_none`, casting probabilities to float64 before subtraction. Scores are written with 17 significant digits and checked for exact round-trip equality. Replacing the added source with Fast reproduced the previous best CSV byte-for-byte in an identity test.

TabLDM uses the same 128 selected gameplay features and 1,860 trusted development pairs as the original Fast production context. The pinned base TabLDM classifier uses four estimators, seed 35935, batch size one, two jobs, enhancement disabled and KV cache disabled. Prediction uses copied input arrays in sorted, fixed 256-row query chunks. This chunk convention matters because a synthetic test found query-cohort dependence. No pair, table or player ID is a predictor.

The model source is pinned to Xiaomi TabLDM commit `3335755ad538b80614f000226e2856b543a0273d`; the checkpoint SHA-256 is `7f8ec8b949c13e7187c0f11b189e03560eb28e02d286aa2cdad2d2a56b62bc9e`. Provenance and the available source/model-card licensing statements are retained in the campaign's TabLDM directory.

Validation was mixed, and the upload was explicitly exploratory. The curated full-history test failed its gate. A separate mostly unlabeled population test improved pooled metrics but narrowly worsened one fold; shortened-history curated results were positive. These failures remain recorded. The public improvement does not retroactively validate those gates. Unknown pairs were not relabeled as confirmed negatives.

Full inference completed in 1,344 seconds. A separate process refitted the model and reproduced the first 1,024 prediction vectors exactly. Root and an independent reviewer checked actual artifact hashes, row alignment, the global formula, unchanged fields and the replay before upload. This verifies cached prediction/assembly integrity; it is not a claim that every historical upstream feature was independently regenerated in this replay.

The build, prediction, replay, independent-review and authenticated submission receipts remain in their original directories. No manual final-submission selection was changed by this upload.
