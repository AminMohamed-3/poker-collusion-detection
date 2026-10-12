# Advanced gameplay model

This iteration retains the original baseline and adds 56 per-hand features based on the actor's and partner's cards, the board prefix available at each street, betting price, stack commitment, previous aggressor, and whether the partner has already folded. Ties are distinguished from strictly stronger hands. No identifier values, hidden labels, or future board cards are prediction inputs.

Card ranks use the open-source [PH Evaluator](https://github.com/HenryRLee/PokerHandEvaluator), version 0.6.0. Its integer rank is used as an ordinal feature, not presented as an equity probability. All data and outputs remain private competition work.

From this directory in PowerShell, after installing `requirements-v2.lock.txt` in the local virtual environment:

```powershell
$env:POLARS_MAX_THREADS='4'
& .\.venv\Scripts\python.exe -X utf8 src\advanced_features.py development
& .\.venv\Scripts\python.exe -X utf8 src\advanced_features.py evaluation
& .\.venv\Scripts\python.exe -X utf8 src\train_v2.py train
& .\.venv\Scripts\python.exe -X utf8 src\robust_pair_experiment.py --fit-final
& .\.venv\Scripts\python.exe -X utf8 src\predict_v2.py --pair-model outputs_v2\pair_model_robust.joblib --output-name submission_advanced_robust.csv
& .\.venv\Scripts\python.exe -X utf8 src\validation.py outputs_v2\submission_advanced_robust.csv --data-dir data --report outputs_v2\submission_advanced_robust.validation.json
```

The original `outputs/` features are required inputs. Advanced files are written only to `outputs_v2/`; its source/input manifest rejects incompatible cache reuse. `predict_v2.py` caches evidence rankings for all three behaviors so alternative pair models can reuse the hand-model computation. Score ties are broken with independent seeded random numbers, without ranking by identifier values. The selected entry file is `outputs_v2/submission_advanced_robust.csv`. The unaugmented candidate is preserved as `outputs_v2/submission_advanced.csv`; the first entry is also preserved.

The final robust pair model uses all 1,860 explicitly labeled pairs in three views: full history, a seeded 70% sample, and a seeded 40% sample. Each view has weight one third, so each pair has total training weight one. Unlisted pairs are never treated as negatives. The separately held-out experiment improved shortened-history AP from 0.965269 to 0.974207. That experiment, model selection, limitations, and final fit provenance are in `reports/robust-pair-experiment/`. The final all-label refit has no held-out score of its own.

Three pool-held-out pair AP scores: 0.980765, 0.979533, 0.983021. Behavior macro AP: 0.959203, 0.973525, 0.969124. Final pair model uses 226 trees, hand model 238. Hand evidence AP against confirmed-negative pairs is 0.700628; evidence MAP@5 given the true family is 0.380969. These are local diagnostics; the latter is not an end-to-end competition score. Development labels have different sampling and player coverage from evaluation candidates, as documented in `reports/validation-gap.md`.

Independent review: 13 focused tests pass in `tests/test_advanced_features.py`, plus structural and raw-gameplay checks in `reports/advanced-feature-review.md`. The first batch preserves every baseline column. The only discovered semantic issue, treating ties as ahead, was fixed before the full build and is covered by tests.

The original entry scored 0.61221 publicly, rank 56/104 at observation. The user submitted the advanced robust file; Kaggle displayed Success with public score 0.77586, rank 35/104 at observation. See outputs_v2/submission_receipt.json. Its SHA-256 is `810ff06973f9b3feedb9f4eab077ef7e0b726aa907ee2a9a8a49c031624e3cfc`. Full validation checked all 112,540 pairs and 562,700 evidence references. No prize or income has been earned.
