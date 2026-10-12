# Detecting suspicious value transfers in poker

Code for the Kaggle competition
[Detect Suspicious Value Transfers in Poker](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker)
(ended 2026-09-20). The data is synthetic: no real players or gambling are involved.

**Result: 16th of 371, private score 0.92056 (public 0.91411).**

> **Research purposes only.** This repository is published for research on collusion detection. It
> may not be used commercially, and it may not be used to develop any solution related to real-world
> gambling, in any capacity. See [LICENSE](LICENSE).

This is a code-only export of the working tree. Competition data, feature caches, models, submission
files, result JSON, run logs and the agent coordination notes are not included, so nothing here runs
as is. It shows the structure of the work and the code behind it.

## The task

For each pair of players at a table, predict

1. whether the pair is colluding (pair ranking),
2. which behaviour family it uses (soft play, directed transfer, coordinated isolation),
3. up to five shared hands that are the best evidence.

Score = `0.7 * PairAP + 0.2 * EvidenceMAP@5 + 0.1 * BehaviourMAP`.

## Approach

- **Features** from raw hands, seats and actions: contributions in big blinds, aggression, responses to
  a specific opponent, hand strength, partner-specific event excess. Identifiers are used for joins and
  grouping only.
- **Validation** holds out whole tables, so players never cross folds. Each table is 60% development
  hands followed by 40% evaluation hands; evaluation stays label-free.
- **Pair model**: LightGBM on labelled development pairs, then positive-unlabelled blends and a
  stage-2 re-ordering matched to the evaluation history lengths.
- **Evidence model**: per-family hand rankers, fused with mechanical event scores and decoded into a
  list of five.
- **Side studies**: synthetic hand generation, tabular foundation models (TabPFN, TabICL), an LLM hand
  judge with LoRA fine-tuning, a search for a hidden fourth family. Most of these did not help.

## Layout

| Path | Contents |
|---|---|
| `src/` | Main pipeline. Top level: baseline and v2 to v15 models (`features.py`, `train.py`, `validation.py`, ...). |
| `src/pu_pairs/` | Pair ranking: positive-unlabelled models, blends, stage 2. |
| `src/evidence_aug/` | Evidence ranking: augmentation studies, fusion, production builds. |
| `src/hidden_family/` | Partner-specific event model and the hidden-family search. |
| `src/llm_judge/` | LLM hand judge: dataset rendering, reward, scoring. |
| `src/policy/` | Action-policy and card-knowledge models. |
| `tests/` | Unit tests for features, cache protocol, validation. |
| `synthetic/` | Hand simulator used for synthetic collusion data (dealer, engine, validator, seat briefs). |
| `experiments/` | One folder per study (60 with code): runners, audits, VM scripts. |
| `goal_top5_20260919/` | Final-week lane: foundation-model pair screens, PU, reproduction of the final file. |
| `codex_resume/` | Second agent lineage: its own `src/`, experiments and research probes. |
| `fable_version/` | Marking-rule induction study. |
| `opencode_tasks/` | Small profiling and GPU launch tasks. |
| `reports/` | A few launch and rebuild scripts kept from the report folders. |
| `docs/` | `REPRODUCE.md` (chain for the 0.893 file) and the early version notes in `docs/history/`. |

## Notes on this export

- The work was done by several coding agents in parallel folders (`Claude Version/`, `Codex Resume/`,
  `Fable Version/`). They are merged here: `Claude Version/src` became `src/`, and both experiment trees
  became `experiments/`. Scripts that locate files through those original folder names need their
  paths adjusted.
- Frozen copies of `src` modules that sat beside experiment runs were dropped. An experiment that
  imports a module it does not contain is using the one in `src/`.
- Third-party code is not included: the TabLDM overlay, the reference model in `experiments/mhc_local`,
  public notebooks from other competitors.
- Machine-specific paths use `C:/Users/USER`, and rented VM addresses are replaced by `vm-host.invalid`.
- Dependencies for the baseline are pinned in `requirements.lock.txt` and `requirements-v2.lock.txt`
  (Python 3.12, polars, LightGBM, scikit-learn). The GPU studies used extra packages installed on
  rented machines.
- The competition data may only be used under the competition rules and must be downloaded from Kaggle.

## License

Released under the [Poker Collusion Detection Research License 1.0](LICENSE), a custom
source-available license. In short:

- research use only, with attribution;
- no commercial use;
- no use in, or to develop, anything related to real-world gambling, including operator-side
  detection systems;
- derivatives, including trained models, carry the same terms.

The license text governs. This is not an open-source license.
