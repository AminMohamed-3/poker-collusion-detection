# STEP276 shared artefacts (evidence session) -- readable by the STEP279 recall lane

Copied here so the pair-risk session can use them without re-deriving. Read-only as far as I am concerned;
if you need a variant, copy rather than edit, because my OBSERVATIONS 19.230/19.231 numbers reference these.

## order_labels.parquet  (1,817 rows = every cited hand in development)
`pair_id, hand_id, evidence_rank, behavior_family, chrono_frac, T1, facing, fold_street, folder_equity,
in_audit, oracle_b2, oracle_b3`

`evidence_rank` is the GENERATOR'S OWN 1-5 ordering of the cited hands, from `data/development_evidence.csv`.
Nothing in this project had read that column before 2026-09-19. It is the supervision signal behind 19.230:
the generator's traversal order is (block, chronology), and the honest cross-fitted block rule

    block 0  <=>  T1  OR  t1_ap_call_aggressor_sm <= 0.5

reaches .9354 concordance against `evidence_rank` versus production's T1-only .9047, with the same rule chosen
in all five folds for directed_transfer and soft_play. `oracle_b2`/`oracle_b3` are the best-fitting block
assignments WITH the labels (ORACLE2 .9989 concordance) -- they leak, use them only for ceilings.

Caveat measured in 19.231: the concordance gain does NOT convert. Through the decoder the rule moves 8 of the
162 audit lists, 3 up and 5 down, for -.00041. Its lever is ~±.0002 composite. Do not spend a slot on it.

## audit_blocks.parquet  (19,325 rows = all hands of the 162 audit pairs)
`pair_id, hand_id, table_id, behavior_family, relevance, t, both_sd, none_sd, any_fold, any_sd, n_sd, n_folded,
w_net, l_net, final_pot, players_at_showdown, players_dealt`
Label-FREE structural flags (`relevance` is present but is the label -- do not feed it). Built for the traversal
battery; all 12 orders I tried are in `dec_battery.json` and production's (T1, CI off) wins at .733740, with
flat chronology at .701399.

## ev_lists.parquet
Per-list view of the same evidence data (372 pairs), for list-level questions.

## Numbers you may want to quote rather than recompute (162-query audit, production = v1_DTfp3)
| quantity | value |
|---|---|
| decoded MAP@5 | .733740 |
| raw MAP@5 (no decoder) | .686353 |
| stage-1 top-20 recall | .9710 |
| within-pair AUC / AP | .93375 / .84677 |
| per-pair sum(p) vs true listed in pool | 5.030 vs 4.759 |
| per-family decoded (DT / SP / CI) | .739514 / .685337 / .728915 |
| my REPRODUCTION of stage 2 (REF), wAP | .83072 -- .016 BELOW production, see 19.231(c) |

## Reproducing the decoder without touching the protected module
`step104_C_typed_decoder_build.decode` is hash-registered. To vary it, rebuild it locally: per pair, over rows
with score > 5, `order = lexsort((chrono_rank, ~T1))`, `q[order] = less_than_five(p[order])`,
`out = 10 + p*q**gamma + 1e-9*p`, with T1 forced False for coordinated_isolation. At gamma = 1 that is
bit-identical to production -- assert it before trusting any variant. A generalised `less_than_k(ps, k)` for
varying the cap is four lines; the production one hard-codes 5 with an absorbing state at 5.
