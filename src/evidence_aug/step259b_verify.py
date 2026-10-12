"""STEP259b step 5: INDEPENDENT verification of the candidate file.

Nothing here imports the builder or reuses any of its frames: the three CSVs (STEP251, STEP259,
STEP259b) are read from disk as plain text tables and every claim in the brief is checked directly.

  * risk_score and predicted_behavior identical to STEP251 and to STEP259
  * only rows whose predicted_behavior is directed_transfer or soft_play differ from STEP251
  * coordinated_isolation rows: 0 changes;  none rows: 0 changes
  * no repeated hand inside a list, in any row
  * no empty evidence cell
  * 112,540 rows and 562,700 evidence cells
  * every evidence hand is one of the pair's own evaluation-phase hands and the seven protected
    submissions are unchanged

Usage: python step259b_verify.py        Writes outputs_step259b_20260918/verify259b.json.
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import polars as pl                                                            # noqa: E402
from evidence_aug.paths import sha, check_protected, write_json                # noqa: E402

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step259b_20260918'
CAND = OUT / 'submission_step259b_geofusion_dt_sp_on_step251.csv'
STEP251_FILE = CV / 'outputs_step251_20260917/submission_step251_ens4_dtfp3_on_step222.csv'
STEP251_SHA = '804c52f2d4a9e2fdffc91769e48edf713ac489d745874456025e2ee95e7761fb'
STEP259_FILE = CV / 'outputs_step259_20260918/submission_step259_mechfusion_dt_sp_on_step251.csv'
STEP259_SHA = '938e9e71997617c4c91b1003051aca00505300c384753bbcc6325d44ea5b89b9'
EV = ['evidence_hand_%d' % k for k in range(1, 6)]


def main() -> None:
    t0 = time.perf_counter()
    prot = check_protected()
    assert sha(STEP251_FILE) == STEP251_SHA and sha(STEP259_FILE) == STEP259_SHA
    a = pl.read_csv(CAND, infer_schema=False)
    b = pl.read_csv(STEP251_FILE, infer_schema=False)
    c = pl.read_csv(STEP259_FILE, infer_schema=False)
    assert a.columns == b.columns == c.columns == ['pair_id', 'risk_score',
                                                   'predicted_behavior'] + EV, a.columns

    rec = {'candidate': str(CAND.relative_to(CV)), 'candidate_sha256': sha(CAND),
           'step251_sha256': sha(STEP251_FILE), 'step259_sha256': sha(STEP259_FILE),
           'rows': a.height, 'rows_expected': 112540,
           'evidence_cells': a.height * 5, 'evidence_cells_expected': 562700}

    # ---- identity of the non-evidence part
    key = ['pair_id', 'risk_score', 'predicted_behavior']
    rec['risk_behaviour_identical_to_step251'] = a.select(key).equals(b.select(key))
    rec['risk_behaviour_identical_to_step259'] = a.select(key).equals(c.select(key))
    rec['pair_order_identical_to_step251'] = a['pair_id'].to_list() == b['pair_id'].to_list()

    # ---- which rows moved, by family
    fams = a['predicted_behavior'].to_list()
    ra, rb, rc = a.select(EV).rows(), b.select(EV).rows(), c.select(EV).rows()
    d251, d259 = {}, {}
    s251, s259 = {}, {}
    for x, y, z, f in zip(ra, rb, rc, fams):
        if x != y:
            d251[f] = d251.get(f, 0) + 1
            s251[f] = s251.get(f, 0) + len(set(x) - set(y))
        if x != z:
            d259[f] = d259.get(f, 0) + 1
            s259[f] = s259.get(f, 0) + len(set(x) - set(z))
    rec['rows_differing_from_step251_by_family'] = d251
    rec['rows_differing_from_step251'] = sum(d251.values())
    rec['top5_set_changes_vs_step251_by_family'] = s251
    rec['rows_differing_from_step259_by_family'] = d259
    rec['rows_differing_from_step259'] = sum(d259.values())
    rec['top5_set_changes_vs_step259_by_family'] = s259
    rec['families_allowed_to_move'] = ['directed_transfer', 'soft_play']
    rec['only_dt_sp_moved_vs_step251'] = set(d251) <= {'directed_transfer', 'soft_play'}
    rec['only_dt_sp_moved_vs_step259'] = set(d259) <= {'directed_transfer', 'soft_play'}
    rec['coordinated_isolation_rows_changed_vs_step251'] = d251.get('coordinated_isolation', 0)
    rec['coordinated_isolation_rows_changed_vs_step259'] = d259.get('coordinated_isolation', 0)
    rec['none_rows_changed_vs_step251'] = d251.get('none', 0)
    rec['rows_by_family'] = {f: fams.count(f) for f in sorted(set(fams))}

    # ---- well-formedness
    rec['empty_or_null_evidence_cells'] = sum(
        int((a[col] == '').sum() + a[col].null_count()) for col in EV)
    rec['rows_with_a_repeated_hand'] = sum(1 for x in ra if len(set(x)) != 5)
    rec['distinct_evidence_hands'] = len({h for x in ra for h in x})

    # ---- every evidence hand really is one of that pair's own evaluation-phase hands
    pairs = pl.read_csv(POKER / 'data/evaluation_pairs.csv',
                        columns=['pair_id', 'player_1', 'player_2'])
    hands = pl.read_parquet(POKER / 'data/hands.parquet', columns=['hand_id', 'phase'])
    used = pl.DataFrame({'hand_id': sorted({h for x in ra for h in x})})
    j = used.join(hands, on='hand_id', how='left')
    rec['evidence_hands_found_in_hands_parquet'] = int(j['phase'].is_not_null().sum())
    rec['evidence_hands_all_evaluation_phase'] = bool((j['phase'] == 'evaluation').all())
    rec['evaluation_pairs_rows'] = pairs.height

    long = a.select(['pair_id'] + EV).unpivot(index='pair_id', on=EV,
                                              variable_name='slot', value_name='hand_id')
    seats = pl.read_parquet(POKER / 'data/seats.parquet', columns=['hand_id', 'player_id'])
    seats = seats.join(used, on='hand_id', how='semi')
    lp = long.join(pairs, on='pair_id', how='left')
    for who in ('player_1', 'player_2'):
        lp = lp.join(seats.with_columns(pl.lit(1).alias('ok_' + who)),
                     left_on=['hand_id', who], right_on=['hand_id', 'player_id'], how='left')
    bad = lp.filter(pl.col('ok_player_1').is_null() | pl.col('ok_player_2').is_null()).height
    rec['evidence_rows_where_both_members_were_dealt'] = lp.height - bad
    rec['evidence_rows_total'] = lp.height
    rec['evidence_membership_violations'] = bad

    rec['protected_unchanged'] = check_protected() == prot
    rec['ALL_CHECKS_PASS'] = all([
        rec['rows'] == 112540, rec['evidence_cells'] == 562700,
        rec['risk_behaviour_identical_to_step251'], rec['risk_behaviour_identical_to_step259'],
        rec['pair_order_identical_to_step251'],
        rec['only_dt_sp_moved_vs_step251'], rec['only_dt_sp_moved_vs_step259'],
        rec['coordinated_isolation_rows_changed_vs_step251'] == 0,
        rec['coordinated_isolation_rows_changed_vs_step259'] == 0,
        rec['none_rows_changed_vs_step251'] == 0,
        rec['empty_or_null_evidence_cells'] == 0, rec['rows_with_a_repeated_hand'] == 0,
        rec['evidence_hands_all_evaluation_phase'], rec['evidence_membership_violations'] == 0,
        rec['protected_unchanged'], rec['rows_differing_from_step259'] > 0])
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    rec['runner_sha256'] = sha(Path(__file__))
    write_json(OUT / 'verify259b.json', rec)
    print(json.dumps(rec, indent=1), flush=True)
    assert rec['ALL_CHECKS_PASS'], 'independent verification FAILED'


if __name__ == '__main__':
    main()
