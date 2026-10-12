"""STEP259b step 2: dump the EXACT fusion scope of the STEP259 builder, without changing any evidence.

This instruments a COPY of src/evidence_aug/step259_P_fusion_build.py's key selection - the twenty
outputs_step251_20260917/work/INC/batches/batch_%03d-lp-scores.parquet files, concatenated and sorted
on CSP.KEYS - together with the base file's pair -> predicted_behavior map, which is what decides
whether a row is fused (directed_transfer -> p_DT, soft_play -> p_SP) or not (everything else).

Nothing is built and no evidence is touched; the output is a key list plus its family labels, checked
against receipt259.json's fusion_rows (DT 184,600 / SP 90,580 / other 46,020 = 321,200).

Usage: python step259b_scope_keys.py        Writes outputs_step259b_20260918/scope_keys.parquet.
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
from evidence_aug.paths import sha, write_json                                 # noqa: E402
from evidence_aug import step167_C_lp_production as S167                       # noqa: E402

CV = POKER / 'Claude Version'
S251 = CV / 'outputs_step251_20260917'
OUT = CV / 'outputs_step259b_20260918'
BASE = CV / 'outputs_step169_20260915/submission_step169.csv'
BASE_SHA = '67060d0a53b32ef6c790ce0255f5dd5b6230abcae464ba0c3baed02b4543b9c1'
STEP251_FILE = S251 / 'submission_step251_ens4_dtfp3_on_step222.csv'
STEP251_SHA = '804c52f2d4a9e2fdffc91769e48edf713ac489d745874456025e2ee95e7761fb'
MEMBERS = ['INC', 'N06_lam_list2_0', 'N07_lam_list2_2', 'N08_w_stage1_3']
FUSED_FAMS = (('directed_transfer', 'p_DT'), ('soft_play', 'p_SP'))
RECEIPT259 = CV / 'outputs_step259_20260918/receipt259.json'


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    S108, _ = S167.modules()
    KEYS = list(S108.CSP.KEYS)
    assert KEYS == ['pair_id', 'hand_id'], KEYS
    assert sha(BASE) == BASE_SHA, 'base file changed'
    assert sha(STEP251_FILE) == STEP251_SHA, 'the STEP251 file changed'
    base = pl.read_csv(BASE, infer_schema=False)
    fam = dict(base.select('pair_id', 'predicted_behavior').iter_rows())

    # ---- exactly the builder's `allkeys`
    allkeys = []
    for i in range(20):
        p = S251 / 'work' / MEMBERS[0] / 'batches' / ('batch_%03d-lp-scores.parquet' % i)
        allkeys.append(pl.read_parquet(p, columns=KEYS))
    allkeys = pl.concat(allkeys).sort(KEYS)

    famcol = [fam[p] for p in allkeys['pair_id'].to_list()]
    fused = [dict(FUSED_FAMS).get(f) for f in famcol]
    scope = allkeys.with_columns([pl.Series('predicted_behavior', famcol),
                                  pl.Series('fused_column', fused, dtype=pl.Utf8)])
    cov = {'directed_transfer': 0, 'soft_play': 0, 'other': 0}
    for f in famcol:
        cov[f if f in dict(FUSED_FAMS) else 'other'] += 1
    exp = json.loads(RECEIPT259.read_text(encoding='utf-8'))['fusion_rows']
    assert cov == exp, ('fusion scope differs from receipt259.json', cov, exp)
    assert scope.height == 321200 == sum(cov.values())
    assert scope.select(KEYS).n_unique() == scope.height, 'duplicate (pair, hand) key'

    scope.write_parquet(OUT / 'scope_keys.parquet')
    rec = {'rows': scope.height, 'fusion_rows': cov, 'receipt259_fusion_rows': exp,
           'matches_receipt259': cov == exp,
           'distinct_hands': int(scope['hand_id'].n_unique()),
           'distinct_pairs': int(scope['pair_id'].n_unique()),
           'distinct_hands_fused': int(scope.filter(pl.col('fused_column').is_not_null())
                                       ['hand_id'].n_unique()),
           'keys': KEYS, 'members_source': MEMBERS[0],
           'base_sha256': BASE_SHA, 'step251_sha256': STEP251_SHA,
           'output': 'outputs_step259b_20260918/scope_keys.parquet',
           'output_sha256': sha(OUT / 'scope_keys.parquet'),
           'runner_sha256': sha(Path(__file__)),
           'seconds': round(time.perf_counter() - t0, 1)}
    write_json(OUT / 'scope_keys_receipt.json', rec)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == '__main__':
    main()
