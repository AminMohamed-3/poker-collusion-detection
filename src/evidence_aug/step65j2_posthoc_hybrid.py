"""Step 65j2: POST-HOC derived quantity from the step-65j read (no new held truth is read; only pairs.csv is used).
The pre-registered primary (r382 for every family) FAILED the gate through the coordinated-isolation family
(pooled -.018, fold 2 -.043) while transfer / soft play gained (+.022 / +.024). The per-family split suggests the
hybrid 'r382 for predicted transfer / soft play, the step-65c struct model for predicted isolation'. Its held MAP is a
re-combination of per-pair MAPs already recorded in pairs.csv (routing by PREDICTED family, as in production), so it
costs no new read; but it is one selection among the 2^3 family subsets made AFTER seeing the per-family results, and
is recorded as such. Output: posthoc_hybrid.json (results.json untouched)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence_aug.paths import write_json
from evidence_aug.frame import FAMILIES
from evidence_aug.step5_evaluate import pool_bootstrap, half
from evidence_aug.step65i_r382_stage2_fit import OUT
import numpy as np
import polars as pl

CI = 'coordinated_isolation'


def main():
    assert not (OUT / 'posthoc_hybrid.json').exists()
    p = pl.read_csv(OUT / 'pairs.csv')
    base = p.filter(pl.col('candidate') == 'allfam_top20_struct'); r = p.filter(pl.col('candidate') == 'allfam_top20_r382'); ff = p.filter(pl.col('candidate') == 'allfam_top20_r382ff50')
    out = {'post_hoc': True, 'note': 'derived from pairs.csv of the single step-65j read; selection among family subsets made after seeing per-family results', 'variants': {}}
    for name, alt in (('hybrid_DTSP_r382', r), ('hybrid_DTSP_r382ff50', ff)):
        j = base.join(alt.select('fold', 'pair_id', pl.col('map5').alias('alt')), on=['fold', 'pair_id'], validate='1:1')
        j = j.with_columns(pl.when(pl.col('predicted') == CI).then(pl.col('map5')).otherwise(pl.col('alt')).alias('hyb'))
        delta = {f"{f}:{q}": float(h - m) for f, q, m, h in j.select('fold', 'pair_id', 'map5', 'hyb').iter_rows()}
        table = {f"{f}:{q}": f"{f}:{t}" for f, q, t in j.select('fold', 'pair_id', 'table_id').iter_rows()}
        fam = {f"{f}:{q}": fm for f, q, fm in j.select('fold', 'pair_id', 'family').iter_rows()}
        per_fold = {str(f): float(j.filter(pl.col('fold') == f).select((pl.col('hyb') - pl.col('map5')).mean()).item()) for f in sorted(set(j['fold']))}
        halves = {str(hh): float(np.mean([d for k, d in delta.items() if half(table[k].split(':')[1]) == hh])) for hh in [0, 1]}
        out['variants'][name] = {'pooled_delta_vs_allfam_top20_struct': float(np.mean(list(delta.values()))), 'pool_bootstrap_95': pool_bootstrap(delta, table), 'per_fold': per_fold,
                                 'by_true_family': {f: float(np.mean([d for k, d in delta.items() if fam[k] == f])) for f in FAMILIES}, 'halves': halves,
                                 'pairs_improved_worsened': [int(sum(d > 1e-12 for d in delta.values())), int(sum(d < -1e-12 for d in delta.values()))],
                                 'map5_pooled': float(j['hyb'].mean()), 'base_map5_pooled': float(j['map5'].mean()), 'n_pairs': j.height, 'predicted_CI_pairs_kept_struct': int((j['predicted'] == CI).sum())}
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['variants'][name].items() if k not in ('by_true_family', 'halves', 'per_fold')}, 'per_fold', {k: round(v, 4) for k, v in per_fold.items()}, 'fam', {k: round(v, 4) for k, v in out['variants'][name]['by_true_family'].items()}, 'halves', {k: round(v, 4) for k, v in halves.items()})
    write_json(OUT / 'posthoc_hybrid.json', out)


if __name__ == '__main__':
    main()
