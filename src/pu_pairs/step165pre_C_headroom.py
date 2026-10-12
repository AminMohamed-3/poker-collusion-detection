"""Step 165 pre-check (Claude): descriptive pair-side headroom on the STEP131 dev surfaces (folds 1-2, prefix-40/60/80, shared_hands >= 38)
under the current production recipe (block risk order + tail FINAL2_16 key). No fits and no decision; development labels used descriptively.
Oracles (each keeps the value multiset): TAIL = positives first inside the tail (risk < .5); BLOCK = positives first inside the block;
BOTH = both; SPLIT = the best possible assignment of which pairs sit in the block (all positives ranked on top, block size unchanged).
Metrics as step159c: PP90, PPcls, RAW AP and behaviour MAP B. Output experiments/headroom_20260915/headroom165.json"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
import numpy as np, polars as pl
from pu_pairs import step159c_C_combine_dev as S159

E = C + 'experiments/'
X, VD = E + 'pu_pairs_20260911/', E + 'viewdis_20260911/'
S159.DIRS['ARCH2'] = (E + 'arch2_views_20260915/', 'arch2_dev_fold')
VT, NEW_VT, A2 = S159.VT, S159.NEW_VT, ['MLP2', 'HGB5']


def main():
    dest = E + 'headroom_20260915/headroom165.json'; os.makedirs(os.path.dirname(dest), exist_ok=True); assert not os.path.exists(dest)
    V = S159.type_means('SXN3', VT).join(S159.type_means('ARCH', NEW_VT), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    V = V.join(S159.type_means('ARCH2', A2), on=['pair_id', 'fold', 'surface'], how='inner', validate='1:1')
    base = [f'SXN3_{v}' for v in VT] + [f'ARCH_{v}' for v in NEW_VT] + [f'ARCH2_{v}' for v in A2]
    V = V.with_columns(pl.mean_horizontal(base).alias('key'))
    R1 = pl.read_parquet(VD + 'view_dev.parquet', columns=['pair_id', 'R1']).rename({'R1': 'R1_full'})
    out = {}
    for p in ('prefix40', 'prefix60', 'prefix80'):
        parts = [pl.read_parquet(X + f'fold{f}-{p}/scores.parquet', columns=['pair_id', 'table_id', 'label', 'behavior_family', 'predicted_class', 'shared_hands', 'risk'])
                 .join(pl.read_parquet(X + f'fold{f}/scores.parquet', columns=['pair_id', 'risk']).rename({'risk': 'risk_full'}), on='pair_id', how='left')
                 .with_columns(pl.lit(f).alias('fold')) for f in (1, 2)]
        d = (pl.concat(parts).join(V.filter(pl.col('surface') == p).drop('surface'), on=['pair_id', 'fold'], how='left')
             .unique(subset=['pair_id'], keep='first', maintain_order=True).filter(pl.col('shared_hands') >= 38).join(R1, on='pair_id', how='left').sort('pair_id'))
        s, lab = d['risk'].to_numpy().astype(float), d['label'].to_numpy()
        fam, pred = d['behavior_family'].to_numpy(), d['predicted_class'].to_numpy()
        yr, unl, rf, r1 = (lab == 1).astype(int), lab == -1, d['risk_full'].fill_null(0).to_numpy().astype(float), d['R1_full'].fill_null(False).to_numpy()
        yp = (yr.astype(bool) | (unl & (rf >= .9))).astype(int); yc = (yr.astype(bool) | (unl & (rf >= .9) & ~r1)).astype(int)
        T = s < .5; B = ~T
        cur = S159.permute(s, d['key'].to_numpy(), T)
        rank_cur = np.argsort(np.argsort(-cur, kind='mergesort'), kind='mergesort') / len(cur)  # 0 = top

        def metrics(sc):
            return {'PP90': S159.ap(yp, sc), 'PPcls': S159.ap(yc, sc), 'RAW': S159.ap(yr, sc), 'B': S159.bmap(sc, lab, fam, pred)}

        res = {'n': len(s), 'n_block': int(B.sum()), 'pos_raw': int(yr.sum()), 'pos_raw_in_block': int((yr[B] == 1).sum()), 'pos_pp90': int(yp.sum()), 'pos_pp90_in_block': int((yp[B] == 1).sum()),
               'current': metrics(cur)}
        for yname, y in (('PP90', yp), ('RAW', yr)):
            z = y * 10.0 + (1 - rank_cur)  # positives first, otherwise current order
            tail_o = S159.permute(cur, z, T); block_o = S159.permute(cur, z, B); both = S159.permute(block_o, z, T)
            split = np.sort(cur)[::-1][np.argsort(np.argsort(-z, kind='mergesort'), kind='mergesort')]
            m = {'TAIL': metrics(tail_o)[yname], 'BLOCK': metrics(block_o)[yname], 'BOTH': metrics(both)[yname], 'SPLIT_all_on_top': metrics(split)[yname]}
            res[f'oracle_{yname}'] = {k: {'value': v, 'gain': v - res['current'][yname]} for k, v in m.items()}
        # where the tail positives sit (current order): rank percentile among all rows
        tp = np.where(T & (yr == 1))[0]
        res['tail_raw_pos_rank_pct'] = [round(float(x), 4) for x in np.sort(rank_cur[tp])]
        bp = np.where(B & (yr == 0) & (yp == 0))[0]
        res['block_non_pos_count'] = int(len(bp))
        # family errors among raw positives in the block
        FAMC = {'directed_transfer': 1, 'soft_play': 2, 'coordinated_isolation': 3}
        bpos = np.where(B & (yr == 1))[0]
        res['block_raw_pos_family_wrong'] = int(sum(1 for i in bpos if FAMC.get(fam[i], -9) != pred[i]))
        allpos = np.where(yr == 1)[0]
        res['raw_pos_family_wrong_all'] = int(sum(1 for i in allpos if FAMC.get(fam[i], -9) != pred[i]))
        out[p] = res
        print(p, json.dumps({k: v for k, v in res.items() if k != 'tail_raw_pos_rank_pct'}), flush=True)
    json.dump(out, open(dest, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
