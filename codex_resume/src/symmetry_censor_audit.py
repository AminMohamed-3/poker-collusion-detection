"""Frozen inexpensive audit: pair-exchange invariance and censored-label weighting.

Writes only inside Codex Resume. Reads cached audit relevance for 162 target pairs
outside every outer holdout; never reads outer evidence truth or evaluation data.
"""
import os
for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import hashlib
import time
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CV = ROOT / 'Claude Version'
sys.path.insert(0, str(CV / 'src'))
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from evidence_aug.paths import check_protected, sha
from evidence_aug.fold import load_fold, fold_pools
from evidence_aug.frame import family_columns, score as rank_score, FAMILIES
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step14_audit_truncation import OUT as AUDIT, CLF, split_of
from evidence_aug.step65_structural_evidence import summarize
from evidence_aug.step65b_family_stage2 import boot
from evidence_aug.step5_evaluate import half

OUT = HERE / 'experiments' / 'symmetry_censor_20260912'
ARMS = ['incumbent', 'swap_diagnostic', 'swap_average', 'symmetric_training',
        'censor_half', 'censor_quarter', 'symmetric_censor_half']

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def swapped(x, columns):
    order = list(range(len(columns)))
    for i, col in enumerate(columns):
        if col.startswith(('A_', 'B_')):
            other = ('B_' if col.startswith('A_') else 'A_') + col[2:]
            assert other in columns, col
            order[i] = columns.index(other)
    return x[:, order]

def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'freeze.json').exists(), 'Write-once study already started.'
    protected = check_protected()
    best = CV / 'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
    sref = CV / 'outputs_stage2_r382_20260911/submission_S_stage2_r382DTSP.csv'
    sources = [Path(__file__), AUDIT / 'audit-oof.parquet', AUDIT / 'audit-structural-features.parquet', best, sref]
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(),
              'arms': ARMS, 'primary': 'symmetric_training',
              'sources': {str(p): sha(p) for p in sources}, 'protected': protected,
              'reference': 'Shipping hybrid evidence design on CURRENT chrono_frac stage-1 OOF; risk unchanged.',
              'data': '162 audit pairs outside union of outer0/1/2; stage2 family GroupKFold5; stage1 pool-hash5.',
              'model': dict(CLF, random_state=11), 'seed_note': 'No subsampling: seeds11/12/13 identical, fit once.',
              'stage1': 'Reproduce base382 OOF exactly; compute chrono383 OOF once with original fixed recipe.',
              'censor_rule': 'Only for queries with >=5 listed hands, unlisted hands later than latest listed receive .5 or .25 weight. Query total weight normalized to original20. Labels unchanged; training weights only.',
              'symmetry_rule': 'Exchange all A_/B_ structural fields. Same inference average for both orientations; training expansion half weights and doubled min_child_samples preserve mass and approximate leaf-count regularization.',
              'gate': {'mean_map5_gain_min': .01, 'bootstrap_lower_gt': 0,
                       'family_loss_floor': -.01, 'both_pool_halves_nonnegative': True,
                       'negative_control': 'Known-worse stage1-only must fail vs complete chain; orientation-swap involution exact.'},
              'progression': 'Report all arms. Passing primary first, otherwise a secondary is explicitly selected among frozen arms and needs separate confirmation. No outer read in this study.',
              'limitations': ['Reused training-only audit; no pristine claim.', 'Stage2 crossfits share stage1 OOF cache; exploratory, not nested.', 'Censor process only a training approximation.', 'True-family audit routing; confirmation predicted-family only.']}
    write(OUT / 'freeze.json', freeze)
    audit = pl.read_parquet(AUDIT / 'audit-oof.parquet').sort('pair_id','hand_id')
    structf = pl.read_parquet(AUDIT / 'audit-structural-features.parquet')
    struct = [c for c in structf.columns if c not in ('pair_id','hand_id')]
    outer = set().union(*(set(fold_pools(f)) for f in (0,1,2)))
    assert not set(audit['table_id']) & outer and audit['pair_id'].n_unique() == 162
    print('Training-only audit:', audit.shape, audit.select('pair_id','behavior_family').unique().group_by('behavior_family').len().to_dicts(), flush=True)
    data, features, _, identity = load_fold(0)
    # Discard held feature rows immediately; load_fold never loads evidence truth.
    data = data.join(audit.select('pair_id','hand_id'), on=['pair_id','hand_id'], how='semi')
    d = data.join(audit.select('pair_id','hand_id','relevance','chrono_rank','chrono_frac','n_hands','oof'), on=['pair_id','hand_id'], validate='1:1').sort('pair_id','hand_id')
    splits = np.array([split_of(x) for x in d['table_id']])
    chrono = None
    for name, cols in [('base382', features), ('chrono383', features + ['chrono_frac'])]:
        pred = np.full(d.height, np.nan)
        for s in range(5):
            train = d.filter(pl.Series(splits != s))
            ix = np.flatnonzero(splits == s)
            model = fit_ranker(train, cols)
            pred[ix] = rank_score(model, family_columns(d[ix], 'behavior_family'), cols)
        assert np.isfinite(pred).all()
        if name == 'base382':
            assert np.array_equal(pred, d['oof'].to_numpy()), 'Original OOF not reproduced'
        else:
            chrono = pred
        print('Stage1', name, 'complete; base parity exact' if name=='base382' else 'chrono computed', flush=True)
    d = d.with_columns(pl.Series('oof', chrono)).with_columns(pl.col('oof').rank('ordinal',descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
    # Structural pot_bb replaces same-name raw pot exactly as the shipping builder does.
    d = d.drop([c for c in struct if c in d.columns]).join(structf, on=['pair_id','hand_id'], validate='1:1')
    rcols = [f for f in features if not f.startswith('family_') and f not in struct]
    shared = struct + ['chrono_rank','chrono_frac','n_hands','oof','oof_rank']
    d = d.with_columns(pl.col(rcols).fill_null(-999.0))
    truths = {k[0]:set(g['hand_id']) for k,g in d.filter(pl.col('relevance')==1).group_by('pair_id')}
    families = dict(d.select('pair_id','behavior_family').unique().iter_rows())
    ptab = dict(d.select('pair_id','table_id').unique().iter_rows()); pools=sorted(set(ptab.values()))
    meta = d.group_by('pair_id').agg(pl.col('relevance').sum().alias('listed_count'), pl.col('chrono_rank').filter(pl.col('relevance')==1).max().alias('last_listed'))
    d = d.join(meta,on='pair_id',validate='m:1').sort('pair_id','hand_id')
    top = (d['oof_rank']<=20).to_numpy()
    pred = {name: d['oof'].to_numpy().copy()-1e6 for name in ARMS}
    receipts=[]
    for fam in FAMILIES:
        mask=((d['behavior_family']==fam)&(d['oof_rank']<=20)).to_numpy(); ix=np.flatnonzero(mask)
        dd=d[ix]; cols=shared+(rcols if fam!='coordinated_isolation' else [])
        x=dd.select(cols).to_numpy().astype(float); xs=swapped(x,cols)
        assert np.array_equal(swapped(xs,cols),x,equal_nan=True)
        y=dd['relevance'].to_numpy().astype(int); groups=dd['table_id'].to_numpy()
        cens=((dd['listed_count']>=5)&(dd['relevance']==0)&(dd['chrono_rank']>dd['last_listed'])).to_numpy()
        pid=dd['pair_id'].to_numpy(); receipts.append({'family':fam,'pairs':dd['pair_id'].n_unique(),'rows':len(ix),'censored_rows':int(cens.sum()),'exchange_fields':[c for c in cols if c.startswith(('A_','B_'))]})
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,groups)):
            assert not set(groups[tr]) & set(groups[te])
            base=lgb.LGBMClassifier(**CLF,random_state=11).fit(x[tr],y[tr])
            p=base.predict_proba(x[te])[:,1]; ps=base.predict_proba(xs[te])[:,1]
            pred['incumbent'][ix[te]]=p+10; pred['swap_diagnostic'][ix[te]]=ps+10; pred['swap_average'][ix[te]]=(p+ps)/2+10
            for name, sym, cw in [('symmetric_training',True,1.),('censor_half',False,.5),('censor_quarter',False,.25),('symmetric_censor_half',True,.5)]:
                w=np.where(cens[tr],cw,1.)
                for pair in np.unique(pid[tr]):
                    q=pid[tr]==pair; w[q]*=q.sum()/w[q].sum()
                xx=x[tr]; yy=y[tr]; params=dict(CLF)
                if sym:
                    xx=np.concatenate([xx,xs[tr]]); yy=np.tile(yy,2); w=np.tile(w/2,2)
                    params['min_child_samples']=2*CLF['min_child_samples']
                m=lgb.LGBMClassifier(**params,random_state=11).fit(xx,yy,sample_weight=w)
                pp=m.predict_proba(x[te])[:,1]
                if sym: pp=(pp+m.predict_proba(xs[te])[:,1])/2
                pred[name][ix[te]]=pp+10
        print('Completed stage2 family',fam,flush=True)
    scorefile=OUT/'scores.parquet'
    d.select('pair_id','hand_id','table_id').with_columns(*[pl.Series(k,v) for k,v in pred.items()],pl.Series('stage1_only',d['oof'].to_numpy())).write_parquet(scorefile)
    write(OUT/'predictions-frozen.json',{'scores_sha256':sha(scorefile),'freeze_sha256':sha(OUT/'freeze.json'),'receipts':receipts,'outer_truth_read':False,'seconds':time.perf_counter()-start})
    result={'audit_only':True,'sets':{},'comparisons':{},'passed':[],'reference':'incumbent shipping hybrid architecture with chrono383 shortlist','limitations':freeze['limitations']}
    maps={}; key=d.select('pair_id','hand_id','table_id')
    for name, p in dict(pred,stage1_only=d['oof'].to_numpy()).items():
        summary, maps[name]=summarize(key.with_columns(pl.Series('score',p)),truths,families)
        result['sets'][name]=summary
    for name in [n for n in maps if n!='incumbent']:
        a,b=maps['incumbent'],maps[name]; comp=boot(a,b,ptab,pools)
        comp['by_family']={f:float(np.mean([b[p]-a[p] for p in a if families[p]==f])) for f in FAMILIES}
        comp['halves']={str(h):float(np.mean([b[p]-a[p] for p in a if half(ptab[p])==h])) for h in (0,1)}
        comp['gate_pass']=comp['mean_delta']>=.01 and comp['ci95'][0]>0 and min(comp['by_family'].values())>=-.01 and min(comp['halves'].values())>=0
        result['comparisons'][name]=comp
        if comp['gate_pass'] and name not in ('swap_diagnostic','stage1_only'):result['passed'].append(name)
        print(name,json.dumps(comp),flush=True)
    result['negative_control_rejected']=not result['comparisons']['stage1_only']['gate_pass']
    result['protected_unchanged']=check_protected()==protected
    result['candidate_hashes_unchanged']=all(sha(p)==freeze['sources'][str(p)] for p in (best,sref))
    result['seconds']=time.perf_counter()-start
    write(OUT/'results.json',result)
    print('DONE',json.dumps({'passed':result['passed'],'negative_control_rejected':result['negative_control_rejected'],'seconds':result['seconds']}),flush=True)

if __name__=='__main__': main()
