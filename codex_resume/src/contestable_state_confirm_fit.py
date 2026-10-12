"""Frozen nested confirmation of the contestable-state evidence feature recipe."""
from audit_common import *
import time
from datetime import datetime, timezone
import lightgbm as lgb
from evidence_aug.frame import training_relevance, attach_relevance, family_columns, score
from evidence_aug.fold import fold_roles
from evidence_aug.step1_reproduce_baseline import fit_ranker
from evidence_aug.step10_chronology_fit import add_chronology, OUT as CHRONO_OUT
from evidence_aug.step14_audit_truncation import split_of, SPLITS
from contestable_state_confirmation_features import build_for_keys
from contestable_state_features import FEATURES as STATE_FEATURES

OUT=HERE/'experiments/contestable_state_confirmation_20260912'
STRUCT_FILE=CV/'experiments/family_stage2_20260911/dev-structural-features.parquet'
OLD=CV/'experiments/family_stage2_r382_20260911'
AUDIT_DIR=HERE/'experiments/contestable_state_20260912'
PRIMARY='contestable_state'

def ordinal(frame,values):
    return frame.select('pair_id').with_columns(pl.Series('s',values)).select(pl.col('s').rank('ordinal',descending=True).over('pair_id'))['s'].to_numpy()

def main():
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); (OUT/'scores').mkdir(exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Confirmation already frozen'
    audit=json.loads((AUDIT_DIR/'results.json').read_text(encoding='utf-8'))
    assert audit['primary_eligible'] and audit['negative_control_rejected']
    protected=check_protected()
    source_files=[HERE/'src'/name for name in ('audit_common.py','contestable_state_features.py','contestable_state_confirmation_features.py','contestable_state_audit.py','contestable_state_confirm_fit.py','contestable_state_confirm_evaluate.py')]
    source_files += [AUDIT_DIR/'freeze.json',AUDIT_DIR/'results.json',STRUCT_FILE,HERE/'research/contestable_state_20260912/wrapper-parity.json']
    best=CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv'
    write(OUT/'freeze.json',{
        'created_at':datetime.now(timezone.utc).isoformat(),'candidate':PRIMARY,
        'source_hashes':{str(p):sha(p) for p in source_files},'protected':protected,'best_sha256':sha(best),
        'audit_result':audit['comparisons'][PRIMARY], 'new_features':STATE_FEATURES,'params':dict(CLF,random_state=11),
        'selection':'One entire29feature recipe added to every family; no selection of subsets, family routing, weights, cutoffs or other arms.',
        'confirmation_folds':[0,1,2],
        'reference':'Complete current383chronology stage1 and shipping hybrid stage2: r382 for predicted DT/SP and structural for predicted CI; predicted none retains stage1. Exact first-stage score parity, stage2 tolerance2e-14 and exact ordinal ranks required.',
        'fit':'For every outerfold, held pools excluded from both model stages. Five pool-hash inner OOF partitions supply stage2 training scores. Public chip-accounting features use only each observed hand and no learned preprocessing. Held routing uses fixed predicted families.',
        'gate':{'primary_mean_unique_pair_delta_min':.01,'raw_pool_bootstrap_lower_gt':0,'all_three_fold_deltas_nonnegative':True,'pooled_true_family_floor':-.01,'each_fold_true_family_floor':-.01,'both_pool_halves_nonnegative':True},
        'pooling':'Average each unique pair across repeated held appearances; bootstrap original pool IDs, not fold:pool IDs. Fold-appearance means descriptive only.',
        'limits':['Repeatedly reused outer partitions, not pristine; audit feature choice is adaptive.','Inner stacking shares reference-policy features fit outside outerpool; not fully nested at inner policy level.','No hidden fourth-family evidence truth available.','State features are public chip-accounting proxies, not strategic expected values.'],
        'negative_control':'Stage1-only must fail against complete incumbent.',
        'progression':'If pass, preserve this qualified evidence component and assess faithful production/incremental combination; a local+.03393evidence gain alone predicts only+.00679composite, below user2-3pointsubmission milestone. If fail, archive with no family/subset/weight rescue.',
        'held_evidence_evaluation_before_freeze':False})
    structural=pl.read_parquet(STRUCT_FILE)
    struct=[c for c in structural.columns if c not in ('pair_id','hand_id')]
    players=pl.read_csv(ROOT/'data/development_labels.csv',columns=['pair_id','player_1','player_2'])
    receipts=[]
    for fold in (0,1,2):
        data,features,pools,identity=load_fold(fold)
        data=add_chronology(data).join(structural,on=['pair_id','hand_id'],validate='1:1').sort('pair_id','hand_id')
        held=data.filter(pl.col('table_id').is_in(pools)).sort('pair_id','hand_id')
        train=data.filter(~pl.col('table_id').is_in(pools)).sort('pair_id','hand_id')
        evidence=training_relevance(set(train['table_id']))
        assert not set(evidence['pair_id'])&set(held['pair_id'])
        train=attach_relevance(train,evidence).sort('pair_id','hand_id')
        roles=fold_roles(fold); held=held.join(roles,on='pair_id',validate='m:1').sort('pair_id','hand_id')
        model=fit_ranker(train,features+['chrono_frac'])
        sh=score(model,family_columns(held,'predicted_behavior'),features+['chrono_frac'])
        old=pl.read_parquet(CHRONO_OUT/'scores'/f'fold{fold}-chrono1.parquet')
        assert held.select('pair_id','hand_id').equals(old.select('pair_id','hand_id'))
        assert np.array_equal(sh,old['score'].to_numpy()),'Stage1 reference mismatch'
        model.booster_.save_model(str(OUT/f'fold{fold}-stage1.txt'))
        split=np.asarray([split_of(p) for p in train['table_id']]); st=np.full(train.height,np.nan)
        for s in range(SPLITS):
            te=np.flatnonzero(split==s)
            inner=fit_ranker(train.filter(pl.Series(split!=s)),features+['chrono_frac'])
            st[te]=score(inner,family_columns(train[te],'behavior_family'),features+['chrono_frac'])
        assert np.isfinite(st).all()
        train=train.with_columns(pl.Series('oof',st)); held=held.with_columns(pl.Series('oof',sh))
        for frame in (train,held):
            assert np.array_equal(frame['pot_bb'].to_numpy().astype(np.float32),frame['pot_bb_right'].to_numpy().astype(np.float32))
        train=train.with_columns(pl.col('pot_bb_right').alias('pot_bb'))
        held=held.with_columns(pl.col('pot_bb_right').alias('pot_bb'))
        train=train.with_columns(pl.col('oof').rank('ordinal',descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        held=held.with_columns(pl.col('oof').rank('ordinal',descending=True).over('pair_id').cast(pl.Float64).alias('oof_rank'))
        shared=struct+['chrono_rank','chrono_frac','n_hands','oof','oof_rank']
        extra=[f for f in features if not f.startswith('family_') and f not in struct]
        train=train.with_columns(pl.col(extra).fill_null(-999.)); held=held.with_columns(pl.col(extra).fill_null(-999.))
        old_struct=pl.read_parquet(OLD/'scores'/f'fold{fold}-allfam_top20_struct.parquet')
        old_r382=pl.read_parquet(OLD/'scores'/f'fold{fold}-allfam_top20_r382.parquet')
        held_keys=held.select('pair_id','hand_id')
        assert held_keys.equals(old_struct.select('pair_id','hand_id')) and held_keys.equals(old_r382.select('pair_id','hand_id'))
        expected=np.where((held['predicted_behavior']=='coordinated_isolation').to_numpy(),old_struct['score'].to_numpy(),old_r382['score'].to_numpy())
        control=sh.copy(); model_dir=OUT/'models'/f'fold{fold}'; model_dir.mkdir(parents=True,exist_ok=True)
        for fam in FAMILIES:
            cols=shared+(extra if fam!='coordinated_isolation' else [])
            tr=train.filter((pl.col('behavior_family')==fam)&(pl.col('oof_rank')<=20))
            ix=np.flatnonzero(((held['predicted_behavior']==fam)&(held['oof_rank']<=20)).to_numpy())
            ref=lgb.LGBMClassifier(**CLF,random_state=11).fit(tr.select(cols).to_numpy().astype(float),tr['relevance'].to_numpy())
            ref.booster_.save_model(str(model_dir/f'{fam}-incumbent.txt'))
            if len(ix): control[ix]=ref.predict_proba(held[ix].select(cols).to_numpy().astype(float))[:,1]+10
            tail=((held['predicted_behavior']==fam)&(held['oof_rank']>20)).to_numpy(); control[tail]=sh[tail]-1e6
        diff=float(np.max(np.abs(control-expected)))
        assert diff<=2e-14 and np.array_equal(ordinal(held,control),ordinal(held,expected)),'Complete reference mismatch'
        write(OUT/f'fold{fold}-reference-preflight.json',{'stage1_exact':True,'stage2_max_abs_diff':diff,'ordinal_ranks_exact':True})
        print('Reference passed',fold,round(time.perf_counter()-start,1),flush=True)
        requested=pl.concat([train.filter(pl.col('oof_rank')<=20).select('pair_id','hand_id'),held.filter(pl.col('oof_rank')<=20).select('pair_id','hand_id')]).unique().sort('pair_id','hand_id')
        new,accounting=build_for_keys(requested,players)
        assert new.select('pair_id','hand_id').equals(requested)
        new.write_parquet(OUT/f'fold{fold}-state-features.parquet'); write(OUT/f'fold{fold}-accounting.json',accounting)
        train_keys=train.select('pair_id','hand_id')
        train=train.join(new,on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','hand_id')
        held=held.join(new,on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','hand_id')
        assert train_keys.equals(train.select('pair_id','hand_id')) and held_keys.equals(held.select('pair_id','hand_id'))
        candidate=sh.copy(); model_receipts=[]
        for fam in FAMILIES:
            tr=train.filter((pl.col('behavior_family')==fam)&(pl.col('oof_rank')<=20))
            ix=np.flatnonzero(((held['predicted_behavior']==fam)&(held['oof_rank']<=20)).to_numpy())
            cols=shared+(extra if fam!='coordinated_isolation' else [])+STATE_FEATURES
            x=tr.select(cols).to_numpy().astype(float); assert np.isfinite(x).all()
            fitted=lgb.LGBMClassifier(**CLF,random_state=11).fit(x,tr['relevance'].to_numpy())
            path=model_dir/f'{fam}-{PRIMARY}.txt'; fitted.booster_.save_model(str(path))
            if len(ix):
                h=held[ix].select(cols).to_numpy().astype(float); assert np.isfinite(h).all()
                p=fitted.predict_proba(h)[:,1]
                replay=lgb.Booster(model_file=str(path)).predict(h,num_threads=2)
                assert np.array_equal(p,replay),'Saved candidate inference mismatch'
                candidate[ix]=p+10
            tail=((held['predicted_behavior']==fam)&(held['oof_rank']>20)).to_numpy(); candidate[tail]=sh[tail]-1e6
            model_receipts.append({'family':fam,'columns':cols,'train_queries':tr['pair_id'].n_unique(),'held_queries':held[ix]['pair_id'].n_unique(),'model_sha256':sha(path),'reload_exact':True})
        write(model_dir/'recipe.json',model_receipts)
        for name,p in [('incumbent',expected),(PRIMARY,candidate),('stage1_only',sh)]:
            assert np.isfinite(p).all()
            held.select('pair_id','hand_id','table_id').with_columns(pl.Series('score',p)).write_parquet(OUT/'scores'/f'fold{fold}-{name}.parquet')
        roles.write_csv(OUT/'scores'/f'fold{fold}-roles.csv')
        receipts.append({'fold':fold,'identity':identity,'held_pairs':held['pair_id'].n_unique(),'training_pairs':train['pair_id'].n_unique(),'reference_max_abs_diff':diff,'accounting':accounting})
        print('Finished fold',fold,round(time.perf_counter()-start,1),flush=True)
    assert sha(best)==json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))['best_sha256']
    write(OUT/'predictions-frozen.json',{'created_at':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(OUT/'freeze.json'),'scores':{f.name:sha(f) for f in sorted((OUT/'scores').iterdir())},'fold_receipts':receipts,'protected_unchanged':check_protected()==protected,'seconds':time.perf_counter()-start,'held_evidence_evaluation':False})
    print('PREDICTIONS FROZEN',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__': main()
