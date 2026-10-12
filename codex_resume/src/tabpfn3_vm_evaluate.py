"""Assemble hardware-resume predictions, then call unchanged confirmation metrics once."""
import json,hashlib,shutil,argparse,sys
from pathlib import Path
import numpy as np
import polars as pl
HERE=Path(__file__).resolve().parents[1]
ORIGIN=HERE/'experiments/tabpfn3_confirmation_20260912'
REMOTE=HERE/'research/tabpfn3_vm_20260912/hardware-run'
OUT=HERE/'experiments/tabpfn3_confirmation_vm_20260912'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def verify():
    import tabpfn3_confirm_infer as original
    plan,ctx=original.check()
    f=read(REMOTE/'freeze.json')
    assert f['source_sha256']==sha(HERE/'src/tabpfn3_hardware_resume.py')
    assert f['transfer_inventory_sha256']==sha(HERE/'research/tabpfn3_vm_20260912/transfer-inventory.json')
    assert f['origin_freeze_sha256']==sha(ORIGIN/'freeze.json')
    assert f['context_receipt_sha256']==sha(ORIGIN/'context-receipt.json') and f['plan_sha256']==sha(ORIGIN/'plan.json')
    assert f['params']==original.pilot.PARAMS and f['chunk']==256 and f['checkpoint_sha256']==original.pilot.CHECKPOINT_SHA
    assert f['runtime']['required_versions']==read(HERE/'research/tabpfn3_runtime_20260912/runtime-receipt.json')['required_versions']
    b=read(REMOTE/'benchmark.json');assert b['passed'] and b['freeze_sha256']==sha(REMOTE/'freeze.json')
    r=read(REMOTE/'replay.json');assert r['freeze_sha256']==sha(REMOTE/'freeze.json') and r['ranking_exact'] and r['replay_sha256']==sha(REMOTE/'replay.npy')
    done=read(REMOTE/'fit-complete.json');assert done['freeze_sha256']==sha(REMOTE/'freeze.json') and len(done['receipts'])==9
    for m in ctx['contexts']:
        rr=REMOTE/'predictions'/(m['stem']+'.json');pp=REMOTE/'predictions'/(m['stem']+'.npy');dd=REMOTE/'predictions'/(m['stem']+'-diagnostic.json')
        assert sha(rr)==done['receipts'][m['stem']];receipt=read(rr)
        assert receipt['freeze_sha256']==sha(REMOTE/'freeze.json') and receipt['context_sha256']==m['context_sha256']
        assert receipt['probabilities_sha256']==sha(pp) and receipt['diagnostic_sha256']==sha(dd)
        info=receipt['runtime'];assert info==read(dd) and info['memory_pass'] and info['estimators']==8
        assert [info['training_rows'],info['width']]==m['train_shape'] and [info['query_rows'],info['width']]==m['query_shape']
        assert info['memory_ceiling_bytes']==f['runtime']['memory_ceiling_bytes']
        assert info['chunks']==[min(256,m['query_shape'][0]-i) for i in range(0,m['query_shape'][0],256)]
        p=np.load(pp,allow_pickle=False);assert p.shape==(m['query_shape'][0],) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    m=next(m for m in ctx['contexts'] if m['stem']==f['replay_stem']);a=np.load(REMOTE/'replay.npy',allow_pickle=False);b=np.load(REMOTE/'predictions'/(m['stem']+'.npy'),allow_pickle=False)
    with np.load(ORIGIN/m['context_file'],allow_pickle=False) as c:ids=c['held_pair_ids']
    assert all(np.array_equal(np.argsort(-a[ids==x],kind='stable'),np.argsort(-b[ids==x],kind='stable')) for x in np.unique(ids))
    assert bool(np.array_equal(a,b))==r['probabilities_exact'] and float(np.max(np.abs(a-b)))==r['maximum_difference']
    return ctx

def assemble():
    ctx=verify();assert not OUT.exists();OUT.mkdir();(OUT/'scores').mkdir()
    write(OUT/'hardware-provenance.json',{'assembly_source_sha256':sha(Path(__file__)),'original_evaluator_sha256':sha(HERE/'src/tabpfn3_confirm_evaluate.py'),'remote_file_hashes':{str(p):sha(p) for p in REMOTE.rglob('*') if p.is_file()},'origin_failed_log_sha256':sha(ORIGIN/'fit.log'),'source_design_unchanged':True,'no_held_metrics_read':True})
    for name in ['plan.json','context-receipt.json',*[f'fold{f}-roles.csv' for f in (0,1,2)]]:shutil.copyfile(ORIGIN/name,OUT/name)
    for name in ['freeze.json','fit-complete.json','replay.json','replay.npy']:shutil.copyfile(REMOTE/name,OUT/name)
    files={}
    for fold in (0,1,2):
        d=pl.read_parquet(ORIGIN/f'fold{fold}-held.parquet');p=d['oof'].to_numpy().copy();covered=np.zeros(d.height,int)
        for m in (m for m in ctx['contexts'] if m['fold']==fold):
            with np.load(ORIGIN/m['context_file'],allow_pickle=False) as c:
                ix=c['held_indices'];assert np.array_equal(c['held_pair_ids'],d['pair_id'].to_numpy()[ix]);p[ix]=np.load(REMOTE/'predictions'/(m['stem']+'.npy'),allow_pickle=False)+10;covered[ix]+=1
            tail=((d['predicted_behavior']==m['family'])&(d['oof_rank']>20)).to_numpy();p[tail]=d['oof'].to_numpy()[tail]-1e6
        recognized=d['predicted_behavior'].is_in(['directed_transfer','soft_play','coordinated_isolation']).to_numpy();top=recognized&(d['oof_rank']<=20).to_numpy()
        assert np.all(covered[top]==1) and np.all(covered[~top]==0) and np.isfinite(p).all()
        np.testing.assert_array_equal(p[~recognized],d['oof'].to_numpy()[~recognized])
        for name,v in [('incumbent',d['currentcontrol'].to_numpy()),('tabpfn3_default',p),('stage1_only',d['oof'].to_numpy())]:
            path=OUT/'scores'/f'fold{fold}-{name}.parquet';d.select('pair_id','hand_id','table_id').with_columns(pl.Series('score',v)).write_parquet(path);files[path.name]=sha(path)
    write(OUT/'predictions-frozen.json',{'plan_sha256':sha(OUT/'plan.json'),'context_receipt_sha256':sha(OUT/'context-receipt.json'),'freeze_sha256':sha(OUT/'freeze.json'),'replay_sha256':sha(OUT/'replay.json'),'fit_complete_sha256':sha(OUT/'fit-complete.json'),'hardware_provenance_sha256':sha(OUT/'hardware-provenance.json'),'scores':files,'all_predictions_before_held_metrics':True})
    print('Full unchanged confirmation predictions frozen before metrics',flush=True)

def evaluate():
    verify();f=read(OUT/'predictions-frozen.json');p=read(OUT/'hardware-provenance.json')
    assert sha(OUT/'hardware-provenance.json')==f['hardware_provenance_sha256']
    assert p['assembly_source_sha256']==sha(Path(__file__)) and p['original_evaluator_sha256']==sha(HERE/'src/tabpfn3_confirm_evaluate.py')
    for name,h in p['remote_file_hashes'].items():assert sha(name)==h
    import tabpfn3_confirm_evaluate as evaluator
    write(OUT/'evaluation-invocation.json',{'held_evaluation_invocations_this_study':1,'original_evaluator_sha256':sha(Path(evaluator.__file__)),'output_override_only':True})
    evaluator.OUT=OUT
    evaluator.main()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['assemble','evaluate']);arg=a.parse_args();globals()[arg.mode]()
