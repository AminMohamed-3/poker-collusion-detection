import json,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;F=H.parent
T=F/'tabldm_screen';S=F/'screen';A=F/'action_policy_fast'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):
    p=H/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2)+'\n')
def prepare():
    paths=[H/'PROTOCOL.md',H/'study.py',T/'evaluate.py',T/'quality_results.json',T/'predictions.npz',T/'runtime.json',T/'replay_receipt.json',S/'pf35_predictions.npz',A/'fast129_predictions.npz',A/'FAST129_RECEIPT.json',A/'REPLAY_RECEIPT.json',A/'MANIFEST.json']
    paths += [S/f'evaluator_fold{f}.npz' for f in [0,1]]+[T/f'context_fold{f}.npz' for f in [0,1]]+[A/f'fast129_context_fold{f}.npz' for f in [0,1]]
    write('MANIFEST.json',{'sources':{str(p.resolve()):sha(p) for p in paths},'fits':0,'comparisons':1,'source_weights':[4,1],'outer_source_weights':[1,1],'bootstrap_seed':35935,'bootstrap_repeats':1000,'scope':'Reused1860known pairs; current source structure proxy, not fullbase OOF.'})
    print('Frozen provenance; no evaluation.')
def evaluate():
    m=json.loads((H/'MANIFEST.json').read_text())
    for p,v in m['sources'].items():assert sha(Path(p))==v,p
    assert not (H/'RESULTS.json').exists()
    write('EVALUATION_STARTED.json',{'manifest_sha256':sha(H/'MANIFEST.json'),'single_comparison':True})
    spec=importlib.util.spec_from_file_location('fixed_tabldm_metrics',T/'evaluate.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);e.parity()
    y,k,t,fold,old,tab,_=e.load(False)
    q=np.load(A/'fast129_predictions.npz',allow_pickle=False);parts=[]
    r=json.loads((A/'FAST129_RECEIPT.json').read_text());assert sha(A/'fast129_predictions.npz')==r['predictions_sha256']
    for f in [0,1]:
        ctx=np.load(A/f'fast129_context_fold{f}.npz',allow_pickle=False)
        assert np.array_equal(ctx['pair_id'],k[fold==f])
        p=q[f'fold{f}'].astype(np.float64);assert p.shape==(sum(fold==f),4) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-5);parts.append(p)
    aug=np.concatenate(parts);assert len(y)==len(np.unique(k))==1860
    route=1+np.argmax(old[:,1:],1)
    rf=e.rank2(1-old[:,0]);rt=e.rank2(1-tab[:,0]);ra=e.rank2(1-aug[:,0])
    baseline=rf+rt;candidate=e.rank2(4*rf+ra)+rt
    saved=json.loads((T/'quality_results.json').read_text())['surfaces']['known']
    base=e.met(y,baseline,route,k);cand=e.met(y,candidate,route,k)
    for metric,v in base.items():assert abs(v-saved['pooled']['candidate_equal_rank'][metric])<1e-14
    folds={}
    for f in [0,1]:
        ix=fold==f;b=e.met(y[ix],baseline[ix],route[ix],k[ix]);c=e.met(y[ix],candidate[ix],route[ix],k[ix])
        for metric,v in b.items():assert abs(v-saved['folds'][str(f)]['candidate_equal_rank'][metric])<1e-14
        folds[str(f)]={'baseline':b,'candidate':c,'delta_joint':c['joint']-b['joint']}
    rng=np.random.default_rng(35935);tabs,inv=np.unique(t,return_inverse=True);ob=e.prep(y,baseline,route,k);oc=e.prep(y,candidate,route,k);d=[]
    for _ in range(1000):
        count=np.bincount(np.searchsorted(tabs,rng.choice(tabs,len(tabs),replace=True)),minlength=len(tabs));weights=count[inv];d.append(e.rmet(y,weights,oc)-e.rmet(y,weights,ob))
    result={'n':1860,'baseline_saved_parity':True,'baseline':base,'candidate':cand,'delta_joint':cand['joint']-base['joint'],'folds':folds,'bootstrap95':np.quantile(d,[.025,.975]).tolist(),'positive_pooled_and_both_folds':cand['joint']>base['joint'] and all(x['delta_joint']>0 for x in folds.values()),'scope':m['scope'],'new_fits':0,'submission':False}
    write('RESULTS.json',result);print(json.dumps(result))
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','evaluate'];globals()[sys.argv[1]]()
