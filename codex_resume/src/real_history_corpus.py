"""Build label-free prior-player-history corpus under frozen study exclusions."""
from audit_common import *
import hashlib,time,math
from datetime import datetime,timezone
from sklearn.model_selection import GroupKFold
from contestable_state_features import target_keys

OUT=HERE/'research/real_history_encoder_20260912'
PROTOCOL=HERE/'research/real_history_pretraining_protocol_20260912.md'
FIELDS=['signed_log_net_bb','log_contribution_bb','folded','showdown','won','log_actions','fold_fraction','check_fraction','call_fraction','aggressive_fraction']

def token_matrix(frame):
    bb=frame['big_blind'].to_numpy().astype(float)
    net=frame['net_chips'].to_numpy().astype(float)
    contribution=frame['total_contribution'].to_numpy().astype(float)
    n=frame['_actions'].to_numpy().astype(float)
    assert np.all(bb>0) and np.all(contribution>=0) and np.all(n>=0)
    rates=np.column_stack([frame[f'_a{k}'].to_numpy()/np.maximum(n,1) for k in range(4)])
    assert np.allclose(rates.sum(1),(n>0).astype(float),rtol=0,atol=1e-12)
    v=np.column_stack([np.sign(net)*np.log1p(np.abs(net)/bb)/5,np.log1p(contribution/bb)/5,frame['folded'].to_numpy(),frame['went_to_showdown'].to_numpy(),frame['won_share'].to_numpy()>0,np.log1p(n)/3,rates]).astype(np.float32)
    assert v.shape==(frame.height,10) and np.isfinite(v).all()
    y=np.column_stack([rates,frame['folded'].to_numpy(),frame['went_to_showdown'].to_numpy()]).astype(np.float32)
    return v,y,n

def window(v,i):
    assert 0<=i<len(v)
    n=min(20,i);w=np.zeros((20,10),dtype=np.float32)
    if n:w[:n]=v[i-n:i]
    return w,n

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'corpus-freeze.json').exists(),'write-once corpus'
    d,shared,rcols,saved=load_current();keys=target_keys();assert keys.height==3240
    outer=set().union(*(set(fold_pools(f)) for f in (0,1,2)))
    rawhands=pl.scan_parquet(ROOT/'data/hands.parquet').filter(pl.col('phase')=='development')
    allpools=set(rawhands.select(pl.col('table_id').unique()).collect()['table_id'])
    allowed=allpools-outer;assert not set(d['table_id'])&outer
    exclusion={k:set() for k in range(5)};design=[]
    for fam in FAMILIES:
        q=d.filter((pl.col('behavior_family')==fam)&(pl.col('oof_rank')<=20))
        g=q['table_id'].to_numpy()
        for k,(tr,te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((q.height,1)),q['relevance'].to_numpy(),g)):
            held=set(g[te]);train=set(g[tr]);assert not train&held
            exclusion[k].update(held)
            design.append({'family':fam,'fold':k,'training_pools':sorted(train),'held_pools':sorted(held),'training_pairs':sorted(q[tr]['pair_id'].unique()),'held_pairs':sorted(q[te]['pair_id'].unique())})
    pmapframe=pl.read_csv(ROOT/'data/development_labels.csv',columns=['pair_id','player_1','player_2']).join(keys.select('pair_id').unique(),on='pair_id',how='semi')
    pmap={pid:(a,b) for pid,a,b in pmapframe.iter_rows()}
    need={}
    for row,(pid,hid) in enumerate(keys.iter_rows()):
        for member,player in enumerate(pmap[pid]):need.setdefault(player,{}).setdefault(hid,[]).append((row,member,pid))
    keys.write_parquet(OUT/'target-keys.parquet')
    sources=[OUT/'target-keys.parquet',Path(__file__),HERE/'src/audit_common.py',HERE/'src/contestable_state_features.py',PROTOCOL,PREV/'scores.parquet',PREV/'predictions-frozen.json',*[ROOT/'data'/n for n in ('hands.parquet','seats.parquet','actions.parquet','development_labels.csv')]]
    write(OUT/'corpus-freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),'fields':FIELDS,'allowed_pools':sorted(allowed),'outer_pools':sorted(outer),'encoder_exclusions':{str(k):sorted(v) for k,v in exclusion.items()},'detector_design':design,'source_hashes':{str(p):sha(p) for p in sources},'history_length':20,'samples_per_player':16,'encoder_max_rows':50000,'sequence_seed_prefix':'real-history-94173|','cap_prefix':'real-history-cap-94173|','limits':['Only started_at order, no ended_at.','Raw history/observed behavior targets; no collusion relabeling.','No relevance used in token values, behavior targets or sampling.']})
    hands=rawhands.filter(pl.col('table_id').is_in(sorted(allowed))).select('hand_id','table_id',pl.col('started_at').cast(pl.Int64),'big_blind').collect()
    actions=pl.scan_parquet(ROOT/'data/actions.parquet').join(hands.lazy().select('hand_id'),on='hand_id',how='inner')
    kind=(pl.when(pl.col('action')=='fold').then(0).when(pl.col('action')=='check').then(1).when((pl.col('action')=='call')|((pl.col('action')=='all_in')&(pl.col('amount')<=pl.col('to_call')))).then(2).otherwise(3))
    assert set(actions.select(pl.col('action').unique()).collect()['action'])<=set(['fold','check','call','all_in','bet','raise'])
    counts=actions.with_columns(kind.alias('_kind')).group_by('hand_id','player_id').agg(pl.len().alias('_actions'),*[(pl.col('_kind')==k).sum().alias(f'_a{k}') for k in range(4)]).collect()
    records=(pl.scan_parquet(ROOT/'data/seats.parquet').select('hand_id','player_id','net_chips','total_contribution','folded','went_to_showdown','won_share').join(hands.lazy(),on='hand_id',how='inner').join(counts.lazy(),on=['hand_id','player_id'],how='left').with_columns(pl.col(['_actions',*[f'_a{k}' for k in range(4)]]).fill_null(0)).sort('player_id','started_at','hand_id').collect())
    del hands,counts
    assert records.select('player_id','hand_id').n_unique()==records.height
    assert records.select('player_id','started_at').n_unique()==records.height
    assert records.group_by('player_id').agg(pl.col('table_id').n_unique().alias('n'))['n'].max()==1
    targetx=np.zeros((keys.height,2,20,10),dtype=np.float32);targetn=np.zeros((keys.height,2),dtype=np.int64);seen=np.zeros((keys.height,2),dtype=int)
    targetmeta=[];cx=[];cn=[];cy=[];cmeta=[];fixture_count=0;players=0
    for (player,),q in records.group_by('player_id',maintain_order=True):
        players+=1;times=q['started_at'].to_numpy();assert np.all(np.diff(times)>0)
        v,y,n=token_matrix(q);ids=q['hand_id'].to_list();pool=q['table_id'][0]
        eligible=np.flatnonzero((np.arange(q.height)>0)&(n>0))
        seed=int.from_bytes(hashlib.sha256(('real-history-94173|'+player).encode()).digest()[:8],'big')
        rng=np.random.default_rng(seed)
        selected=np.sort(rng.choice(eligible,size=min(16,len(eligible)),replace=False))
        for i in selected:
            w,l=window(v,int(i));cx.append(w);cn.append(l);cy.append(y[i])
            priority=hashlib.sha256(('real-history-cap-94173|'+player+'|'+ids[i]).encode()).hexdigest()
            cmeta.append((player,ids[i],pool,priority,int(times[i]),int(times[i-l]),int(times[i-1])))
            assert times[i-1]<times[i]
        for i,hid in enumerate(ids):
            for row,member,pid in need.get(player,{}).get(hid,[]):
                w,l=window(v,i);targetx[row,member]=w;targetn[row,member]=l;seen[row,member]+=1
                assert l==0 or np.all(times[i-l:i]<times[i])
                targetmeta.append((pid,hid,player,member,pool,i,l,int(times[i]),None if l==0 else int(times[i-1])))
                if fixture_count<20:
                    changed=v.copy();changed[i:]+=31.;assert np.array_equal(window(changed,i)[0],w)
                    assert np.array_equal(w[:l],v[max(0,i-20):i])
                    # Same currency rescaling leaves both money features exactly unchanged.
                    scaled=q.with_columns(pl.col('big_blind')*2,pl.col('net_chips')*2,pl.col('total_contribution')*2)
                    sv,_,_=token_matrix(scaled);assert np.array_equal(sv,v)
                    fixture_count+=1
    assert np.all(seen==1) and fixture_count==20
    corpus=np.stack(cx);lengths=np.asarray(cn,dtype=np.int64);targets=np.stack(cy)
    meta=pl.DataFrame(cmeta,schema=['player_id','hand_id','table_id','priority','focal_time','first_history_time','last_history_time'],orient='row')
    assert meta.select('player_id','hand_id').n_unique()==meta.height==len(corpus)
    assert not set(meta['table_id'])&outer and meta.group_by('player_id').len()['len'].max()<=16
    np.savez_compressed(OUT/'train-corpus.npz',x=corpus,lengths=lengths,y=targets,pool=np.asarray(meta['table_id'].to_list(),dtype=str),priority=np.asarray(meta['priority'].to_list(),dtype=str))
    np.savez_compressed(OUT/'target-history.npz',x=targetx,lengths=targetn,pair_id=np.asarray(keys['pair_id'].to_list(),dtype=str),hand_id=np.asarray(keys['hand_id'].to_list(),dtype=str))
    meta.write_parquet(OUT/'train-keys.parquet');pl.DataFrame(targetmeta,schema=['pair_id','hand_id','player_id','member','table_id','prior_count','length','focal_time','last_history_time'],orient='row').sort('pair_id','hand_id','member').write_parquet(OUT/'target-members.parquet')
    means=targetx.sum(axis=2)/np.maximum(targetn[:,:,None],1)
    m=np.concatenate([means,targetn[:,:,None]/20],axis=2)
    simple=np.concatenate([(m[:,0]+m[:,1])/2,np.abs(m[:,0]-m[:,1])],axis=1)
    swapped=np.concatenate([(m[:,1]+m[:,0])/2,np.abs(m[:,1]-m[:,0])],axis=1);assert np.array_equal(simple,swapped)
    cols=[f'rh_mean_{i}' for i in range(22)]
    keys.with_columns(*[pl.Series(name,simple[:,i]) for i,name in enumerate(cols)]).write_parquet(OUT/'history_mean.parquet')
    files=['train-corpus.npz','target-history.npz','train-keys.parquet','target-members.parquet','history_mean.parquet']
    result={'status':'complete','created_at':datetime.now(timezone.utc).isoformat(),'corpus_rows':len(corpus),'source_player_hand_rows':records.height,'players':players,'target_pair_hands':keys.height,'target_members':len(targetmeta),'minimum_history':int(lengths.min()),'maximum_history':int(lengths.max()),'no_player_time_ties':True,'max_pools_per_player':1,'future_suffix_scale_fixtures':fixture_count,'pair_swap_exact':True,'token_fields':FIELDS,'simple_features':cols,'corpus_freeze_sha256':sha(OUT/'corpus-freeze.json'),'artifacts':{name:sha(OUT/name) for name in files},'seconds':time.perf_counter()-start,'no_outer_evidence_truth_read':True}
    for path,digest in json.loads((OUT/'corpus-freeze.json').read_text(encoding='utf-8'))['source_hashes'].items():assert sha(path)==digest
    write(OUT/'corpus-receipt.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
