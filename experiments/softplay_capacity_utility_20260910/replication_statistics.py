"""Equal fit averaging and paired-pool uncertainty; no fit is a new observation."""
import numpy as np
KEYS=['log_loss','pair_ap','sparse_ap','soft_play_ap']
def mean_metrics(rows):
    assert rows
    return {k:float(np.mean([r[k] for r in rows])) for k in KEYS}
def deltas(treatment,baseline):
    return {'logloss_improvement':baseline['log_loss']-treatment['log_loss'],**{k:treatment[k]-baseline[k] for k in KEYS if k!='log_loss'}}
def pool_interval(contrasts,groups,repetitions=2000,seed=20319):
    """contrasts: fit/history-averaged per-pair CE(baseline)-CE(treatment)."""
    contrasts=np.asarray(contrasts);groups=np.asarray(groups);pools=np.unique(groups)
    totals=np.array([contrasts[groups==p].sum() for p in pools]);counts=np.array([(groups==p).sum() for p in pools])
    boot=np.random.default_rng(seed).integers(0,len(pools),(repetitions,len(pools)))
    est=totals[boot].sum(1)/counts[boot].sum(1)
    return np.quantile(est,[.025,.975]).tolist()
def gate(mean,worst_history_ap,halves,by_batch,by_seed):
    checks={'mean_loss_ge_001':mean['logloss_improvement']>=.001,'mean_pair_ap_nonnegative':mean['pair_ap']>=0,'mean_sparse_ap_nonnegative':mean['sparse_ap']>=0,'worst_history_pair_ap_ge_minus001':worst_history_ap>=-.001,'both_pool_halves_nonnegative':min(halves)>=0,'every_batch_loss_positive':min(by_batch)>0,'every_model_seed_loss_positive':min(by_seed)>0}
    return {'checks':{k:bool(v) for k,v in checks.items()},'passed':bool(all(checks.values()))}
