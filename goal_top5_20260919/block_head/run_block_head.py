from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "2"

import json, site, sys, time
from pathlib import Path

ROOT = Path(r"C:\Data\Projects\Money Making")
POKER = ROOT / "work/poker"
site.addsitedir(str(POKER / ".venv/Lib/site-packages"))

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

OUT = POKER / "goal_top5_20260919/block_head"
BASE = POKER / "Fable Version/marking_rule/out/base_frame.parquet"
AUD = POKER / "Claude Version/experiments/step276_evidence_headroom_20260919/audit_production_stage2_p.parquet"
SEEDS = list(range(6201, 6211))
FAMS = ("directed_transfer", "soft_play", "coordinated_isolation")
KEYS = ["pair_id", "hand_id"]

def less_than_five(ps):
    dist, out = np.zeros(6), np.empty(len(ps))
    dist[0] = 1.0
    for i, p in enumerate(ps):
        out[i] = dist[:5].sum()
        new = dist * (1-p); new[1:] += dist[:-1]*p; new[5] += dist[5]*p
        dist = new
    return out

def decode(d, priority):
    pid=d["pair_id"].to_numpy(); cr=d["chrono_rank"].to_numpy(); pp=d["prod_p_stage2"].to_numpy()
    out=np.zeros(len(d), dtype=float)
    for _, ix in d.with_row_index("ix").group_by("pair_id", maintain_order=True):
        g=ix["ix"].to_numpy(); p=np.clip(pp[g],0,1)
        order=np.lexsort((cr[g], -priority[g])); q=np.empty(len(g)); q[order]=less_than_five(p[order])
        out[g]=10+p*q+1e-9*p
    return out

def per_query_ap(d, score, ntruth):
    z=d.select(KEYS+["table_id","behavior_family","relevance"]).with_columns(pl.Series("score",score))
    rows=[]
    for (pid,), g in z.group_by("pair_id", maintain_order=True):
        g=g.sort("score", descending=True); y=g["relevance"].to_numpy().astype(int); n=ntruth[pid]
        hit=0; s=0.
        for k,v in enumerate(y[:5],1):
            if v: hit+=1; s+=hit/k
        rows.append((pid,g["table_id"][0],g["behavior_family"][0],s/min(5,n)))
    return pl.DataFrame(rows,schema=["pair_id","table_id","family","ap"],orient="row")

def models(seed):
    a=lgb.LGBMClassifier(objective="binary",n_estimators=120,num_leaves=7,max_depth=3,learning_rate=.04,
        min_child_samples=20,colsample_bytree=.5,reg_lambda=2.,random_state=seed,n_jobs=2,verbosity=-1)
    b=ExtraTreesClassifier(n_estimators=200,max_depth=5,min_samples_leaf=8,max_features="sqrt",
        class_weight="balanced",random_state=seed,n_jobs=2)
    return a,b

def boot_ci(q, n=5000, seed=777):
    rng=np.random.default_rng(seed); tabs=q["table_id"].unique().to_list(); by={t:q.filter(pl.col("table_id")==t)["delta"].to_numpy() for t in tabs}
    vals=[]
    for _ in range(n):
        take=rng.choice(tabs,len(tabs),replace=True); vals.append(np.concatenate([by[t] for t in take]).mean())
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],float(np.mean(np.array(vals)>0))

def main():
    t0=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    b=pl.read_parquet(BASE); afull=pl.read_parquet(AUD); a=afull.filter(pl.col("in_top20")).sort(KEYS)
    ntruth=dict(afull.group_by("pair_id").agg(pl.col("relevance").sum()).select("pair_id","relevance").iter_rows())
    audit_pairs=set(a["pair_id"].unique().to_list()); audit_tables=set(a["table_id"].unique().to_list())
    # Direct evidence-rank descent labels already frozen by s01. Only exact one-descent lists are identified.
    train=b.filter((pl.col("listed")==1)&(pl.col("n_desc")==1)&(pl.col("block").is_in([0,1])))
    train=train.filter(~pl.col("pair_id").is_in(list(audit_pairs)) & ~pl.col("table_id").is_in(list(audit_tables)))
    # Mechanical predictors only. Remove chronology/session/cumulative/order proxies and all keys/targets.
    forbidden=("id","rank","idx","frac","time","session","cum","rebuy","stackchg","age")
    meta=set(["pair_id","hand_id","listed","family","fold","table_id","big_blind","final_pot","players_dealt","players_at_showdown","board_cards","t_us","evidence_rank","idx","n_hands","frac","block","n_desc"])
    feats=[c for c,dt in b.schema.items() if c not in meta and dt.is_numeric() and not any(x in c.lower() for x in forbidden)]
    # Ensure audit keys can be joined to the exact same development tensor rows.
    af=a.select(KEYS).join(b.select(KEYS+feats),on=KEYS,how="left",validate="1:1")
    assert af.height==a.height and af.select(pl.any_horizontal(pl.all().is_null()).sum()).item()==0
    Xte=np.nan_to_num(af.select(feats).to_numpy().astype(np.float32),nan=0,posinf=0,neginf=0)
    prod_priority=(a["T1"].to_numpy() & (a["behavior_family"].to_numpy()!="coordinated_isolation")).astype(float)
    replay=decode(a,prod_priority); ref=a["prod_decoded_score"].to_numpy(); replay_err=float(np.max(np.abs(replay-ref)))
    assert replay_err<=1e-12,replay_err
    q0=per_query_ap(a,ref,ntruth); tabs=sorted(train["table_id"].unique().to_list())
    reps=[]; pred_real=[]; pred_shuf=[]
    for sd in SEEDS:
        rng=np.random.default_rng(sd); chosen=set(rng.choice(tabs,max(2,int(.8*len(tabs))),replace=False).tolist())
        tr=train.filter(pl.col("table_id").is_in(list(chosen)))
        X=np.nan_to_num(tr.select(feats).to_numpy().astype(np.float32),nan=0,posinf=0,neginf=0); y=tr["block"].to_numpy().astype(int)
        # target probability means block 0
        yy=1-y
        m1,m2=models(sd); m1.fit(X,yy); m2.fit(X,yy); pr=(m1.predict_proba(Xte)[:,1]+m2.predict_proba(Xte)[:,1])/2
        ys=yy.copy()
        # Family-stratified label permutation, identical row population.
        fam=tr["family"].to_numpy()
        for f in FAMS:
            ix=np.flatnonzero(fam==f); ys[ix]=rng.permutation(ys[ix])
        c1,c2=models(sd+10000); c1.fit(X,ys); c2.fit(X,ys); pc=(c1.predict_proba(Xte)[:,1]+c2.predict_proba(Xte)[:,1])/2
        fam_a=a["behavior_family"].to_numpy(); live=fam_a!="coordinated_isolation"
        pri=prod_priority.copy(); pri[live]=.75*prod_priority[live]+.25*pr[live]
        prc=prod_priority.copy(); prc[live]=.75*prod_priority[live]+.25*pc[live]
        qr=per_query_ap(a,decode(a,pri),ntruth).rename({"ap":"new"}).join(q0,on=["pair_id","table_id","family"]).with_columns((pl.col("new")-pl.col("ap")).alias("delta"))
        qc=per_query_ap(a,decode(a,prc),ntruth).rename({"ap":"new"}).join(q0,on=["pair_id","table_id","family"]).with_columns((pl.col("new")-pl.col("ap")).alias("delta"))
        reps.append({"seed":sd,"tables":len(chosen),"train_rows":tr.height,"delta":float(qr["delta"].mean()),"control_delta":float(qc["delta"].mean())})
        pred_real.append(pr); pred_shuf.append(pc)
        print("rep",sd,reps[-1],flush=True)
    pr=np.mean(pred_real,axis=0); pc=np.mean(pred_shuf,axis=0); fam_a=a["behavior_family"].to_numpy(); live=fam_a!="coordinated_isolation"
    pri=prod_priority.copy(); pri[live]=.75*prod_priority[live]+.25*pr[live]
    prc=prod_priority.copy(); prc[live]=.75*prod_priority[live]+.25*pc[live]
    newscore=decode(a,pri); ctlscore=decode(a,prc)
    q=per_query_ap(a,newscore,ntruth).rename({"ap":"new"}).join(q0,on=["pair_id","table_id","family"]).with_columns((pl.col("new")-pl.col("ap")).alias("delta"))
    qc=per_query_ap(a,ctlscore,ntruth).rename({"ap":"new"}).join(q0,on=["pair_id","table_id","family"]).with_columns((pl.col("new")-pl.col("ap")).alias("delta"))
    ci,pgt=boot_ci(q); half=(q["table_id"].hash(seed=91)%2).to_numpy()
    fam_delta={f:float(q.filter(pl.col("family")==f)["delta"].mean()) for f in FAMS}
    ablk=b.filter((pl.col("listed")==1)&(pl.col("n_desc")==1)&(pl.col("block").is_in([0,1]))&pl.col("pair_id").is_in(list(audit_pairs))).select(KEYS+["block"])
    aj=a.select(KEYS).with_row_index("ix").join(ablk,on=KEYS,how="inner")
    block_auc=float(roc_auc_score(1-aj["block"].to_numpy(),pr[aj["ix"].to_numpy()]))
    results={"protocol":"PROTOCOL.md","replay_max_abs":replay_err,"train_rows":train.height,"train_pairs":train["pair_id"].n_unique(),"train_tables":len(tabs),"features":len(feats),"audit_block_auc":block_auc,
      "baseline_map5":float(q["ap"].mean()),"primary_map5":float(q["new"].mean()),"primary_delta":float(q["delta"].mean()),"ci95_table_cluster":ci,"bootstrap_p_gt0":pgt,
      "halves":[float(q["delta"].to_numpy()[half==0].mean()),float(q["delta"].to_numpy()[half==1].mean())],"family_delta":fam_delta,
      "control_delta":float(qc["delta"].mean()),"replicates":reps,"replicates_positive":sum(r["delta"]>0 for r in reps),"seconds":time.time()-t0}
    results["pass"]=bool(results["primary_delta"]>0 and ci[0]>0 and min(results["halves"])>=0 and results["replicates_positive"]>=8 and results["primary_delta"]>results["control_delta"] and abs(fam_delta["coordinated_isolation"])<1e-15)
    results["worth_public_slot"]=bool(results["pass"] and results["primary_delta"]>=.003)
    (OUT/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    a.select(KEYS+["table_id","behavior_family","relevance"]).with_columns(pl.Series("block_p",pr),pl.Series("score_primary",newscore),pl.Series("score_control",ctlscore)).write_parquet(OUT/"audit_scores.parquet")
    print(json.dumps(results,indent=2),flush=True)

if __name__=="__main__": main()
