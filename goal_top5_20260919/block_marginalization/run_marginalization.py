from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","POLARS_MAX_THREADS"): os.environ[k]="2"
import itertools, json, site, sys, time
from pathlib import Path
ROOT=Path(r"C:\Data\Projects\Money Making"); POKER=ROOT/"work/poker"
site.addsitedir(str(POKER/".venv/Lib/site-packages"))
import numpy as np, polars as pl

OUT=POKER/"goal_top5_20260919/block_marginalization"
AUD=POKER/"Claude Version/experiments/step276_evidence_headroom_20260919/audit_production_stage2_p.parquet"
HEAD=POKER/"goal_top5_20260919/block_head/audit_scores.parquet"
FAMS=("directed_transfer","soft_play","coordinated_isolation")

def q_lt5(probs):
    d=np.zeros(6); d[0]=1.
    for p in probs:
        n=d*(1-p); n[1:]+=d[:-1]*p; n[5]+=d[5]*p; d=n
    return float(d[:5].sum())

def expected_accept(p,b,chrono):
    n=len(p); out=np.empty(n)
    for i in range(n):
        other=np.arange(n)!=i
        early=(chrono<chrono[i])&other
        later=(chrono>chrono[i])&other
        qe=q_lt5(p[early]*b[early])
        ql=q_lt5(np.r_[p[early],p[later]*b[later]])
        out[i]=p[i]*(b[i]*qe+(1-b[i])*ql)
    return out

def brute_one(p,b,chrono,i):
    js=[j for j in range(len(p)) if j!=i]; ans=0.
    for bi in (0,1):
      pbi=b[i] if bi else 1-b[i]
      for bs in itertools.product((0,1),repeat=len(js)):
        pb=pbi
        for j,x in zip(js,bs): pb*=b[j] if x else 1-b[j]
        for ys in itertools.product((0,1),repeat=len(js)):
          py=1.; ahead=0
          for j,x,y in zip(js,bs,ys):
            py*=p[j] if y else 1-p[j]
            if y and ((bi and x and chrono[j]<chrono[i]) or ((not bi) and (x or chrono[j]<chrono[i]))): ahead+=1
          ans += pb*py*(ahead<5)
    return p[i]*ans

def decode_groups(a,b):
    p=a["prod_p_stage2"].to_numpy(); cr=a["chrono_rank"].to_numpy(); pid=a["pair_id"].to_numpy(); out=np.empty(len(a))
    for _,gdf in a.with_row_index("ix").group_by("pair_id",maintain_order=True):
        g=gdf["ix"].to_numpy(); z=expected_accept(p[g],b[g],cr[g]); out[g]=10+z+1e-9*p[g]
    return out

def hard_decode(a,t1):
    p=a["prod_p_stage2"].to_numpy(); cr=a["chrono_rank"].to_numpy(); out=np.empty(len(a))
    for _,gdf in a.with_row_index("ix").group_by("pair_id",maintain_order=True):
        g=gdf["ix"].to_numpy(); order=np.lexsort((cr[g],~t1[g])); q=np.empty(len(g));
        d=np.zeros(6); d[0]=1.
        for j in order:
            q[j]=d[:5].sum(); x=p[g[j]]; n=d*(1-x); n[1:]+=d[:-1]*x; n[5]+=d[5]*x; d=n
        out[g]=10+p[g]*q+1e-9*p[g]
    return out

def perq(a,score,ntruth):
    z=a.select("pair_id","table_id","behavior_family","relevance").with_columns(pl.Series("score",score)); rows=[]
    for (pid,),g in z.group_by("pair_id",maintain_order=True):
        y=g.sort("score",descending=True)["relevance"].to_numpy(); hit=0;s=0.
        for k,v in enumerate(y[:5],1):
            if v: hit+=1;s+=hit/k
        rows.append((pid,g["table_id"][0],g["behavior_family"][0],s/min(5,ntruth[pid])))
    return pl.DataFrame(rows,schema=["pair_id","table_id","family","ap"],orient="row")

def boot(q,n=5000):
    rng=np.random.default_rng(8801); ts=q["table_id"].unique().to_list(); d={t:q.filter(pl.col("table_id")==t)["delta"].to_numpy() for t in ts}; z=[]
    for _ in range(n): z.append(np.concatenate([d[t] for t in rng.choice(ts,len(ts),replace=True)]).mean())
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))],float(np.mean(np.asarray(z)>0))

def main():
    t=time.time(); rng=np.random.default_rng(4401); tests=[]
    for n in range(1,6):
      for _ in range(5):
        p=rng.uniform(.02,.98,n); b=rng.uniform(.02,.98,n); c=rng.permutation(np.arange(n)).astype(float); an=expected_accept(p,b,c); br=np.array([brute_one(p,b,c,i) for i in range(n)]); tests.append(float(np.max(np.abs(an-br))))
    enum_err=max(tests); assert enum_err<1e-12,enum_err
    full=pl.read_parquet(AUD); a=full.filter(pl.col("in_top20")).sort("pair_id","hand_id")
    h=pl.read_parquet(HEAD).select("pair_id","hand_id","block_p","score_control")
    a=a.join(h,on=["pair_id","hand_id"],how="left",validate="1:1")
    fam=a["behavior_family"].to_numpy(); t1=(a["T1"].to_numpy()&(fam!="coordinated_isolation"))
    binary=decode_groups(a,t1.astype(float)); hard=hard_decode(a,t1); ref=a["prod_decoded_score"].to_numpy()
    binary_err=float(max(np.max(np.abs(binary-hard)),np.max(np.abs(binary-ref))))
    binary_lists=all(x.sort("s",descending=True).select("hand_id").head(5).equals(y.sort("s",descending=True).select("hand_id").head(5)) for (_,x),(_,y) in zip(a.with_columns(pl.Series("s",binary)).group_by("pair_id",maintain_order=True),a.with_columns(pl.Series("s",ref)).group_by("pair_id",maintain_order=True)))
    assert binary_err<1e-12 and binary_lists,(binary_err,binary_lists)
    zero=np.zeros(len(a)); znew=decode_groups(a,zero); zhard=hard_decode(a,np.zeros(len(a),dtype=bool)); zero_err=float(np.max(np.abs(znew-zhard))); assert zero_err<1e-12,zero_err
    # Only after all three correctness gates do we score against relevance.
    live=fam!="coordinated_isolation"; bp=t1.astype(float); bc=t1.astype(float)
    learned=a["block_p"].to_numpy()
    bp[live]=.75*t1[live]+.25*learned[live]
    sp=decode_groups(a,bp)
    nt=dict(full.group_by("pair_id").agg(pl.col("relevance").sum()).select("pair_id","relevance").iter_rows())
    q0=perq(a,ref,nt); q=perq(a,sp,nt).rename({"ap":"new"}).join(q0,on=["pair_id","table_id","family"]).with_columns((pl.col("new")-pl.col("ap")).alias("delta"))
    ci,pgt=boot(q); half=(q["table_id"].hash(seed=91)%2).to_numpy(); dv=q["delta"].to_numpy(); fd={f:float(q.filter(pl.col("family")==f)["delta"].mean()) for f in FAMS}
    changed=int(sum(not x.sort("s",descending=True).select("hand_id").head(5).equals(y.sort("s",descending=True).select("hand_id").head(5)) for (_,x),(_,y) in zip(a.with_columns(pl.Series("s",sp)).group_by("pair_id",maintain_order=True),a.with_columns(pl.Series("s",ref)).group_by("pair_id",maintain_order=True))))
    r={"enum_max_abs":enum_err,"binary_identity_max_abs":binary_err,"binary_lists_identical":binary_lists,"all_zero_chrono_max_abs":zero_err,"baseline_map5":float(q["ap"].mean()),"primary_map5":float(q["new"].mean()),"primary_delta":float(dv.mean()),"ci95_table_cluster":ci,"bootstrap_p_gt0":pgt,"halves":[float(dv[half==0].mean()),float(dv[half==1].mean())],"family_delta":fd,"control_delta":None,"control_status":"UNAVAILABLE: prior block_head artifact saved decoded control score, not frozen shuffled block probability; no refit permitted", "lists_changed":changed,"seconds":time.time()-t}
    r["pass"]=False; r["worth_public_slot"]=False
    (OUT/"results.json").write_text(json.dumps(r,indent=2),encoding="utf-8"); a.select("pair_id","hand_id","table_id","behavior_family","relevance").with_columns(pl.Series("block_p_mixed",bp),pl.Series("score_primary",sp)).write_parquet(OUT/"audit_scores.parquet"); print(json.dumps(r,indent=2))
if __name__=="__main__": main()
