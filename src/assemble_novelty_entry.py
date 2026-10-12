"""Combine the frozen novelty-risk alternative with the submitted v3 evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs_v3'
OUT=ROOT/'outputs_novelty'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_path=BASE/'submission_evidence_ranker.csv'
    candidate_path=BASE/'novelty_candidate.parquet'
    original_path=OUT/'original_robust_risk.parquet'
    dest=OUT/'submission_novelty_floor.csv'
    if dest.exists():
        raise ValueError('Candidate entry already exists; review rather than overwrite it')
    source=pl.read_csv(source_path,infer_schema=False)
    candidate=pl.read_parquet(candidate_path)
    original=pl.read_parquet(original_path)
    if source.height!=112540 or any(f['pair_id'].n_unique()!=source.height or set(f['pair_id'])!=set(source['pair_id']) for f in [candidate,original]):
        raise ValueError('Exact candidate/source pair coverage required')
    joined=source.join(candidate,on='pair_id',maintain_order='left').join(
        original.select('pair_id','original_robust_risk'),on='pair_id',maintain_order='left')
    old=joined['risk_score'].cast(pl.Float64).to_numpy()
    assert np.array_equal(old,joined['original_robust_risk'].to_numpy())
    percentile=joined['reference_percentile'].to_numpy()
    floor=.5*np.clip((percentile-.95)/.05,0,1)
    blended=joined['blended_risk'].to_numpy()
    np.testing.assert_allclose(floor,joined['risk_floor'].to_numpy(),rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.maximum(old,floor),blended,rtol=0,atol=1e-14)
    assert np.isfinite(blended).all() and np.all((blended>=0)&(blended<=1))
    assert np.array_equal(old[old>=.5],blended[old>=.5])
    changed=blended>old
    result=joined.with_columns(
        pl.when(pl.col('blended_risk')>pl.col('risk_score').cast(pl.Float64))
        .then(pl.col('blended_risk').cast(pl.String)).otherwise(pl.col('risk_score')).alias('risk_score'))
    result=result.select(source.columns)
    fixed=[c for c in source.columns if c!='risk_score']
    assert result.select(fixed).equals(source.select(fixed))
    assert result.null_count().sum_horizontal().sum()==0
    temporary=dest.with_suffix('.csv.tmp')
    result.write_csv(temporary)
    reread=pl.read_csv(temporary,infer_schema=False)
    assert reread.equals(result)
    np.testing.assert_allclose(reread['risk_score'].cast(pl.Float64).to_numpy(),blended,rtol=0,atol=1e-14)
    temporary.replace(dest)
    report={'submission':dest.name,'sha256':sha(dest),'rows':source.height,
            'risk_scores_raised':int(changed.sum()),'risk_scores_lowered':int((blended<old).sum()),
            'non_risk_cells_exactly_preserved':True,'high_risks_unchanged':True,
            'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
                [source_path,candidate_path,original_path,OUT/'provenance.json',Path(__file__)]},
            'recipe':'max(robust risk, 0.5*clip((negative-training novelty percentile-0.95)/0.05,0,1))',
            'note':'Frozen exploratory risk alternative. Three development splits show proxy-unseen gains and known-family losses. No new evidence or behavior changes.'}
    (OUT/'submission_novelty_floor.provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
