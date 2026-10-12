"""Freeze five continuity cases and all 25 v6 evidence records for review."""
import os
os.environ['POLARS_MAX_THREADS'] = '2'
import json
from pathlib import Path
import polars as pl
from features import file_sha256 as sha

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports'
SOURCE = ROOT/'outputs_v6/submission_sequence_evidence.csv'


def main():
    target = REPORT/'selected-v6-evidence-cases.json'
    if target.exists():
        raise FileExistsError('Preserve extracted cases')
    prior = json.loads((REPORT/'selected-v5-case-ids.json').read_text())
    selected_ids = [r['pair_id'] for r in prior['reviewed_pairs']]
    source = pl.read_csv(SOURCE)
    selected = source.filter(pl.col('pair_id').is_in(selected_ids)).sort(['risk_score','pair_id'],descending=[True,False]).join(
        pl.read_csv(ROOT/'data/evaluation_pairs.csv'),on='pair_id',validate='1:1')
    columns = [f'evidence_hand_{i}' for i in range(1,6)]
    hand_ids = selected.select(columns).unpivot()['value'].unique().to_list()
    frames = {}
    for name in ['hands','seats','actions']:
        scan = pl.scan_parquet(ROOT/'data'/f'{name}.parquet').filter(pl.col('hand_id').is_in(hand_ids))
        if name == 'hands':
            scan = scan.with_columns(pl.col('started_at').cast(pl.String))
        frames[name] = scan.collect()
    assert frames['hands'].height == len(hand_ids)
    cases = []
    for row in selected.iter_rows(named=True):
        evidence = []
        for rank, column in enumerate(columns,1):
            h = row[column]
            hand = frames['hands'].filter(pl.col('hand_id')==h).to_dicts()[0]
            seats = frames['seats'].filter(pl.col('hand_id')==h).sort('seat_no').to_dicts()
            actions = frames['actions'].filter(pl.col('hand_id')==h).sort('action_no').to_dicts()
            assert hand['phase']=='evaluation' and {row['player_1'],row['player_2']}.issubset({s['player_id'] for s in seats})
            evidence.append({'rank':rank,'hand':hand,'seats':seats,'actions':actions})
        cases.append({**row,'evidence':evidence})
    payload = {'status':'PRIVATE exploratory v6, synthetic predictions not confirmed labels',
        'selection':'Same five pair IDs as v5, reordered by unchanged risk; review first v6 hand for each pair.',
        'submission_sha256':sha(SOURCE),'evidence_model_sha256':sha(ROOT/'outputs_v6/sequence_ranker.joblib'),
        'case_count':len(cases),'evidence_records':sum(len(c['evidence']) for c in cases),'cases':cases,
        'source_sha256':{str(ROOT/'data'/f'{n}.parquet'):sha(ROOT/'data'/f'{n}.parquet') for n in frames}}
    target.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'evidence_records':payload['evidence_records']}),flush=True)


if __name__=='__main__':
    main()
