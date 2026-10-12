"""Extract five private v3 cases and their 25 submitted hands from original records."""
import hashlib
import json
import os
from pathlib import Path

os.environ['POLARS_MAX_THREADS'] = '2'
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORT = ROOT / 'data', ROOT / 'reports'
SUBMISSION = ROOT / 'outputs_v3' / 'submission_evidence_ranker.csv'
REVIEW_RANKS = {'PA37302E52E6D': 5}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    protected = [SUBMISSION, ROOT / 'outputs_v2' / 'submission_advanced_robust.csv',
                 REPORT / 'methodology-v2-draft.md', ROOT / 'outputs_v3' / 'evidence_ranker.joblib']
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    source = pl.read_csv(SUBMISSION)
    old = pl.read_csv(protected[1])
    fixed = ['pair_id', 'risk_score', 'predicted_behavior']
    if not source.select(fixed).equals(old.select(fixed)):
        raise ValueError('The v3 entry must preserve v2 pair risk and behavior')
    ranked = source.sort(['risk_score', 'pair_id'], descending=[True, False]).with_row_index('global_risk_rank', offset=1)
    selected = pl.concat([ranked.filter(pl.col('predicted_behavior') == family).head(count)
                          for family, count in [('directed_transfer', 2), ('soft_play', 2), ('coordinated_isolation', 1)]])
    selected = selected.sort('global_risk_rank').join(pl.read_csv(DATA / 'evaluation_pairs.csv'), on='pair_id', how='left')
    evidence_columns = [f'evidence_hand_{i}' for i in range(1, 6)]
    selected_hands = set(selected.select(evidence_columns).unpivot()['value'])
    frames = {}
    for name in ['hands', 'seats', 'actions']:
        scan = pl.scan_parquet(DATA / f'{name}.parquet').filter(pl.col('hand_id').is_in(selected_hands))
        if name == 'hands':
            scan = scan.with_columns(pl.col('started_at').cast(pl.String))
        frames[name] = scan.collect(engine='streaming')
    if frames['hands'].height != len(selected_hands) or set(frames['hands']['hand_id']) != selected_hands:
        raise ValueError('A selected hand is missing or duplicated in the source')
    cases, ids = [], []
    for row in selected.to_dicts():
        records = []
        for rank, column in enumerate(evidence_columns, 1):
            hand_id = row[column]
            hand = frames['hands'].filter(pl.col('hand_id') == hand_id).to_dicts()[0]
            seats = frames['seats'].filter(pl.col('hand_id') == hand_id).sort('seat_no').to_dicts()
            actions = frames['actions'].filter(pl.col('hand_id') == hand_id).sort('action_no').to_dicts()
            if hand['phase'] != 'evaluation' or not {row['player_1'], row['player_2']}.issubset({s['player_id'] for s in seats}):
                raise ValueError('A selected hand does not contain both pair members in evaluation')
            if len({a['action_no'] for a in actions}) != len(actions):
                raise ValueError('Selected actions are duplicated')
            records.append({'submitted_evidence_rank': rank, 'hand': hand, 'seats': seats, 'actions': actions})
        if len({record['hand']['table_id'] for record in records}) != 1:
            raise ValueError('Selected evidence crosses table pools')
        cases.append({**row, 'evidence': records})
        ids.append({**{key: row[key] for key in ['pair_id', 'player_1', 'player_2', 'global_risk_rank', 'risk_score', 'predicted_behavior', 'shared_hands']},
                    'table_id': records[0]['hand']['table_id'], 'phase': 'evaluation',
                    'reviewed_hand_id': row[f'evidence_hand_{REVIEW_RANKS.get(row["pair_id"], 1)}'],
                    'reviewed_submitted_evidence_rank': REVIEW_RANKS.get(row['pair_id'], 1),
                    'submitted_evidence_hand_ids': [row[c] for c in evidence_columns]})
    if {str(path.relative_to(ROOT)): digest(path) for path in protected} != before:
        raise ValueError('A protected submission, model, or v2 draft changed during extraction')
    common = {'status': 'PRIVATE v3 DRAFT; synthetic records; predictions are not verified labels',
              'submission_file': str(SUBMISSION.relative_to(ROOT)), 'submission_sha256': digest(SUBMISSION),
              'pair_model_file': 'outputs_v2/pair_model_robust.joblib',
              'pair_model_sha256': digest(ROOT / 'outputs_v2' / 'pair_model_robust.joblib'),
              'evidence_model_file': 'outputs_v3/evidence_ranker.joblib',
              'evidence_model_sha256': digest(ROOT / 'outputs_v3' / 'evidence_ranker.joblib'),
              'selection': 'two highest-risk transfer pairs, two highest-risk soft-play pairs, highest-risk isolation pair; review v3 rank 1 except isolation rank 5, chosen after inspecting its five submitted hands',
              'risk_and_behavior_identical_to_v2': True, 'protected_files_unchanged': True,
              'raw_source_metadata': {str((DATA / f'{name}.parquet').relative_to(ROOT)):
                  {'bytes': (DATA / f'{name}.parquet').stat().st_size, 'mtime_ns': (DATA / f'{name}.parquet').stat().st_mtime_ns}
                  for name in frames}, 'extractor_sha256': digest(Path(__file__))}
    (REPORT / 'selected-v3-evidence-cases.json').write_text(json.dumps({**common, 'cases': cases}, indent=2) + '\n', encoding='utf-8')
    draft = REPORT / 'methodology-v3-draft.md'
    (REPORT / 'selected-v3-case-ids.json').write_text(json.dumps({**common, 'case_count': len(ids),
        'methodology_file': 'reports/methodology-v3-draft.md',
        'methodology_word_count': len(draft.read_text(encoding='utf-8').split()) if draft.exists() else None,
        'methodology_sha256': digest(draft) if draft.exists() else None, 'reviewed_pairs': ids}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(cases), 'submitted_hands': len(selected_hands), 'protected_files_unchanged': True}), flush=True)


if __name__ == '__main__':
    run()
