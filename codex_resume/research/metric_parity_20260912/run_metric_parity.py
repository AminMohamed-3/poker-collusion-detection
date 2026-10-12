"""Compact synthetic parity checks for the downloaded organizer metric.

This reads only source code and uses invented pair IDs, labels, scores, and
hand IDs.  It does not open any development, evaluation, or submission data.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
POKER = HERE.parent
OFFICIAL = HERE / 'research' / 'official_metric_20260912' / 'slash-poker-competition-metric.ipynb'
ROBUST = POKER / 'src' / 'robust_pair_experiment.py'
CONDITIONAL = POKER / 'src' / 'evaluate_conditional_behavior.py'
STEP5 = POKER / 'Claude Version' / 'src' / 'evidence_aug' / 'step5_evaluate.py'

# The existing environment keeps pandas in this isolated overlay.
sys.path.insert(0, str(HERE / 'vendor' / 'catboost'))
import numpy as np
import pandas as pd
import polars as pl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_from(path, name, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def source_from_notebook(path):
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    return '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'] if cell['cell_type'] == 'code')


def rows(spec):
    """Build an organizer-schema fixture from (id, risk, behavior, evidence)."""
    output = []
    for pair_id, risk, behavior, evidence in spec:
        row = {'pair_id': pair_id, 'risk_score': risk, 'predicted_behavior': behavior}
        for i in range(5):
            row[f'evidence_hand_{i + 1}'] = evidence[i] if i < len(evidence) else 'NO_EVIDENCE'
        output.append(row)
    return pd.DataFrame(output)


def official_breakdown(solution, submission, official):
    """The three terms from the inspected official score body, for assertions."""
    truth = solution.set_index('pair_id').sort_index()
    prediction = submission.set_index('pair_id').loc[truth.index]
    y = truth['risk_score'].to_numpy(dtype=int)
    risk = prediction['risk_score'].to_numpy(dtype=float)
    behavior = prediction['predicted_behavior'].astype(str).to_numpy()
    pair_ap = official['_average_precision'](y, risk)
    bterms = []
    for family in official['TARGET_BEHAVIORS']:
        target = (truth['predicted_behavior'].astype(str).to_numpy() == family).astype(int)
        bterms.append(0. if target.sum() == 0 else official['_average_precision'](target, np.where(behavior == family, risk, 0.)))
    eterms = []
    for position in np.flatnonzero(y == 1):
        relevant = set(official['_clean_evidence'](truth.iloc[position].loc[list(official['EVIDENCE_COLUMNS'])].tolist()))
        submitted = official['_clean_evidence'](prediction.iloc[position].loc[list(official['EVIDENCE_COLUMNS'])].tolist())
        hits = precision_sum = 0.
        for rank, hand in enumerate(submitted[:5], 1):
            if hand in relevant:
                hits += 1
                precision_sum += hits / rank
        eterms.append(0. if not relevant else precision_sum / min(len(relevant), 5))
    evidence_map = float(np.mean(eterms)) if eterms else 0.
    return {'pair_ap': pair_ap, 'behavior_map': float(np.mean(bterms)), 'evidence_map': evidence_map,
            'score': .7 * pair_ap + .1 * float(np.mean(bterms)) + .2 * evidence_map}


def main():
    # The notebook was inspected before this script is executed. It is then
    # executed in-process only to call its metric with synthetic DataFrames.
    official = {'np': np, 'pd': pd}
    exec(compile(source_from_notebook(OFFICIAL), str(OFFICIAL), 'exec'), official)
    stable_ap = function_from(ROBUST, 'stable_ap', {'np': np})
    conditional_ns = {'np': np, 'stable_ap': stable_ap,
                      'LABELS': {'none': 0, 'directed_transfer': 1, 'soft_play': 2, 'coordinated_isolation': 3}}
    family_metrics = function_from(CONDITIONAL, 'family_metrics', conditional_ns)
    step5_ns = {'np': np, 'pl': pl, 'TIE_SEED': 931207}
    function_from(STEP5, 'ap5', step5_ns)
    per_pair_map = function_from(STEP5, 'per_pair_map', step5_ns)

    # Organizer first sorts pair_id; stable_ap agrees exactly when given that
    # documented order, including a stable mergesort tie.
    tied_truth = np.array([1, 0, 1])
    tied_scores = np.array([.5, .5, .5])
    tie_official = official['_average_precision'](tied_truth, tied_scores)
    tie_local = stable_ap(tied_truth, tied_scores)
    assert np.isclose(tie_official, 5 / 6) and np.isclose(tie_local, 5 / 6)
    # This is deliberately *not* pair_id order: it demonstrates the explicit
    # ordering precondition of stable_ap, rather than an official mismatch.
    unordered_truth = np.array([0, 1, 1])  # pair IDs b, a, c
    assert np.isclose(stable_ap(unordered_truth, tied_scores), 7 / 12)

    absent_solution = rows([
        ('p1', 1, 'directed_transfer', []), ('p2', 0, 'none', []), ('p3', 0, 'none', [])])
    absent_submission = rows([
        ('p1', .9, 'directed_transfer', []), ('p2', .8, 'directed_transfer', []), ('p3', .1, 'none', [])])
    absent = official_breakdown(absent_solution, absent_submission, official)
    present_only = family_metrics(np.array([1, 0, 0]), np.array([.9, .8, .1]), np.array([1, 1, 0]))
    assert np.isclose(absent['behavior_map'], 1 / 3)
    assert np.isclose(present_only['behavior_map'], 1.)

    # other_coordination is a positive pair/evidence case for the organizer,
    # but is not a fourth term in its behavior macro average.
    other_solution = rows([
        ('p1', 1, 'directed_transfer', []), ('p2', 1, 'other_coordination', []), ('p3', 0, 'none', [])])
    other_submission = rows([
        ('p1', .9, 'directed_transfer', []), ('p2', .95, 'directed_transfer', []), ('p3', .1, 'none', [])])
    other = official_breakdown(other_solution, other_submission, official)
    retained_as_non_target = family_metrics(np.array([1, 0, 0]), np.array([.9, .95, .1]), np.array([1, 1, 0]))
    assert np.isclose(other['pair_ap'], 1.)
    assert np.isclose(other['behavior_map'], 1 / 6)
    assert np.isclose(retained_as_non_target['by_family_ap']['directed_transfer'], .5)
    assert np.isclose(stable_ap(np.array([1, 0, 0]), np.array([.9, .95, .1])), .5)

    evidence_solution = rows([
        ('p1', 1, 'directed_transfer', ['h2', 'h1']),
        ('p2', 0, 'none', []), ('p3', 0, 'none', [])])
    evidence_good = rows([
        ('p1', .9, 'directed_transfer', ['h1', 'h2']),
        ('p2', .2, 'none', []), ('p3', .1, 'none', [])])
    evidence_ungated = rows([
        ('p1', .1, 'none', ['h1', 'h2']),
        ('p2', .9, 'none', []), ('p3', .8, 'none', [])])
    good_terms = official_breakdown(evidence_solution, evidence_good, official)
    ungated_terms = official_breakdown(evidence_solution, evidence_ungated, official)
    assert np.isclose(good_terms['evidence_map'], 1.) and np.isclose(ungated_terms['evidence_map'], 1.)
    assert np.isclose(official['score'](evidence_solution, evidence_good, 'pair_id'), good_terms['score'])
    assert np.isclose(official['score'](evidence_solution, evidence_ungated, 'pair_id'), ungated_terms['score'])

    # Step5 creates a deterministic submitted ordering for tied hand scores;
    # once that order is submitted, its AP@5 matches organizer evidence AP.
    hand_frame = pl.DataFrame({'pair_id': ['p1'] * 4, 'hand_id': ['h1', 'h2', 'h3', 'h4'],
                                'table_id': ['t'] * 4, 'score': [.5] * 4})
    local_hand_ap = per_pair_map(hand_frame, {'p1': {'h1', 'h3'}})['p1']
    ordered = hand_frame.sort('pair_id', 'hand_id').with_columns(
        pl.Series('tie', np.random.default_rng(931207).random(4))).sort(['score', 'tie'], descending=[True, False])['hand_id'].to_list()
    hand_solution = rows([('p1', 1, 'directed_transfer', ['h1', 'h3'])])
    hand_submission = rows([('p1', 1, 'directed_transfer', ordered)])
    hand_terms = official_breakdown(hand_solution, hand_submission, official)
    assert np.isclose(local_hand_ap, hand_terms['evidence_map'])

    receipt = {
        'synthetic_only': True,
        'source_hashes': {str(p): sha(p) for p in (OFFICIAL, ROBUST, CONDITIONAL, STEP5)},
        'checks': {
            'stable_pair_ap_tie': {'official': tie_official, 'stable_ap_sorted_pair_id': tie_local,
                                   'unordered_stable_ap': stable_ap(unordered_truth, tied_scores),
                                   'finding': 'Exact only when stable_ap receives its documented pair_id order.'},
            'absent_target_family': {'official_behavior_map': absent['behavior_map'],
                                      'family_metrics_present_only': present_only['behavior_map'],
                                      'finding': 'Official always divides behavior MAP by three targets; family_metrics intentionally averages only present targets.'},
            'other_coordination': {'official_pair_ap': other['pair_ap'], 'official_behavior_map': other['behavior_map'],
                                   'retained_non_target_directed_ap': retained_as_non_target['by_family_ap']['directed_transfer'],
                                   'local_pair_ap_if_encoded_none': stable_ap(np.array([1, 0, 0]), np.array([.9, .95, .1])),
                                   'finding': 'Organizer treats other_coordination as risk-positive but excludes it from target behavior terms; omitting it from a validation universe changes pair AP and can inflate target AP.'},
            'evidence_gating': {'evidence_map_with_high_risk_correct_behavior': good_terms['evidence_map'],
                                'evidence_map_with_low_risk_none_behavior': ungated_terms['evidence_map'],
                                'finding': 'Evidence MAP uses every truth risk-positive pair and does not depend on submitted risk or behavior.'},
            'step5_evidence_ap': {'local_per_pair_map': local_hand_ap, 'official_evidence_map_after_same_submitted_order': hand_terms['evidence_map'],
                                  'tied_hand_order': ordered,
                                  'finding': 'AP@5 formula agrees; Step5 resolves score ties with its fixed seeded hand ordering before an implied submission order.'}
        },
        'implications': [
            'Use stable_ap for official pair/behavior parity only on explicitly pair_id-sorted rows; this is already its documented contract.',
            'family_metrics is a present-family diagnostic, not the official behavior macro metric whenever a target family is absent.',
            'Any validation containing other_coordination must retain its binary risk-positive status for pair and evidence terms, and retain its rows as negatives in each disclosed-family AP; it receives no fourth behavior-macro term.',
            'Evidence MAP@5 should be evaluated across all truth risk-positive pairs without risk or predicted-behavior gating. Step5 per_pair_map is formula-compatible after its deterministic score-tie ordering.'
        ]
    }
    destination = Path(__file__).with_name('receipt.json')
    destination.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'receipt': str(destination), 'checks': list(receipt['checks'])}))


if __name__ == '__main__':
    main()
