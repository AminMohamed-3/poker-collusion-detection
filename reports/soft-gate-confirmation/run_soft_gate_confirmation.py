"""Measure pretrained hand-evidence scoring and fixed rank fusion on two splits."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '4'
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import rankdata
from huggingface_hub import hf_hub_download
from tabicl import TabICLClassifier


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_results(rows, scores, candidate, job):
    pairs = {}
    for i, row in enumerate(rows): pairs.setdefault(row['pair_id'], []).append(i)
    tie = np.random.default_rng(931207).random(len(rows))
    result = []
    for pair, indices in pairs.items():
        indices = np.array(indices)
        order = indices[np.lexsort((tie[indices], -scores[indices]))[:5]]
        positives = np.array([int(rows[i]['relevance']) for i in order])
        total = sum(int(rows[i]['relevance']) for i in indices)
        value = float(np.sum(np.cumsum(positives) / np.arange(1, len(order) + 1) * positives) / min(total, 5))
        result.append({'pair_id': pair, 'table_id': rows[indices[0]]['table_id'],
            'family': rows[indices[0]]['behavior_family'], 'job': job, 'candidate': candidate, 'map5': value})
    return result


def main():
    root = Path(__file__).resolve().parent
    plan_path = root / 'plan.json'; plan = json.loads(plan_path.read_text())
    assert all(sha(root / n) == h for n, h in plan['files'].items())
    out = root / 'results'; out.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(4)
    checkpoint = Path(hf_hub_download(plan['checkpoint_repo'], plan['checkpoint_file'], revision=plan['checkpoint_revision']))
    assert sha(checkpoint) == plan['checkpoint_sha256']
    allrows = []
    for job in plan['jobs']:
        receipt = out / (job + '.json'); predfile = out / (job + '.npz')
        if receipt.exists():
            saved = json.loads(receipt.read_text()); assert saved['plan_sha256'] == sha(plan_path) and saved['predictions_sha256'] == sha(predfile)
            allrows.extend(saved['pairs']); continue
        arrays = np.load(root / 'inputs' / job / 'arrays.npz', allow_pickle=False)
        with (root / 'inputs' / job / 'held-rows.csv').open(newline='') as f: rows = list(csv.DictReader(f))
        model = TabICLClassifier(model_path=checkpoint, allow_auto_download=False, **plan['parameters'])
        start = time.perf_counter(); model.fit(arrays['x_train'], arrays['y_train'])
        predictions = []
        chunk = plan['inference_chunk_rows']
        for offset in range(0, len(rows), chunk):
            probability = model.predict_proba(arrays['x_held'][offset:offset+chunk])
            assert np.array_equal(model.classes_, [0, 1]) and np.isfinite(probability).all()
            predictions.append(probability[:, 1])
        scores = {'equity': arrays['equity'], 'sequence': arrays['sequence'], 'tabicl': np.concatenate(predictions)}
        pairs = {}
        for i, row in enumerate(rows): pairs.setdefault(row['pair_id'], []).append(i)
        fusion = np.zeros(len(rows))
        for indices in pairs.values():
            for member in ['equity', 'tabicl']:
                fusion[indices] += .5 / (plan['rrf_k'] + rankdata(-scores[member][indices], method='average'))
        scores['rrf_equity_tabicl'] = fusion
        scores['soft_gate'] = np.where(np.array([r['predicted_behavior'] for r in rows]) == 'soft_play', fusion, scores['equity'])
        records = [r for candidate, score in scores.items() for r in pair_results(rows, score, candidate, job)]
        np.savez_compressed(predfile, **scores)
        result = {'plan_sha256': sha(plan_path), 'predictions_sha256': sha(predfile), 'seconds': time.perf_counter()-start,
            'pairs': records, 'torch': torch.__version__, 'cuda': torch.version.cuda}
        receipt.write_text(json.dumps(result, indent=2) + '\n'); allrows.extend(records)
        print(json.dumps({'job': job, 'seconds': result['seconds'], 'map5': {c: float(np.mean([r['map5'] for r in records if r['candidate'] == c])) for c in scores}}), flush=True)
        del model; torch.cuda.empty_cache()
    lookup = {(r['pair_id'], r['candidate']): r for r in allrows}
    comparisons = {}; passing = []
    for candidate in plan['candidates']:
        items = [r for r in allrows if r['candidate'] == candidate]
        deltas = [r['map5'] - lookup[r['pair_id'], 'equity']['map5'] for r in items]
        byjob = {j: float(np.mean([r['map5'] - lookup[r['pair_id'], 'equity']['map5'] for r in items if r['job'] == j])) for j in plan['jobs']}
        byfamily = {f: float(np.mean([r['map5'] - lookup[r['pair_id'], 'equity']['map5'] for r in items if r['family'] == f])) for f in sorted(set(r['family'] for r in items))}
        passed = np.mean(deltas) >= .005 and min(byjob.values()) >= -.005 and min(byfamily.values()) >= -.025
        comparisons[candidate] = {'delta': float(np.mean(deltas)), 'split_deltas': byjob, 'family_deltas': byfamily, 'passed': bool(passed)}
        if passed: passing.append(candidate)
    selected = max(passing, key=lambda c: comparisons[c]['delta']) if passing else None
    (out / 'summary.json').write_text(json.dumps({'comparisons': comparisons, 'selected_for_review': selected, 'pairs': allrows}, indent=2) + '\n')
    print(json.dumps({'selected_for_review': selected, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
