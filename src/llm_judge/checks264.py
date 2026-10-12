"""STEP264: run every registered assertion over the built package and write checks264.json.

examples.txt is written by make_examples.py; this script only hashes it.
Nothing here trains, downloads or uploads.  Usage:  python checks264.py
"""
from __future__ import annotations

import os

os.environ.setdefault('POLARS_MAX_THREADS', '3')

import hashlib
import json
import math
import re
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import build_dataset as B                                                    # noqa: E402
import render_hand as R                                                      # noqa: E402

OUT = B.OUT
DATA = R.DATA
K = ['pair_id', 'hand_id']
ID_RE = re.compile(r'[PHUT][0-9A-F]{8,}')
TIME_RE = re.compile(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\bUTC\b|started_at', re.IGNORECASE)
ORDER_RE = re.compile(r'\bhand\s*(?:#|no\.?|number|index|idx)\b|\bhand_id\b|\btable_id\b'
                      r'|\bpair_id\b|\bplayer_id\b|\bseat_no\b|\bstarted\b', re.IGNORECASE)
NET_RE = re.compile(r'^  Net \(bb\): (.+)\.$', re.MULTILINE)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> list:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def chi2_2x2(t) -> tuple:
    t = np.asarray(t, float)
    n = t.sum()
    exp = np.outer(t.sum(1), t.sum(0)) / n
    chi = float(((t - exp) ** 2 / exp).sum())
    p = math.erfc(math.sqrt(chi / 2.0))          # 1 dof survival function
    return round(chi, 4), round(p, 4)


def main() -> None:
    res: dict = {'step': 'STEP264', 'date': '2026-09-18'}
    tr = load_jsonl(OUT + r'\train.jsonl')
    ho = load_jsonl(OUT + r'\heldout.jsonl')
    ktr = pl.read_parquet(OUT + r'\keys_train.parquet')
    kho = pl.read_parquet(OUT + r'\keys_heldout.parquet')
    prompts = {r['uid']: r['prompt'] for r in tr + ho}
    assert len(prompts) == len(tr) + len(ho), 'duplicate uid across the two files'

    # ---------------------------------------------------------------- 1 development phase
    hids = pl.concat([ktr.select('hand_id'), kho.select('hand_id')]).unique()
    ph = (pl.scan_parquet(DATA + r'\hands.parquet').join(hids.lazy(), on='hand_id')
          .select('hand_id', 'phase', 'players_dealt').collect())
    res['development_phase'] = {
        'unique_hands': hids.height, 'rows_found': ph.height,
        'all_development': bool((ph['phase'] == 'development').all()),
        'all_six_handed': bool((ph['players_dealt'] == 6).all())}
    assert res['development_phase']['all_development'] and ph.height == hids.height
    assert res['development_phase']['all_six_handed']

    # ---------------------------------------------------------------- 2 split disjointness
    ov = {}
    for col in ('table_id', 'pair_id', 'hand_id', 'uid'):
        a, b = set(ktr[col].to_list()), set(kho[col].to_list())
        ov[col] = len(a & b)
    res['split'] = {
        'train_folds': sorted({int(x) for x in ktr['fold'].to_list()}),
        'heldout_folds': sorted({int(x) for x in kho['fold'].to_list()}),
        'overlap_counts': ov,
        'n_tables_train': ktr['table_id'].n_unique(), 'n_tables_heldout': kho['table_id'].n_unique(),
        'n_pairs_train': ktr['pair_id'].n_unique(), 'n_pairs_heldout': kho['pair_id'].n_unique()}
    assert all(v == 0 for v in ov.values()), res['split']

    # ---------------------------------------------------------------- 3/4 prompt hygiene
    seats = (pl.scan_parquet(DATA + r'\seats.parquet').join(hids.lazy(), on='hand_id')
             .select('hand_id', 'player_id').collect())
    players_of: dict[str, list] = {}
    for h, p in seats.iter_rows():
        players_of.setdefault(h, []).append(p)
    id_hits = time_hits = order_hits = literal_hits = 0
    bad_examples = []
    keymap = pl.concat([ktr.select('uid', *K, 'table_id'), kho.select('uid', *K, 'table_id')])
    for uid, pid, hid, tid in keymap.iter_rows():
        t = prompts[uid]
        if ID_RE.search(t):
            id_hits += 1
            bad_examples.append(('id', uid, ID_RE.search(t).group(0)))
        if TIME_RE.search(t):
            time_hits += 1
            bad_examples.append(('time', uid, TIME_RE.search(t).group(0)))
        if ORDER_RE.search(t):
            order_hits += 1
            bad_examples.append(('order', uid, ORDER_RE.search(t).group(0)))
        for lit in (pid, hid, tid, *players_of[hid]):
            if lit in t:
                literal_hits += 1
                bad_examples.append(('literal', uid, lit))
                break
    res['prompt_hygiene'] = {
        'prompts_scanned': len(prompts),
        'id_like_token_hits  [PHUT][0-9A-F]{8,}': id_hits,
        'timestamp_or_started_at_hits': time_hits,
        'order_index_or_column_name_hits': order_hits,
        'literal_pair/hand/table/player_id_hits': literal_hits,
        'examples': bad_examples[:5]}
    assert id_hits == 0 and time_hits == 0 and order_hits == 0 and literal_hits == 0, bad_examples[:5]

    # ---------------------------------------------------------------- 5 A/B randomisation
    res['ab_randomisation'] = {}
    for name, rows in (('train', tr), ('heldout', ho)):
        a1 = np.array([bool(r['a_is_player_1']) for r in rows])
        y = np.array([int(r['label']) for r in rows])
        tab = [[int(((y == lab) & (a1 == v)).sum()) for v in (False, True)] for lab in (0, 1)]
        chi, p = chi2_2x2(tab)
        res['ab_randomisation'][name] = {
            'share_a_is_player_1': round(float(a1.mean()), 4),
            'table_rows_label_0_then_1_cols_a_is_player_1_False_then_True': tab,
            'share_by_label': {'label_0': round(float(a1[y == 0].mean()), 4),
                               'label_1': round(float(a1[y == 1].mean()), 4)},
            'chi2_1dof': chi, 'p_value': p}
        assert .45 < a1.mean() < .55, 'A/B share off .5'
        assert p > .01, 'a_is_player_1 is not independent of the label'

    # ---------------------------------------------------------------- 6 class counts
    res['counts'] = {}
    for name, kdf, rows in (('train', ktr, tr), ('heldout', kho, ho)):
        res['counts'][name] = {
            'n': len(rows),
            'by_class': {c: int(n) for c, n in
                         kdf.group_by('source_class').len().sort('source_class').iter_rows()},
            'by_class_and_family': {f'{c}|{f}': int(n) for c, f, n in
                                    kdf.group_by('source_class', 'family').len()
                                    .sort('source_class', 'family').iter_rows()},
            'label_1': int(kdf['label'].sum()),
            'label_0': int((kdf['label'] == 0).sum()),
            'n_pairs': kdf['pair_id'].n_unique(), 'n_hands': kdf['hand_id'].n_unique()}

    # ---------------------------------------------------------------- 7 token lengths
    res['token_lengths_chars_over_3.6'] = {}
    for name, rows in (('train', tr), ('heldout', ho)):
        t = np.array([len(r['prompt']) for r in rows]) / 3.6
        res['token_lengths_chars_over_3.6'][name] = {
            'mean': round(float(t.mean()), 1),
            'quantiles': {str(q): round(float(np.quantile(t, q / 100)), 1)
                          for q in (0, 1, 5, 25, 50, 75, 90, 95, 99, 100)},
            'share_over_800': round(float((t > 800).mean()), 4),
            'share_over_900': round(float((t > 900).mean()), 4),
            'share_under_350': round(float((t < 350).mean()), 4)}
    per_class = {}
    for name, kdf, rows in (('train', ktr, tr), ('heldout', kho, ho)):
        L = {r['uid']: len(r['prompt']) / 3.6 for r in rows}
        for c in sorted(set(kdf['source_class'].to_list())):
            v = np.array([L[u] for u in kdf.filter(pl.col('source_class') == c)['uid'].to_list()])
            per_class[f'{name}|{c}'] = {'n': len(v), 'median': round(float(np.median(v)), 1)}
    res['token_lengths_chars_over_3.6']['median_by_class'] = per_class

    # ---------------------------------------------------------------- 8 determinism
    rng = np.random.default_rng(264)
    sample = keymap[rng.choice(keymap.height, 300, replace=False)]
    ctx, render = R.make_renderer(sample.select(*K))
    roles = {r['uid']: bool(r['a_is_player_1']) for r in tr + ho}
    same_twice = same_as_file = 0
    for uid, pid, hid, _t in sample.iter_rows():
        a = render(pid, hid, roles[uid])
        b = render(pid, hid, roles[uid])
        same_twice += (a == b)
        same_as_file += (a == prompts[uid])
    res['renderer_determinism'] = {'sampled': sample.height, 'identical_on_rerender': same_twice,
                                   'identical_to_the_jsonl': same_as_file}
    assert same_twice == sample.height and same_as_file == sample.height

    # ---------------------------------------------------------------- 9 chip accounting
    rng = np.random.default_rng(2640)
    chk = keymap[rng.choice(keymap.height, 200, replace=False)]
    sums, worst = [], None
    for uid, _pid, _hid, _t in chk.iter_rows():
        m = NET_RE.search(prompts[uid])
        assert m, 'no Net line in the prompt'
        vals = [float(x.rsplit(' ', 1)[1]) for x in m.group(1).split(', ')]
        assert len(vals) == 6, 'not six net entries'
        sums.append(sum(vals))
        if worst is None or abs(sum(vals)) > abs(worst):
            worst = sum(vals)
    sums = np.array(sums)
    res['chip_accounting_200_hands'] = {
        'max_abs_net_sum_bb': round(float(np.abs(sums).max()), 4),
        'mean_abs_net_sum_bb': round(float(np.abs(sums).mean()), 5),
        'share_exactly_zero': round(float((np.abs(sums) < 1e-9).mean()), 3),
        'explanation': 'net_chips sums to exactly 0 chips over the six seats (verified on the raw '
                       'table: no rake). The prompt prints each net rounded to one decimal of a '
                       'big blind, so the printed six can miss zero by up to 6 * 0.05 = 0.3 bb.'}
    assert np.abs(sums).max() <= 0.31, res['chip_accounting_200_hands']

    # ---------------------------------------------------------------- 10 GPU-side purity
    # reward.py / trace_logger.py / select_best_traces.py run on the rented box: they may import
    # only the standard library (plus each other).
    stdlib = set(sys.stdlib_module_names) | {'reward', 'trace_logger', 'select_best_traces'}
    imports = {}
    for f in ('reward.py', 'trace_logger.py', 'select_best_traces.py'):
        mods = set()
        for ln in open(HERE + '\\' + f, encoding='utf-8'):
            m = re.match(r'\s*(?:from|import)\s+([A-Za-z_][\w.]*)', ln)
            if m:
                mods.add(m.group(1).split('.')[0])
        imports[f] = sorted(mods)
        assert mods <= stdlib, f'{f} imports a third-party module: {sorted(mods - stdlib)}'
    res['gpu_side_pure_stdlib'] = imports

    # ---------------------------------------------------------------- 11 the vm scripts
    vm = OUT + r'\vm'
    vmfiles = ('train_grpo_qlora.py', 'score_heldout.py', 'rejection_sample.py')
    import py_compile
    import tempfile
    tmpc = tempfile.mkdtemp(prefix='step264_pyc_')
    vmres = {}
    for f in vmfiles:
        p = vm + '\\' + f
        py_compile.compile(p, doraise=True, cfile=os.path.join(tmpc, f + 'c'))
        head = open(p, encoding='utf-8').read(2000)
        vmres[f] = {'py_compile': True, 'has_untested_banner': 'UNTESTED' in head,
                    'has_smoke_flag': '--smoke' in open(p, encoding='utf-8').read()}
        assert vmres[f]['has_untested_banner'] and vmres[f]['has_smoke_flag'], f
    vmres['note'] = ('these three run on the rented GPU only; they were never executed on a GPU '
                     'and torch is not installed on this machine. The reward + trace-logging '
                     'hook of train_grpo_qlora.py WAS unit-tested locally with a fake tokenizer.')
    res['vm_scripts'] = vmres

    # ---------------------------------------------------------------- 12 files
    res['files'] = {}
    for f in ('registration264.json', 'train.jsonl', 'heldout.jsonl', 'keys_train.parquet',
              'keys_heldout.parquet', 'build_stats264.json', 'examples.txt', 'README.txt'):
        p = OUT + '\\' + f
        res['files'][f] = {'bytes': os.path.getsize(p), 'sha256': sha256(p)}
    for f in vmfiles + ('requirements.txt', 'RUNBOOK.txt'):
        p = vm + '\\' + f
        res['files']['vm/' + f] = {'bytes': os.path.getsize(p), 'sha256': sha256(p)}
    for f in ('render_hand.py', 'build_dataset.py', 'reward.py', 'trace_logger.py',
              'select_best_traces.py', 'score_heldout_eval.py', 'make_examples.py',
              'checks264.py'):
        p = HERE + '\\' + f
        res['files']['src/llm_judge/' + f] = {'bytes': os.path.getsize(p), 'sha256': sha256(p)}

    json.dump(res, open(OUT + r'\checks264.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps(res, indent=1))
    print('wrote', OUT + r'\checks264.json')


if __name__ == '__main__':
    main()
