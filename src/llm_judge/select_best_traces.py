"""STEP264: pick the traces worth distilling into a small (0.6B) student.  PURE STANDARD LIBRARY.

Input   trace files written by trace_logger.py  +  train.jsonl (the TRAIN uid whitelist and the
        prompt text; the JSONL carries no pair / hand id, only the opaque uid).
Output  sft_traces.jsonl  (prompt, completion, uid, label, family, source_class, reward,
        n_think_tokens, teacher_mean_conf)  +  sft_stats.json

HARD RULE, asserted twice: a held-out uid can never be emitted.  Only uids present in the
supplied train.jsonl survive, and the emitted set is asserted to be a subset of it.  If a
--heldout-jsonl is given its uids are asserted to be disjoint from the output as well.  Rollouts
whose checkpoint tag looks like held-out scoring ('final_heldout', anything containing 'heldout')
are dropped before anything else.

Selection
  keep a trace only if   verdict is correct for the label (label 1 -> the pair's family,
                         label 0 -> NONE)
                   and   |confidence/100 - label| <= .2
                   and   it mentions at least one concrete street / action word
                   and   it contains no ID-like token  (regex [PHUT][0-9A-F]{8,})
                   and   finish_reason is a normal stop
  per uid keep at most 3, preferring higher total reward, then SHORTER thinking, and requiring
  pairwise word-5-gram Jaccard < .7 between the kept traces of that uid
  balance label 1 against label 0 at 50/50 and the three families evenly, by down-sampling
  over-sample NEG_HARD and NEG_WITHIN x2 inside the negative pool (the instructive negatives)
  teacher_mean_conf = the mean parsed confidence over ALL of that uid's rollouts from the FINAL
  checkpoint (the rejected ones included: it is the teacher's belief, not the kept sample's)

Usage
  python select_best_traces.py --traces traces/ --train-jsonl train.jsonl --out sft_traces.jsonl
        [--heldout-jsonl heldout.jsonl] [--final-checkpoint final_rs] [--max-per-uid 3]
        [--jaccard 0.7] [--conf-tol 0.2] [--seed 264]
  python select_best_traces.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_logger import read_traces                                         # noqa: E402

ID_RE = re.compile(r'[PHUT][0-9A-F]{8,}')
WORD_RE = re.compile(r"[a-z0-9']+")
CONCRETE = {'preflop', 'flop', 'turn', 'river', 'board', 'bet', 'bets', 'betting', 'raise',
            'raises', 'raised', 'call', 'calls', 'called', 'check', 'checks', 'checked', 'fold',
            'folds', 'folded', 'showdown', 'pot', 'blind', 'blinds', 'button', 'all-in'}
GOOD_FINISH = {'stop', 'eos', 'end_turn', 'stop_sequence', None, ''}
FAMS = ('directed_transfer', 'soft_play', 'coordinated_isolation')
FAM_VERDICT = {'directed_transfer': 'DIRECTED_TRANSFER', 'soft_play': 'SOFT_PLAY',
               'coordinated_isolation': 'COORDINATED_ISOLATION'}
OVERSAMPLE = {'NEG_HARD': 2, 'NEG_WITHIN': 2}


def ngrams(text: str, n: int = 5) -> set:
    w = WORD_RE.findall((text or '').lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def load_jsonl(path: str) -> list:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def keep_trace(t: dict, conf_tol: float) -> bool:
    if not t.get('format_ok'):
        return False
    if (t.get('finish_reason') or '') not in GOOD_FINISH:
        return False
    label = int(t['label'])
    want = FAM_VERDICT.get(str(t.get('family')), 'NONE') if label == 1 else 'NONE'
    if t.get('verdict') != want:
        return False
    conf = t.get('confidence')
    if conf is None or abs(conf / 100.0 - label) > conf_tol:
        return False
    c = t.get('completion') or ''
    if ID_RE.search(c):
        return False
    if not (CONCRETE & set(WORD_RE.findall(c.lower()))):
        return False
    return True


def select(traces, train_rows, heldout_uids=None, final_checkpoint=None, max_per_uid=3,
           jac_thr=.7, conf_tol=.2, seed=264) -> tuple:
    train_uids = {r['uid'] for r in train_rows}
    prompt_of = {r['uid']: r['prompt'] for r in train_rows}
    rows = [t for t in traces
            if 'heldout' not in str(t.get('checkpoint') or '').lower()
            and t.get('uid') in train_uids]
    stats = {'traces_seen': 0, 'traces_on_train_uids': len(rows)}

    ckpts = sorted({str(t.get('checkpoint')) for t in rows})
    final = final_checkpoint or (ckpts[-1] if ckpts else None)
    conf_sum: dict[str, list] = {}
    for t in rows:
        if str(t.get('checkpoint')) == final and t.get('confidence') is not None:
            conf_sum.setdefault(t['uid'], []).append(float(t['confidence']))
    teacher = {u: sum(v) / len(v) for u, v in conf_sum.items()}

    kept_all = [t for t in rows if keep_trace(t, conf_tol)]
    stats['traces_passing_filters'] = len(kept_all)

    by_uid: dict[str, list] = {}
    for t in kept_all:
        by_uid.setdefault(t['uid'], []).append(t)
    picked = []
    for u in sorted(by_uid):
        cand = sorted(by_uid[u], key=lambda t: (-float(t['total']),
                                                int(t.get('n_think_tokens') or 0),
                                                len(t.get('completion') or '')))
        keep, grams = [], []
        for t in cand:
            g = ngrams(t.get('completion') or '')
            if all(jaccard(g, h) < jac_thr for h in grams):
                keep.append(t)
                grams.append(g)
            if len(keep) >= max_per_uid:
                break
        picked.extend(keep)
    stats['traces_after_per_uid_dedup'] = len(picked)

    rng = random.Random(seed)
    pos = {f: sorted([t for t in picked if int(t['label']) == 1 and t.get('family') == f],
                     key=lambda t: (t['uid'], -float(t['total']))) for f in FAMS}
    stats['pos_by_family_before'] = {f: len(pos[f]) for f in FAMS}
    neg_pool = []
    for t in sorted([t for t in picked if int(t['label']) == 0],
                    key=lambda t: (t['uid'], -float(t['total']))):
        neg_pool.extend([t] * OVERSAMPLE.get(t.get('source_class'), 1))
    rng.shuffle(neg_pool)
    stats['neg_pool_after_oversample'] = len(neg_pool)

    # even families AND 50/50 labels: one quota per family, bounded by the negative pool
    n_fam = min(len(pos[f]) for f in FAMS) if all(pos[f] for f in FAMS) else 0
    n_fam = min(n_fam, len(neg_pool) // len(FAMS))
    if n_fam == 0:                               # degenerate input: emit what there is, unbalanced
        out_rows = [t for f in FAMS for t in pos[f]] + neg_pool
        stats['balanced'] = False
    else:
        pos_sel = []
        for f in FAMS:
            v = list(pos[f])
            rng.shuffle(v)
            pos_sel.extend(v[:n_fam])
        out_rows = pos_sel + neg_pool[:len(pos_sel)]
        stats['balanced'] = True
    rng.shuffle(out_rows)

    out = []
    for t in out_rows:
        out.append({'uid': t['uid'], 'prompt': prompt_of[t['uid']],
                    'completion': t['completion'], 'label': int(t['label']),
                    'family': t.get('family'), 'source_class': t.get('source_class'),
                    'reward': float(t['total']),
                    'n_think_tokens': t.get('n_think_tokens'),
                    'teacher_mean_conf': teacher.get(t['uid'])})
    emitted = {r['uid'] for r in out}
    assert emitted <= train_uids, 'a non-TRAIN uid reached sft_traces.jsonl'
    if heldout_uids:
        assert not (emitted & set(heldout_uids)), 'a HELD-OUT uid reached sft_traces.jsonl'
    stats.update({
        'final_checkpoint': final, 'checkpoints_seen': ckpts, 'n_out': len(out),
        'n_out_label1': sum(1 for r in out if r['label'] == 1),
        'n_unique_uids': len(emitted),
        'by_source_class': {c: sum(1 for r in out if r['source_class'] == c)
                            for c in sorted({r['source_class'] for r in out})} if out else {},
        'by_family': {f: sum(1 for r in out if r['family'] == f) for f in FAMS},
        'teacher_conf_available': sum(1 for r in out if r['teacher_mean_conf'] is not None),
        'params': {'max_per_uid': max_per_uid, 'jaccard': jac_thr, 'conf_tol': conf_tol,
                   'seed': seed, 'oversample': OVERSAMPLE},
    })
    return out, stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--train-jsonl', required=True)
    ap.add_argument('--heldout-jsonl', default=None)
    ap.add_argument('--out', default='sft_traces.jsonl')
    ap.add_argument('--stats', default=None)
    ap.add_argument('--final-checkpoint', default=None)
    ap.add_argument('--max-per-uid', type=int, default=3)
    ap.add_argument('--jaccard', type=float, default=.7)
    ap.add_argument('--conf-tol', type=float, default=.2)
    ap.add_argument('--seed', type=int, default=264)
    a = ap.parse_args(argv)
    train_rows = load_jsonl(a.train_jsonl)
    held = {r['uid'] for r in load_jsonl(a.heldout_jsonl)} if a.heldout_jsonl else None
    traces = list(read_traces(a.traces))
    out, stats = select(traces, train_rows, held, a.final_checkpoint, a.max_per_uid,
                        a.jaccard, a.conf_tol, a.seed)
    stats['traces_seen'] = len(traces)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=True) + '\n')
    sp = a.stats or (os.path.splitext(a.out)[0] + '_stats.json')
    json.dump(stats, open(sp, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(stats, indent=1))
    print('wrote', a.out, 'and', sp)


# ---------------------------------------------------------------------------- self-test


def _selftest() -> None:
    import shutil
    import tempfile
    from trace_logger import TraceLogger
    d = tempfile.mkdtemp(prefix='step264_sel_')
    try:
        train = [{'uid': f'tr{i:02d}', 'prompt': f'PROMPT {i}', 'label': 0, 'family': 'NONE',
                  'source_class': 'NEG_RANDOM', 'fold': 0, 'a_is_player_1': True}
                 for i in range(12)]
        # 3 positives per family + negatives of each class
        spec = []
        for j, f in enumerate(FAMS):
            for k in range(3):
                spec.append((f'tr{j * 3 + k:02d}', 1, f, 'POS'))
        for k, c in enumerate(('NEG_RANDOM', 'NEG_HARD', 'NEG_WITHIN')):
            spec.append((f'tr{9 + k:02d}', 0, 'NONE', c))
        good = {1: {f: f'On the flop A bet and B folded, a {f} pattern.\n'
                       f'VERDICT: {FAM_VERDICT[f]}\nCONFIDENCE: 9{n}'
                    for f in FAMS} for n in (0,)}[1]
        p = os.path.join(d, 'traces', 'run.jsonl.gz')
        with TraceLogger(p, run_id='t') as tl:
            for uid, label, fam, cls in spec:
                for s in range(3):
                    if label == 1:
                        comp = good[fam].replace('flop', ('flop', 'turn', 'river')[s])
                    else:
                        comp = (f'The pot was small and everyone folded {s} times '
                                f'on the {("flop", "turn", "river")[s]}.\n'
                                f'VERDICT: NONE\nCONFIDENCE: {s}')
                    tl.log(step=1, checkpoint='final_rs', uid=uid, fold=0, source_class=cls,
                           label=label, family=fam, sample_idx=s, temperature=.8,
                           completion=comp, n_think_tokens=100 + 10 * s, n_answer_tokens=10,
                           finish_reason='stop')
            # rubbish that must be filtered out
            tl.log(step=1, checkpoint='final_rs', uid='tr00', fold=0, source_class='POS',
                   label=1, family=FAMS[0], sample_idx=9, temperature=.8,
                   completion='no answer at all', finish_reason='stop')
            tl.log(step=1, checkpoint='final_rs', uid='tr01', fold=0, source_class='POS',
                   label=1, family=FAMS[0], sample_idx=9, temperature=.8,
                   completion='Hand HB69C76ADD34120 on the flop.\nVERDICT: '
                              'DIRECTED_TRANSFER\nCONFIDENCE: 95', finish_reason='stop')
            tl.log(step=1, checkpoint='final_rs', uid='tr09', fold=0, source_class='NEG_RANDOM',
                   label=0, family='NONE', sample_idx=9, temperature=.8,
                   completion='On the flop nothing happened.\nVERDICT: NONE\nCONFIDENCE: 5',
                   finish_reason='length')
            # a held-out uid that must never survive
            tl.log(step=1, checkpoint='final_heldout', uid='HELDOUT1', fold=3,
                   source_class='POS', label=1, family=FAMS[0], sample_idx=0, temperature=.7,
                   completion='On the flop A bet.\nVERDICT: DIRECTED_TRANSFER\nCONFIDENCE: 95',
                   finish_reason='stop')
        traces = list(read_traces(os.path.join(d, 'traces')))
        out, stats = select(traces, train, heldout_uids={'HELDOUT1'})
        uids = {r['uid'] for r in out}
        assert 'HELDOUT1' not in uids, '1 held-out uid never emitted'
        assert all(u.startswith('tr') for u in uids), '2 only TRAIN uids emitted'
        assert stats['n_out'] == 2 * stats['n_out_label1'], '3 labels balanced 50/50'
        assert len({r['label'] for r in out}) == 2, '4 both labels present'
        fam_counts = [stats['by_family'][f] for f in FAMS]
        assert len(set(fam_counts)) == 1, '5 families balanced'
        per_uid: dict = {}
        for r in out:                     # distinct traces; the x2 oversampling repeats rows
            per_uid.setdefault(r['uid'], set()).add(r['completion'])
        assert max(len(v) for v in per_uid.values()) <= 3, '6 at most 3 distinct traces per uid'
        assert not any(ID_RE.search(r['completion']) for r in out), '7 no ID-like token survives'
        assert all('VERDICT:' in r['completion'] for r in out), '8 every kept trace has an answer'
        assert all(r['teacher_mean_conf'] is not None for r in out), '9 teacher confidence filled'
        assert stats['final_checkpoint'] == 'final_rs', '10 final checkpoint detected'
        # the negative oversampling really doubled the two instructive classes
        assert stats['neg_pool_after_oversample'] > sum(
            1 for t in traces if int(t['label']) == 0 and t['uid'].startswith('tr')
            and keep_trace(t, .2)), '11 NEG_HARD / NEG_WITHIN oversampled'
        # determinism
        out2, _ = select(traces, train, heldout_uids={'HELDOUT1'})
        assert [r['uid'] for r in out] == [r['uid'] for r in out2], '12 selection deterministic'
        print('select_best_traces.py selftest: 12 assertions passed  '
              f'(kept {stats["n_out"]} of {len(traces)} traces)')
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        main()
