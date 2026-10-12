"""STEP267: average several reranker score files into one harness-format score file.

ENS of registration267.json: per architecture the mean of the 3 seeds' probabilities, then the
mean of the two architectures.  This script does ONE mean over the files it is given, so the
two-level mean is two calls:

    python ensemble_scores.py --inputs runs/FT_A_s267/scores.jsonl runs/FT_A_s268/scores.jsonl \\
        runs/FT_A_s269/scores.jsonl --out runs/ENS_A/scores.jsonl
    python ensemble_scores.py --inputs runs/FT_B_s267/scores.jsonl ... --out runs/ENS_B/scores.jsonl
    python ensemble_scores.py --inputs runs/ENS_A/scores.jsonl runs/ENS_B/scores.jsonl \\
        --out runs/ENS/scores.jsonl

The averaged quantity is p_primary = mean_conf / 100 of each input row, so whatever --primary each
input was scored with is what is averaged.  Every input must cover the same uid set (asserted);
the output carries mean_conf = 100 * mean(p) and k = the number of files averaged.

NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU - but this file needs no GPU and its --dry-run has
been executed locally.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rr_common as C                                                          # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='mean of p[primary] across reranker score files')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-partial', action='store_true', dest='allow_partial',
                    help='average over the uids present in every file instead of asserting')
    ap.add_argument('--dry-run', action='store_true', dest='dry_run')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    files = []
    for p in a.inputs:
        rows = C.read_jsonl(p)
        assert rows, 'empty score file ' + p
        m = {}
        for r in rows:
            assert r['uid'] not in m, 'duplicate uid in ' + p
            m[r['uid']] = float(r['mean_conf']) / 100.0
        files.append((p, m))
        print('%s: %d rows' % (p, len(m)))

    sets = [set(m) for _p, m in files]
    common = set.intersection(*sets)
    union = set.union(*sets)
    if not a.allow_partial:
        assert common == union, ('the inputs do not cover the same uids (%d common, %d union) - '
                                 'pass --allow-partial to average the intersection'
                                 % (len(common), len(union)))
    order = [r['uid'] for r in C.read_jsonl(files[0][0]) if r['uid'] in common]
    out = [{'uid': u, 'k': len(files),
            'mean_conf': round(100.0 * sum(m[u] for _p, m in files) / len(files), 4),
            'p': {'ensemble_mean': round(sum(m[u] for _p, m in files) / len(files), 6)},
            'inputs': [os.path.basename(os.path.dirname(p)) or os.path.basename(p)
                       for p, _m in files]}
           for u in order]
    if a.dry_run:
        print('DRY RUN: would write %d rows to %s; first row %s'
              % (len(out), a.out, json.dumps(out[0]) if out else 'NONE'))
        return 0
    C.write_jsonl(a.out, out)
    conf = [r['mean_conf'] for r in out]
    json.dump({'n': len(out), 'inputs': a.inputs,
               'mean_conf': {'mean': round(sum(conf) / max(len(conf), 1), 4),
                             'min': round(min(conf), 4) if conf else None,
                             'max': round(max(conf), 4) if conf else None}},
              open(os.path.splitext(a.out)[0] + '_summary.json', 'w', encoding='utf-8'), indent=1)
    print('wrote %s (%d rows, mean of %d files)' % (a.out, len(out), len(files)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
