"""STEP262 runner: a lock-free job queue over the (surface, shard) feature/score jobs.

Each worker claims the next unclaimed job with an atomic mkdir and runs
step262_tail_features.shard for it.  Workers may be started at any time, so the pool can grow from
4 (while the STEP257 evaluation build is still running) to 8 without re-splitting the work.

Usage:  python step262_run.py NSHARDS [TAG]
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ.setdefault(_k, '1')
import sys, time, traceback
sys.dont_write_bytecode = True
C = 'C:/Data/Projects/Money Making/work/poker/Claude Version/'
sys.path.insert(0, C + 'src')
from pu_pairs import step262_tail_features as TF                        # noqa: E402

OUT = TF.OUT
CLAIM = OUT + 'claims/'


def main():
    n = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else str(os.getpid())
    os.makedirs(CLAIM, exist_ok=True)
    jobs = [(s, i) for s in ('prefix80', 'prefix60', 'prefix40') for i in range(n)]
    t0 = time.time()
    done = 0
    for surface, i in jobs:
        name = f'{surface}_s{n:03d}_{i:03d}'
        try:
            os.mkdir(CLAIM + name)
        except FileExistsError:
            continue
        open(CLAIM + name + '/by.txt', 'w', encoding='utf-8').write(tag)
        try:
            TF.shard(surface, i, n)
            done += 1
        except Exception:
            open(CLAIM + name + '/error.txt', 'w', encoding='utf-8').write(traceback.format_exc())
            raise
        print(f'WORKER {tag} finished {name} ({done} jobs, {round(time.time() - t0)}s)', flush=True)
    print(f'WORKER {tag} EXIT after {done} jobs, {round(time.time() - t0)}s', flush=True)


if __name__ == '__main__':
    main()
