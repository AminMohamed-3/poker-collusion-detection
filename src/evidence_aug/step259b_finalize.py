"""STEP259b: record the sha256 of the two gate check CSVs in receipt259b.json, then remove them.

The brief allows the gate check files to be removed at the end ONLY if they are byte-identical to the
files they reproduce, and only after their shas are recorded in the receipt.  Both conditions are
re-verified here from disk; if either fails nothing is removed.  Only files this step created are ever
touched.

Usage: python step259b_finalize.py
"""
import os
for _key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src'))
from evidence_aug.paths import sha, write_json                                 # noqa: E402

CV = POKER / 'Claude Version'
OUT = CV / 'outputs_step259b_20260918'
STEP251_SHA = '804c52f2d4a9e2fdffc91769e48edf713ac489d745874456025e2ee95e7761fb'
STEP259_SHA = '938e9e71997617c4c91b1003051aca00505300c384753bbcc6325d44ea5b89b9'
CHECKS = {'neutral_check_259b.csv': (STEP251_SHA, 'outputs_step251_20260917/'
                                     'submission_step251_ens4_dtfp3_on_step222.csv'),
          'arith_check_259b.csv': (STEP259_SHA, 'outputs_step259_20260918/'
                                   'submission_step259_mechfusion_dt_sp_on_step251.csv')}


def main() -> None:
    rec = json.loads((OUT / 'receipt259b.json').read_text(encoding='utf-8'))
    if 'check_csvs_removed' in rec:
        print('already finalized')
        return
    info = {}
    for name, (want, twin) in CHECKS.items():
        p = OUT / name
        assert p.exists(), p
        got = sha(p)
        assert got == want, ('the check file is NOT byte-identical to the file it reproduces', name,
                             got, want)
        assert sha(CV / twin) == want, 'the registered twin changed'
        info[name] = {'sha256': got, 'identical_to': twin, 'bytes': p.stat().st_size,
                      'removed': True}
    rec['check_csvs_removed'] = info
    write_json(OUT / 'receipt259b.json', rec)
    for name in CHECKS:
        (OUT / name).unlink()
        print('removed', name, info[name]['sha256'])
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
