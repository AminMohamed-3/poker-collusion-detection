"""Step 107 (Claude): the step-104 typed decoder over the frozen step-98 S* held scores, plus controls. Label-free build and freeze only.
Registration: experiments/stacked_typed_decoder_C_20260913/prediction-stacked-typed-decoder.json. Held truth is read only by step107b."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
import time
from datetime import datetime, timezone
from evidence_aug.step104_C_typed_decoder_build import decode, chronology, align, OUT as S104, AUDIT_SCORES, KEYS

OUT = CV / 'experiments/stacked_typed_decoder_C_20260913'
PREREG = OUT / 'prediction-stacked-typed-decoder.json'
PREREG_SHA = 'c47a83ba1c551c1a61e7662a970e3478c1761d3bdac384eade96f45799778505'
S98 = CV / 'experiments/prefix_strength_confirmation_C_20260913'
AUDIT_POINT, AUDIT_TOL = .043203, 1e-6
CODEX_READ = [HERE / 'src/audit_common.py']


def audit_replay():
    """Training-only replay of the registered audit look: decoder(S*) vs B0 on the 162-query audit."""
    d = pl.read_parquet(AUDIT / 'audit-oof.parquet', columns=KEYS + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(KEYS)
    sc = pl.read_parquet(AUDIT_SCORES, columns=KEYS + ['table_id', 'B0', 'prefix_strength']).sort(KEYS)
    assert sc.select(KEYS).equals(d.select(KEYS)) and np.array_equal(sc['B0'].to_numpy() > 5, sc['prefix_strength'].to_numpy() > 5)
    fa = chronology(sc.select(KEYS + ['table_id', 'B0']).rename({'B0': 'score'}))
    assert fa.select(KEYS).equals(sc.select(KEYS))
    ta = align(fa, pl.read_parquet(S104 / 'audit-type-attributes.parquet'))
    fs = fa.with_columns(pl.Series('score', sc['prefix_strength'].to_numpy()))
    c = evaluate(d, {'incumbent': fa['score'].to_numpy(), 'typed_S_star': decode(fs, ta)})['comparisons']['typed_S_star']
    replay = json.loads(json.dumps({k: c[k] for k in ('mean_delta', 'ci95', 'by_family', 'halves', 'gate_pass')}, default=lambda o: o.item()))
    replay.update(T1_rows=int(ta.sum()), within_tolerance=bool(abs(float(c['mean_delta']) - AUDIT_POINT) <= AUDIT_TOL))
    return replay


def main():
    start = time.perf_counter()
    assert not (OUT / 'predictions-frozen.json').exists() and not (OUT / 'scores').exists(), 'Preserve a completed freeze'
    assert sha(PREREG) == PREREG_SHA
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    for k, v in reg['inputs'].items():
        assert sha(ROOT / v['path']) == v['sha256'], k
    protected, snap0 = check_protected(), {str(p): sha(p) for p in CODEX_READ}
    frozen98 = json.loads((S98 / 'predictions-frozen.json').read_text(encoding='utf-8'))
    frozen104 = json.loads((S104 / 'predictions-frozen.json').read_text(encoding='utf-8'))
    replay = audit_replay()
    print('audit replay', json.dumps(replay), flush=True)
    if not replay['within_tolerance']:
        write(OUT / 'stopped-audit-replay.json', replay)
        raise SystemExit('audit replay outside tolerance; no held build')
    (OUT / 'scores').mkdir()
    receipts = []
    for fold in (0, 1, 2):
        arms = {}
        for a in ('B0', 'S_star'):
            f = S98 / 'scores' / f'fold{fold}-{a}.parquet'
            assert sha(f) == frozen98['scores'][f.name], f.name
            arms[a] = pl.read_parquet(f).sort(KEYS)
        assert arms['S_star'].select(KEYS).equals(arms['B0'].select(KEYS))
        assert np.array_equal(arms['S_star']['score'].to_numpy() > 5, arms['B0']['score'].to_numpy() > 5)
        fb = chronology(arms['B0'])
        fs = fb.with_columns(pl.Series('score', arms['S_star']['score'].to_numpy()))
        af = S104 / f'fold{fold}-type-attributes.parquet'
        assert sha(af) == frozen104['attributes'][af.name], af.name
        t1 = align(fb, pl.read_parquet(af))
        assert int(t1.sum()) == frozen104['fold_receipts'][fold]['T1_rows']
        of = S104 / 'scores' / f'fold{fold}-typed_B0.parquet'
        assert sha(of) == frozen104['scores'][of.name], of.name
        old = pl.read_parquet(of).sort(KEYS)
        assert old.select(KEYS).equals(fb.select(KEYS)) and np.array_equal(decode(fb, t1), old['score'].to_numpy()), 'typed_B0 rebuild differs from step 104'
        new = {'typed_S_star': decode(fs, t1), 'time_only_S_star': decode(fs, np.zeros(len(t1), bool)),
               'shuffled_type_S_star': decode(fs, t1, np.random.default_rng(104))}
        for name, p in new.items():
            assert np.isfinite(p).all() and np.array_equal(p > 5, fb['score'].to_numpy() > 5)
            fb.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', p)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        receipts.append({'fold': fold, 'rows': fb.height, 'top20_rows': int((fb['score'] > 5).sum()), 'pairs': fb['pair_id'].n_unique(),
                         'T1_rows': int(t1.sum()), 'typed_B0_rebuilt_exact': True})
        print('fold', fold, receipts[-1], round(time.perf_counter() - start, 1), flush=True)
    frozen = {'created_at': datetime.now(timezone.utc).isoformat(), 'registration_sha256': PREREG_SHA, 'audit_replay': replay,
              'base_scores_step98': {f'fold{f}-{a}.parquet': frozen98['scores'][f'fold{f}-{a}.parquet'] for f in (0, 1, 2)
                                     for a in ('B0', 'S_star', 'incumbent', 'stage1_only')},
              'decoder_scores_step104': {f'fold{f}-typed_B0.parquet': frozen104['scores'][f'fold{f}-typed_B0.parquet'] for f in (0, 1, 2)},
              'attributes_step104': {f'fold{f}-type-attributes.parquet': frozen104['attributes'][f'fold{f}-type-attributes.parquet'] for f in (0, 1, 2)},
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').iterdir())}, 'fold_receipts': receipts,
              'script_sha256': sha(Path(__file__)), 'step104_module_sha256': sha(CV / 'src/evidence_aug/step104_C_typed_decoder_build.py'),
              'protected_unchanged': check_protected() == protected, 'codex_files_unchanged': snap0 == {str(p): sha(p) for p in CODEX_READ},
              'seconds': time.perf_counter() - start, 'held_evidence_evaluation': False}
    assert frozen['protected_unchanged'] and frozen['codex_files_unchanged']
    write(OUT / 'predictions-frozen.json', frozen)
    print('PREDICTIONS FROZEN', round(frozen['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
