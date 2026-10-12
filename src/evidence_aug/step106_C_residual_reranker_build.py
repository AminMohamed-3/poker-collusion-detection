"""Step 106 (Claude): gameplay residual reranker over the step-104 typed listing decoder. Audit-only ridge GLM; label-free held build and freeze.
Registration: experiments/residual_reranker_C_20260913/prediction-residual-reranker.json. Held truth is read only by step106b."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Data/Projects/Money Making/work/poker/Codex Resume/src')
from audit_common import *
import time
from datetime import datetime, timezone
from evidence_aug import step106_attrs as A6
from evidence_aug import step106_features as F6
from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align

OUT = CV / 'experiments/residual_reranker_C_20260913'
PREREG = OUT / 'prediction-residual-reranker.json'
PREREG_SHA = '2ee464189ba41bb258629272422b98321094cb881fa1b792db0c6dad034fbde5'
S98 = CV / 'experiments/prefix_strength_confirmation_C_20260913'
S104 = CV / 'experiments/typed_listing_decoder_C_20260913'
AUDIT_SCORES = CV / 'experiments/codex_ideas_sweep_C_20260913/scores-singles.parquet'
CODEX_READ = [HERE / 'src/audit_common.py']
MODULES = [CV / 'src/evidence_aug/step106_attrs.py', CV / 'src/evidence_aug/step106_features.py', CV / 'src/evidence_aug/step104_C_typed_decoder_build.py']
KEYS = ['pair_id', 'hand_id']
AUDIT_POINT, AUDIT_TOL = .027586, .001
STRUCTURE = ['T1_before', 'T1_n', 'tpos', 'trank', 'prank', 'hi_before', 'p']
SEED = 106


def structure_only(x):
    return x.with_columns([pl.lit(0.).alias(c) for c in F6.FG if c not in STRUCTURE])


def arms(frame, attrs, t1, dec, models):
    x = F6.candidate_features(frame, attrs, t1, dec)
    xs = F6.candidate_features(frame, F6.shuffle_within_pair(attrs, SEED), t1, dec)
    assert xs.select(KEYS).equals(x.select(KEYS))
    return {'resid_B0': F6.to_scores(frame, x, F6.margin(models['gameplay'], x)),
            'resid_shuffled_B0': F6.to_scores(frame, x, F6.margin(models['gameplay'], xs)),
            'resid_structure_B0': F6.to_scores(frame, x, F6.margin(models['structure'], structure_only(x)))}


def audit(players, frozen104):
    d = pl.read_parquet(AUDIT / 'audit-oof.parquet', columns=KEYS + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(KEYS)
    fr = chronology(pl.read_parquet(AUDIT_SCORES, columns=KEYS + ['table_id', 'B0']).rename({'B0': 'score'}).sort(KEYS))
    assert fr.select(KEYS).equals(d.select(KEYS))
    ta = S104 / 'audit-type-attributes.parquet'
    assert sha(ta) == frozen104['attributes'][ta.name]
    t1 = align(fr, pl.read_parquet(ta))
    dec = decode(fr, t1)
    attrs = A6.build(fr.filter(pl.col('score') > 5).select(KEYS), players)
    attrs.write_parquet(OUT / 'audit-reranker-attributes.parquet')
    x = F6.candidate_features(fr.with_columns(d['relevance']), attrs, t1, dec)
    y = x['relevance'].to_numpy().astype(float)
    c = evaluate(d, {'incumbent': fr['score'].to_numpy(), 'resid_B0': F6.to_scores(fr, x, F6.cv_margin(x, y))})['comparisons']['resid_B0']
    replay = json.loads(json.dumps({k: c[k] for k in ('mean_delta', 'ci95', 'by_family', 'halves', 'pairs_better', 'pairs_worse', 'gate_pass')},
                                   default=lambda o: o.item()))
    replay.update(top20_rows=x.height, listed_rows=int(y.sum()), R1_rows=int(x['R1'].sum()), T1_rows=int(t1.sum()),
                  within_tolerance=bool(abs(float(c['mean_delta']) - AUDIT_POINT) <= AUDIT_TOL))
    return x, y, replay


def main():
    start = time.perf_counter()
    assert not (OUT / 'predictions-frozen.json').exists(), 'Preserve a completed freeze'
    assert sha(PREREG) == PREREG_SHA
    reg = json.loads(PREREG.read_text(encoding='utf-8'))
    for k, v in reg['inputs'].items():
        assert sha(ROOT / v['path']) == v['sha256'], k
    protected, snap0 = check_protected(), {str(p): sha(p) for p in CODEX_READ}
    frozen98 = json.loads((S98 / 'predictions-frozen.json').read_text(encoding='utf-8'))
    frozen104 = json.loads((S104 / 'predictions-frozen.json').read_text(encoding='utf-8'))
    players = {r[0]: (r[1], r[2]) for r in pl.read_csv(ROOT / 'data/development_labels.csv', columns=['pair_id', 'player_1', 'player_2']).iter_rows()}
    x, y, replay = audit(players, frozen104)
    print('audit replay', json.dumps(replay), round(time.perf_counter() - start, 1), flush=True)
    if not replay['within_tolerance']:
        write(OUT / 'stopped-audit-replay.json', replay)
        raise SystemExit('audit replay outside tolerance; no held build')
    models = {'gameplay': F6.fit(x, y), 'structure': F6.fit(structure_only(x), y)}
    write(OUT / 'models.json', models)
    (OUT / 'scores').mkdir(exist_ok=True)
    receipts = []
    for fold in (0, 1, 2):
        base = S98 / 'scores' / f'fold{fold}-B0.parquet'
        ta, tb = S104 / f'fold{fold}-type-attributes.parquet', S104 / 'scores' / f'fold{fold}-typed_B0.parquet'
        assert sha(base) == frozen98['scores'][base.name] and sha(ta) == frozen104['attributes'][ta.name] and sha(tb) == frozen104['scores'][tb.name]
        fb = chronology(pl.read_parquet(base).sort(KEYS))
        t1 = align(fb, pl.read_parquet(ta))
        dec = decode(fb, t1)
        typed = pl.read_parquet(tb).sort(KEYS)
        assert typed.select(KEYS).equals(fb.select(KEYS))
        dec_diff = float(np.abs(typed['score'].to_numpy() - dec).max())
        assert dec_diff <= 1e-12, dec_diff
        attrs = A6.build(fb.filter(pl.col('score') > 5).select(KEYS), players)
        attrs.write_parquet(OUT / f'fold{fold}-reranker-attributes.parquet')
        for name, p in arms(fb, attrs, t1, dec, models).items():
            assert np.isfinite(p).all() and np.array_equal(p > 5, fb['score'].to_numpy() > 5)
            fb.select('pair_id', 'hand_id', 'table_id').with_columns(pl.Series('score', p)).write_parquet(OUT / 'scores' / f'fold{fold}-{name}.parquet')
        receipts.append({'fold': fold, 'rows': fb.height, 'top20_rows': int((fb['score'] > 5).sum()), 'pairs': fb['pair_id'].n_unique(),
                         'T1_rows': int(t1.sum()), 'decoder_max_abs_diff_vs_step104': dec_diff})
        print('fold', fold, receipts[-1], round(time.perf_counter() - start, 1), flush=True)
    frozen = {'created_at': datetime.now(timezone.utc).isoformat(), 'registration_sha256': PREREG_SHA, 'audit_replay': replay,
              'models_sha256': sha(OUT / 'models.json'),
              'base_scores_step98': {f'fold{f}-{a}.parquet': frozen98['scores'][f'fold{f}-{a}.parquet'] for f in (0, 1, 2) for a in ('B0', 'incumbent', 'stage1_only')},
              'decoder_scores_step104': {f'fold{f}-typed_B0.parquet': frozen104['scores'][f'fold{f}-typed_B0.parquet'] for f in (0, 1, 2)},
              'scores': {p.name: sha(p) for p in sorted((OUT / 'scores').iterdir())},
              'attributes': {p.name: sha(p) for p in sorted(OUT.glob('*-reranker-attributes.parquet'))}, 'fold_receipts': receipts,
              'script_sha256': sha(Path(__file__)), 'modules_sha256': {p.name: sha(p) for p in MODULES},
              'protected_unchanged': check_protected() == protected, 'codex_files_unchanged': snap0 == {str(p): sha(p) for p in CODEX_READ},
              'seconds': time.perf_counter() - start, 'held_evidence_evaluation': False}
    assert frozen['protected_unchanged'] and frozen['codex_files_unchanged']
    write(OUT / 'predictions-frozen.json', frozen)
    print('PREDICTIONS FROZEN', round(frozen['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
