"""STEP253 -- partner-specific evidence first inside the lower block of the STEP207 risk order (risk column only).

Base: outputs_step207_20260917/submission_step207_ens4_ci_dt_rolestrength_on_step169.csv (sha 2a475c7a..., public .91152).
Lower block = STEP207 risk in [.5, .99): band b = [.9, .99) (187 pairs), band cd = [.5, .9) (345 pairs).
PS flag (label-free, evaluation phase only, partner-specific = every baseline is leave-this-partner-out):
    PS = (STEP248 instrument fires >= 2 event cells) or (its minimum cell p < 1e-5)
         or (STEP250 leave-partner-out Poisson excess of sharp known-family hands p < 1e-3)
Two registered candidates, both a PERMUTATION of the existing risk strings among lower-block rows (multiset preserved,
every other byte of the file identical):
    B   : inside band b, PS pairs first (stable: STEP207 order inside each class)
    BCD : inside the whole lower block: PS of b, PS of cd, rest of b, rest of cd (stable)
Not a learned model.  Ensemble form: two label-free instruments (a 126-event partner-specific min-p instrument; three
per-family supervised event classifiers summarised by a leave-partner-out Poisson excess), combined by OR.
prediction253.json is written BEFORE either file.  Never uploaded by Claude.
"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(r'C:\Data\Projects\Money Making\work\poker')
CV = ROOT / 'Claude Version'
sys.path.insert(0, str(CV / 'src'))
OUT = CV / 'outputs_step253_20260917'
OUT.mkdir(exist_ok=True)
BASE = CV / 'outputs_step207_20260917' / 'submission_step207_ens4_ci_dt_rolestrength_on_step169.csv'
BASE_SHA = '2a475c7a168ee178fc6232cf02c96c2ce92b7084acc14a39d6aa315f1f2038f9'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE) == BASE_SHA

raw = BASE.read_bytes().decode('utf-8')
nl = '\r\n' if '\r\n' in raw else '\n'
lines = raw.split(nl)
assert lines[-1] == '' and lines[0].split(',')[:3] == ['pair_id', 'risk_score', 'predicted_behavior']
rows = [l.split(',') for l in lines[1:-1]]
assert all(len(r) == 8 for r in rows) and len(rows) == 112540
pid = [r[0] for r in rows]; rs = [r[1] for r in rows]; rv = np.array([float(v) for v in rs])
inst = pl.read_parquet(CV / 'experiments' / 'step248_hidden_family_eval_20260917' / 'step248b-eval-pair-scores.parquet').select('pair_id', 'n_cells', 'min_pval')
exc = pl.read_parquet(CV / 'experiments' / 'step250_event_count_tail_20260917' / 'eval_pair_excess.parquet').select('pair_id', 'p_excess', pl.col('risk_score').alias('risk_in_250'))
d = pl.DataFrame({'pair_id': pid, 'risk': rv, 'pos': np.arange(len(pid))}).join(inst, on='pair_id', how='left').join(exc, on='pair_id', how='left').sort('pos')
assert d.height == len(pid) and d['p_excess'].null_count() == 0 and float((d['risk'] - d['risk_in_250']).abs().max()) < 1e-12, 'instruments were not computed on the STEP207 order'
d = d.with_columns(pl.col('n_cells').fill_null(0), pl.col('min_pval').fill_null(1.0))
d = d.with_columns(ps=((pl.col('n_cells') >= 2) | (pl.col('min_pval') < 1e-5) | (pl.col('p_excess') < 1e-3)))
ps = d['ps'].to_numpy()
in_b = (rv >= .9) & (rv < .99); in_cd = (rv >= .5) & (rv < .9)
assert int(in_b.sum()) == 187 and int(in_cd.sum()) == 345 and int((rv >= .99).sum()) == 377
counts = {'band_b': int(in_b.sum()), 'band_b_PS': int((in_b & ps).sum()), 'band_cd': int(in_cd.sum()), 'band_cd_PS': int((in_cd & ps).sum()),
          'top_band_PS_share': round(float(ps[rv >= .99].mean()), 4), 'near_tail_910_7630_PS_share': round(float(ps[np.argsort(-rv)[909:7630]].mean()), 4)}

PRED = OUT / 'prediction253.json'
if not PRED.exists():
    json.dump({
        'step': 'STEP253', 'written_before': 'either submission file', 'at': time.strftime('%Y-%m-%d %H:%M:%S local (Egypt, UTC+3)'),
        'base': {'file': str(BASE.relative_to(CV)), 'sha256': BASE_SHA, 'public': 0.91152},
        'rule': __doc__, 'class_counts': counts,
        'why': ('PS share by STEP207 band: >= .99 .923; [.9,.99) .348 (thirds .56 / .35 / .14); [.5,.9) .078; ranks 910-7630 .055; deep tail .022.  A two-component mixture puts ~65-100 real pairs and '
                '~90-120 false ones in band b, the false ones mostly coordinated_isolation-labelled with raw sharp hands but no partner-specific excess (spill-over), which is also what B = .985 '
                '(CI AP ~ .955) implies.  OBS 19.200.'),
        'scenario_simulation': {'valid_high_fire_rate': {'B': '+.0064', 'BCD': '+.0119'}, 'valid_low_fire_rate': {'B': '+.0041', 'BCD': '+.0085'}, 'weakly_valid': {'B': '+.0021', 'BCD': '+.0023'},
                                'null': {'B': '+.0000 to -.0009', 'BCD': '-.0027 to -.0036'}, 'note': 'scenario P/B levels are below the measured .9652/.985, so valid-scenario gains are upper-ish'},
        'prediction_composite_delta_vs_0.91152': {'B': {'point': 0.0025, 'band80': [-0.0010, 0.0060]}, 'BCD': {'point': 0.0035, 'band80': [-0.0040, 0.0110]}, 'read_SE': 0.00065,
                                                  'P_valid_my_estimate': 0.6},
        'decision_rules_fixed_now': [
            'B > +.0013: partner-specific evidence separates real from false pairs inside band b; keep the order and build a graded version next.',
            'BCD - B > +.0013: the PS pairs of [.5,.9) are real too, adopt BCD; BCD - B < -.0013: they are chance fires, keep B only.',
            'both <= -.0013: the lever is closed; no re-definition of PS, no threshold search on these reads.',
            'Third upload of the night = the better of {B, BCD, STEP207 order} by these two reads, carrying the STEP251 evidence cells (risk and evidence are additive in the metric).'],
        'uploads': 'never by Claude; Amin decides.'}, open(PRED, 'w', encoding='utf-8'), indent=1)
    print('prediction written', PRED)

def build(tag, order_fn):
    idx = np.flatnonzero(in_b | in_cd)
    idx = idx[np.argsort(-rv[idx], kind='stable')]                    # lower block in STEP207 order (no ties: asserted below)
    assert len(set(rs[i] for i in idx)) == len(idx)
    new_order = order_fn(idx)                                           # the pairs, best first
    assert sorted(new_order.tolist()) == sorted(idx.tolist())
    new_rs = list(rs)
    for slot, pair in zip(idx, new_order):                              # slot = the row that owned the k-th largest value; give that value to the k-th pair
        new_rs[pair] = rs[slot]
    assert sorted(new_rs[i] for i in idx) == sorted(rs[i] for i in idx)
    out_lines = [lines[0]] + [','.join([r[0], new_rs[k]] + r[2:]) for k, r in enumerate(rows)] + ['']
    path = OUT / f'submission_step253_psfirst_{tag}_on_step207.csv'
    assert PRED.exists()
    path.write_bytes(nl.join(out_lines).encode('utf-8'))
    changed = [k for k in range(len(rows)) if new_rs[k] != rs[k]]
    return path, changed

stable = lambda idx, key: idx[np.argsort(key, kind='stable')]
fB = lambda idx: np.concatenate([stable(idx[in_b[idx]], ~ps[idx[in_b[idx]]]), idx[in_cd[idx]]])
fBCD = lambda idx: stable(idx, np.where(ps[idx] & in_b[idx], 0, np.where(ps[idx], 1, np.where(in_b[idx], 2, 3))))
receipt = {'base_sha256': BASE_SHA, 'class_counts': counts, 'files': {}}
from validation import validate_submission
from evidence_aug.paths import check_protected
for tag, fn in (('b', fB), ('bcd', fBCD)):
    path, changed = build(tag, fn)
    chk = pl.read_csv(path, schema_overrides={'risk_score': pl.Utf8}); base = pl.read_csv(BASE, schema_overrides={'risk_score': pl.Utf8})
    other = [c for c in base.columns if c != 'risk_score']
    assert chk.select(other).equals(base.select(other)), 'a non-risk column changed'
    diff = (chk['risk_score'] != base['risk_score']).to_numpy()
    assert diff.sum() == len(changed) and bool(((in_b | in_cd) | ~diff).all()), 'a row outside the lower block changed'
    rep = validate_submission(path, ROOT / 'data' / 'evaluation_pairs.csv', ROOT / 'data' / 'hands.parquet', ROOT / 'data' / 'seats.parquet', ROOT / 'data' / 'sample_submission.csv')
    errs = getattr(rep, 'errors', None)
    newv = chk['risk_score'].cast(pl.Float64).to_numpy()
    top = np.argsort(-newv, kind='stable')
    receipt['files'][tag] = {'path': str(path.relative_to(CV)), 'sha256': sha(path), 'rows_changed': int(diff.sum()), 'rows_changed_in_band_b': int((diff & in_b).sum()), 'rows_changed_in_band_cd': int((diff & in_cd).sum()),
                             'validation_errors': [str(e) for e in errs] if errs is not None else str(rep), 'PS_share_of_first_65_lower_block_slots': round(float(ps[top[377:442]].mean()), 3),
                             'spearman_like_top909_membership_identical': bool(set(top[:909].tolist()) == set(np.argsort(-rv, kind='stable')[:909].tolist()))}
    print(tag, json.dumps(receipt['files'][tag]))
receipt['protected'] = {k: v[:8] for k, v in check_protected().items()}
json.dump(receipt, open(OUT / 'receipt253.json', 'w', encoding='utf-8'), indent=1)
print('protected unchanged:', len(receipt['protected']), 'files')
