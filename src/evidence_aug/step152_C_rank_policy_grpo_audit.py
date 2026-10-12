"""Step 152 (Claude): GRPO-style ranking policy over the top-20 evidence candidates, a correction over the production incumbent
decode(v1) with T1 off for coordinated isolation (the lists of the public .90173/.90321 files). Training-only 162-query audit,
cross-fitted by table (GroupKFold 5); 10 seeds per fold and arm, scores averaged. No held folds, no evaluation pairs.
Registration: experiments/rank_policy_grpo_C_20260914/registration152.json (read before any policy output). Never uploads."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
POKER = Path('C:/Data/Projects/Money Making/work/poker')
sys.path.insert(0, str(POKER / 'Claude Version/src')); sys.path.insert(0, str(POKER / 'Codex Resume/src'))
import numpy as np, polars as pl
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold

CV = POKER / 'Claude Version'
EXP = CV / 'experiments/rank_policy_grpo_C_20260914'
REG = EXP / 'registration152.json'
K = ['pair_id', 'hand_id']
FAMS = ['directed_transfer', 'soft_play', 'coordinated_isolation']
ARMS = ['G1', 'G0', 'L']
SEEDS = [15200 + i for i in range(10)]
FEATURES = ['logit_p', 'logit_pq', 'q', 'typed', 'time_position', 'p_rank', 'prior_mass', 'ss_minus_v1', 'fam_DT', 'fam_SP', 'fam_CI',
            'list_mass', 'list_typed']


def lgt(x, eps):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def load():
    import audit_common as ac
    from evidence_aug.paths import sha
    from evidence_aug.step104_C_typed_decoder_build import chronology, decode, align, less_than_five
    reg = json.loads(REG.read_text(encoding='utf-8'))
    for name, v in reg['inputs'].items():
        assert sha(POKER / v['path']) == v['sha256'], name
    d = pl.read_parquet(ac.AUDIT / 'audit-oof.parquet', columns=K + ['table_id', 'behavior_family', 'relevance', 'chrono_rank']).sort(K)
    v = pl.read_parquet(CV / 'experiments/view_averaging_incumbent_C_20260913/audit_v1_scores.parquet').sort(K)
    assert v.select(K).equals(d.select(K))
    fr = chronology(d.select(K + ['table_id']).with_columns(pl.Series('score', v['S_star'].to_numpy())))
    assert fr.select(K).equals(d.select(K))
    t1 = align(fr, pl.read_parquet(CV / 'experiments/typed_listing_decoder_C_20260913/audit-type-attributes.parquet'))
    V1, SS = v['v1'].to_numpy(), v['S_star'].to_numpy()
    assert np.abs(decode(fr.with_columns(pl.Series('score', V1)), t1) - v['dec_v1'].to_numpy()).max() == 0
    fam = d['behavior_family'].to_numpy()
    typed = t1 & (fam != 'coordinated_isolation')
    inc = decode(fr.with_columns(pl.Series('score', V1)), typed)
    top = np.flatnonzero(V1 > 5)
    pid = d['pair_id'].to_numpy()[top]
    _, first = np.unique(pid, return_index=True)
    groups = np.split(np.arange(len(top)), first[1:])
    assert len(groups) == 162 and all(len(g) == 20 for g in groups)
    rel_all = d['relevance'].to_numpy()
    nlisted = dict(d.group_by('pair_id').agg(pl.col('relevance').sum()).iter_rows())
    cr = fr['chrono_rank'].to_numpy()[top]
    Q = len(groups)
    X, base, rel, n_norm, rows, pairs, tables, famq = np.zeros((Q, 20, len(FEATURES))), np.zeros((Q, 20)), np.zeros((Q, 20)), np.zeros(Q), [], [], [], []
    tab = d['table_id'].to_numpy()[top]
    for qi, g in enumerate(groups):
        ix = top[g]
        p, ps, t = V1[ix] - 10, SS[ix] - 10, typed[ix]
        order = np.lexsort((cr[g], ~t))
        q = np.empty(20); q[order] = less_than_five(p[order])
        prior = np.empty(20); prior[order] = np.r_[0, np.cumsum(p[order])[:-1]]
        tpos = np.empty(20); tpos[np.argsort(cr[g], kind='stable')] = np.arange(20) / 19
        prank = np.empty(20); prank[np.argsort(-p, kind='stable')] = np.arange(20) / 19
        f = fam[ix[0]]
        X[qi] = np.column_stack([lgt(p, 1e-6), lgt(p * q, 1e-6), q, t, tpos, prank, prior / 5, lgt(ps, 1e-6) - lgt(p, 1e-6),
                                 np.full(20, f == FAMS[0]), np.full(20, f == FAMS[1]), np.full(20, f == FAMS[2]),
                                 np.full(20, p.sum() / 5), np.full(20, t.sum() / 20)])
        base[qi] = lgt(inc[ix] - 10, 1e-15)
        rel[qi] = rel_all[ix]
        n_norm[qi] = min(5, nlisted[pid[g[0]]])
        rows.append(ix); pairs.append(pid[g[0]]); tables.append(tab[g[0]]); famq.append(f)
    return dict(ac=ac, d=d, inc=inc, X=X, base=base, rel=rel, n_norm=n_norm, rows=np.array(rows), pairs=pairs, tables=np.array(tables),
                fam=np.array(famq), reg=reg, sha=sha)


def fit_one(job):
    from evidence_aug import step152_rank_policy as RP
    arm, fold, seed, Xtr, btr, rtr, ntr, Xte, bte = job
    t0 = time.perf_counter()
    pol, trace = RP.train(arm, btr, Xtr, rtr, ntr, seed)
    return arm, fold, seed, pol.forward(bte, Xte), pol.forward(btr, Xtr), trace, float(pol.p['w0']), round(time.perf_counter() - t0, 2)


def main():
    t0 = time.perf_counter()
    from evidence_aug import step152_rank_policy as RP
    D = load()
    ac, d, sha, reg = D['ac'], D['d'], D['sha'], D['reg']
    assert sha(CV / 'src/evidence_aug/step152_rank_policy.py') == reg['code']['module_sha256']
    assert sha(Path(__file__)) == reg['code']['runner_sha256']
    assert json.loads(json.dumps(RP.CONFIG)) == reg['method']['config'] and RP.LAMBDA == reg['method']['reward_lambdas']
    assert not (EXP / 'results152.json').exists(), 'preserve a completed read'
    from evidence_aug.paths import check_protected
    prot = check_protected()
    X, base, rel, n_norm, tables = D['X'], D['base'], D['rel'], D['n_norm'], D['tables']
    Q = len(base)
    greedy0 = RP.greedy_ap(base, rel, n_norm)
    inc_map = ac.evaluate(d, {'incumbent': D['inc']})['sets']['incumbent']['map5']
    replay = {'incumbent_map5_ac': inc_map, 'base_greedy_ap_mean': float(greedy0.mean()),
              'ties_in_base': int(sum(len(np.unique(b)) < 20 for b in base)),
              'oracle_top20_ap': float(RP.greedy_ap(rel + 1e-3 * base / 100, rel, n_norm).mean())}
    print('replay', json.dumps(replay), flush=True)
    assert abs(replay['incumbent_map5_ac'] - .7050668724279835) < 1e-12 and abs(replay['base_greedy_ap_mean'] - inc_map) < 1e-9
    if os.environ.get('STEP152_DRY') == '1':
        print('dry run: data and replay only, no fits'); return
    folds = list(GroupKFold(n_splits=5).split(np.zeros(Q), groups=tables))
    jobs = []
    for fi, (tr, te) in enumerate(folds):
        mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-9
        Xs = (X - mu) / sd
        for arm in ARMS:
            for seed in SEEDS:
                jobs.append((arm, fi, seed, Xs[tr], base[tr], rel[tr], n_norm[tr], Xs[te], base[te]))
    with Pool(14) as pool:
        out = pool.map(fit_one, jobs, chunksize=1)
    S = {a: np.zeros((len(SEEDS), Q, 20)) for a in ARMS}
    seed_ap, train_gain, traces, w0s, secs = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {a: [] for a in ARMS}, {a: [] for a in ARMS}, 0.
    for arm, fi, seed, ste, str_, trace, w0, s in out:
        tr, te = folds[fi]
        S[arm][SEEDS.index(seed)][te] = ste
        train_gain[arm].append(float(RP.greedy_ap(str_, rel[tr], n_norm[tr]).mean() - greedy0[tr].mean()))
        traces[arm].append(trace); w0s[arm].append(w0); secs += s
    scores, res = {'incumbent': D['inc']}, {}
    for arm in ARMS:
        ens = S[arm].mean(0)
        full = D['inc'].copy()
        full[D['rows'].reshape(-1)] = 10 + ens.reshape(-1) / 1000
        scores[arm] = full
        per_seed = [float(RP.greedy_ap(S[arm][k], rel, n_norm).mean()) for k in range(len(SEEDS))]
        top_new = np.argsort(-ens, axis=1, kind='stable')[:, :5]
        top_old = np.argsort(-base, axis=1, kind='stable')[:, :5]
        same = np.array([set(a) == set(b) for a, b in zip(top_new, top_old)])
        tr0 = np.array([t[0][1] for t in traces[arm]]); tr1 = np.array([t[-1][1] for t in traces[arm]]); kl1 = np.array([t[-1][2] for t in traces[arm]])
        res[arm] = {'ensemble_greedy_ap': float(RP.greedy_ap(ens, rel, n_norm).mean()), 'seed_greedy_ap_mean': float(np.mean(per_seed)),
                    'seed_greedy_ap_range': [min(per_seed), max(per_seed)], 'copy_share_top5_set': float(same.mean()),
                    'mean_top5_set_change': float(np.mean([len(set(a) - set(b)) for a, b in zip(top_new, top_old)])),
                    'train_objective_first_last_mean': [float(tr0.mean()), float(tr1.mean())], 'final_kl_first_position_mean': float(kl1.mean()),
                    'train_in_sample_greedy_gain_mean': float(np.mean(train_gain[arm])), 'w0_mean': float(np.mean(w0s[arm]))}
        print(arm, json.dumps(res[arm]), flush=True)
    R = lambda x: round(float(x), 6)

    def cmp(a, b):
        c = ac.evaluate(d, {'incumbent': scores[a], b: scores[b]})['comparisons'][b]
        return {'mean': R(c['mean_delta']), 'ci95': [R(x) for x in c['ci95']], 'better_worse': [c['pairs_better'], c['pairs_worse']],
                'fam': {k[:2]: R(x) for k, x in c['by_family'].items()}, 'halves': {k: R(x) for k, x in c['halves'].items()}, 'ac_gate': bool(c['gate_pass'])}
    comp = {f'{b}_vs_incumbent': cmp('incumbent', b) for b in ARMS}
    comp['G1_vs_L'] = cmp('L', 'G1'); comp['G1_vs_G0'] = cmp('G0', 'G1'); comp['G0_vs_L'] = cmp('L', 'G0')
    p = comp['G1_vs_incumbent']
    small = p['mean'] >= .006 and p['ci95'][0] > -.004 and min(p['fam'].values()) >= -.01 and min(p['halves'].values()) >= 0
    verdict = 'PASS' if p['ac_gate'] else ('SMALL' if small else 'FAIL')
    for k, v in comp.items():
        print(k, json.dumps(v), flush=True)
    out_scores = d.select(K).with_columns(*[pl.Series(a, scores[a]) for a in ['incumbent'] + ARMS])
    out_scores.write_parquet(EXP / 'audit_scores152.parquet')
    assert check_protected() == prot
    results = {'registration_sha256': sha(REG), 'replay': replay, 'arms': res, 'comparisons': comp, 'verdict_primary_G1': verdict,
               'other_arms_passing_ac_gate': [a for a in ('G0', 'L') if comp[f'{a}_vs_incumbent']['ac_gate']],
               'scores_sha256': sha(EXP / 'audit_scores152.parquet'), 'fits': len(out), 'fit_cpu_seconds': round(secs, 1),
               'wall_seconds': round(time.perf_counter() - t0, 1), 'held_reads': 9}
    (EXP / 'results152.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('VERDICT', verdict, 'wall', results['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
