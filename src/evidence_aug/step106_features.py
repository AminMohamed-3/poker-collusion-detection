"""Step 106 (Claude): candidate features over top-20 evidence rows and the ridge residual GLM boosted from the step-104 decoder margin."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import polars as pl

NM = ['preflop', 'flop', 'turn', 'river']
FG = ['R1', 'R1_before', 'R1_n', 'T1_before', 'T1_n', 'R2dt', 'R2dt_before', 'R2sp', 'R2sp_before', 'L_Sfold', 'W_Sfold', 'L_fold_eq', 'W_fold_eq',
      'L_Smax', 'W_Smax', 'both_sd', 'log_pot', 'out_fold_to_pair', 'fold_partner_after_outfold', 'pair_reraise', 'sandwich', 'tpos', 'trank', 'prank',
      'hi_before', 'p']
LAMBDA = 10.


def logit(v):
    v = np.clip(v, 1e-4, 1 - 1e-4)
    return np.log(v / (1 - v))


def candidate_features(frame, attrs, t1, dec):
    """frame: pair_id, hand_id, table_id, score (top-20 rows have score > 5), chrono_rank (0-based within pair over the frame), sorted by keys;
    t1 and dec are aligned to frame rows. Returns top-20 rows sorted by pair and time, with the FG columns and the decoder probability dec."""
    f = frame.with_columns(pl.Series('T1', np.asarray(t1, float)), pl.Series('dec', np.asarray(dec, float) - 10), (pl.col('score') - 10).alias('p'),
                           pl.len().over('pair_id').alias('n_shared')).filter(pl.col('score') > 5)
    x = f.join(attrs, on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left')
    sf = lambda r: pl.coalesce([pl.when(pl.col(f'{r}_fold_s') == i).then(pl.col(f'{r}_S_{n}')) for i, n in enumerate(NM)])
    r1 = lambda r: (pl.col(f'{r}_fold_facing') == 'partner') & (pl.col(f'{r}_fold_eq').fill_null(0) > .5) & (sf(r).fill_null(0) > .6)
    x = x.with_columns(sf('W').alias('W_Sfold'), sf('L').alias('L_Sfold'), (r1('W') | r1('L')).cast(pl.Int32).alias('R1'),
                       ((pl.col('L_weak_call_partner') > 0) & (pl.col('pot_bb') > 40)).cast(pl.Int32).alias('R2dt'),
                       pl.max_horizontal([pl.col(f'W_S_{n}') for n in NM]).alias('W_Smax'), pl.max_horizontal([pl.col(f'L_S_{n}') for n in NM]).alias('L_Smax'),
                       (pl.col('chrono_rank') / pl.col('n_shared')).alias('tpos'), pl.col('pot_bb').log1p().alias('log_pot'))
    x = x.with_columns(((pl.col('both_sd') == 1) & (pl.col('W_Smax').fill_null(0) > .7)).cast(pl.Int32).alias('R2sp'), (pl.col('p') > .5).cast(pl.Int32).alias('hi'))
    x = x.sort('pair_id', 'chrono_rank').with_columns(
        pl.col('p').rank('ordinal', descending=True).over('pair_id').alias('prank'), pl.col('chrono_rank').rank('ordinal').over('pair_id').alias('trank'),
        *[(pl.col(c).cum_sum().over('pair_id') - pl.col(c)).alias(f'{c}_before') for c in ['R1', 'T1', 'R2dt', 'R2sp', 'hi']],
        *[pl.col(c).sum().over('pair_id').alias(f'{c}_n') for c in ['R1', 'T1']])
    assert x.height == f.height and x['R1'].null_count() == 0
    return x


def matrix(x):
    return x.select(FG).fill_null(0).to_numpy().astype(float)


def glm_fit(M, y, off, lam=LAMBDA):
    A = np.hstack([np.ones((len(M), 1)), M])
    b, P = np.zeros(A.shape[1]), lam * np.eye(A.shape[1])
    P[0, 0] = 0
    for _ in range(60):
        mu = 1 / (1 + np.exp(-(off + A @ b)))
        step = np.linalg.solve((A * (mu * (1 - mu))[:, None]).T @ A + P, A.T @ (y - mu) - P @ b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def fit(x, y, lam=LAMBDA):
    M = matrix(x)
    mu, sd = M.mean(0), M.std(0) + 1e-9
    b = glm_fit((M - mu) / sd, y, logit(x['dec'].to_numpy()), lam)
    return {'features': FG, 'lambda': lam, 'mean': mu.tolist(), 'scale': sd.tolist(), 'coef': b.tolist()}


def margin(model, x):
    assert model['features'] == FG
    M = (matrix(x) - np.array(model['mean'])) / np.array(model['scale'])
    return logit(x['dec'].to_numpy()) + np.hstack([np.ones((len(M), 1)), M]) @ np.array(model['coef'])


def cv_margin(x, y, lam=LAMBDA, seeds=(0, 1, 2, 3, 4)):
    """Out-of-fold margins: 5 folds grouped by the pair's first table, averaged over fold seeds."""
    tab = x.select('pair_id').join(x.group_by('pair_id').agg(pl.col('table_id').min().alias('t')), on='pair_id', how='left', maintain_order='left')['t'].to_numpy()
    ug, M, off, oof = np.unique(tab), matrix(x), logit(x['dec'].to_numpy()), np.zeros(len(y))
    for sd in seeds:
        perm = np.random.RandomState(sd).permutation(len(ug))
        fmap = dict(zip(ug[perm], np.arange(len(ug)) % 5))
        fo = np.array([fmap[t] for t in tab])
        for k in range(5):
            tr, te = fo != k, fo == k
            mu, sdv = M[tr].mean(0), M[tr].std(0) + 1e-9
            b = glm_fit((M[tr] - mu) / sdv, y[tr], off[tr], lam)
            oof[te] += (off[te] + np.hstack([np.ones((te.sum(), 1)), (M[te] - mu) / sdv]) @ b) / len(seeds)
    return oof


def to_scores(frame, x, m):
    s = x.select('pair_id', 'hand_id').with_columns(pl.Series('new', 10 + 1 / (1 + np.exp(-m))))
    j = frame.select('pair_id', 'hand_id').join(s, on=['pair_id', 'hand_id'], how='left', validate='1:1', maintain_order='left')['new'].fill_null(np.nan).to_numpy()
    return np.where(np.isnan(j), frame['score'].to_numpy(), j)


def shuffle_within_pair(attrs, seed):
    """Negative control: reassign attribute rows to other top-20 hands of the same pair."""
    a = attrs.sort('pair_id', 'hand_id')
    pid, hid, rng = a['pair_id'].to_numpy(), a['hand_id'].to_numpy().copy(), np.random.default_rng(seed)
    starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    for g in np.split(np.arange(len(pid)), starts[1:]):
        hid[g] = rng.permutation(hid[g])
    return a.with_columns(pl.Series('hand_id', hid))
