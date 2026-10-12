"""STEP258 step 2: MECHPLUS per-hand features for the certified prefix-60 block testbed.

Builds, for a chosen hand universe, the 850-column MECHPLUS per-hand matrix in the exact
step254_eval.py order (163 production + 25 STEP237 + 662 STEP254 mechanical):

  mode 'prefix'   the 779 testbed pairs x their PREFIX-60 hands (30,731 rows).
                  163: experiments/viewdis_20260911/prefix_fold{1,2}_raw.parquet, fold-matched to the
                       testbed row (the frames step61 built from exactly these hands).
                  25 : evidence_aug/step237_features.pairhand_features, recomputed on the prefix keys.
                  662: the evidence_aug/step254_features.py code path on the prefix keys, with the
                       step254_prep.py chronology block rebuilt FROM THE PREFIX HANDS ONLY (the A/B
                       role and every backward-looking t4_* running quantity see prefix hands alone).

  mode 'full149'  the 149 LABELLED testbed pairs x their FULL development history, built by the same
                  code, for the parity gate: every column must match the production caches
                  (outputs_v2/hands_development, step237_features.parquet, step254_features.parquet).

Development phase asserted on every hand.  New files only; nothing existing is touched.
Usage:  python step258_prefix_features.py {prefix|full149}
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '4'
import sys, json, time
sys.dont_write_bytecode = True
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
DATA = ROOT + 'data/'
C = ROOT + 'Claude Version/'
sys.path.insert(0, C + 'src')
sys.path.insert(0, C + 'src/evidence_aug')
import numpy as np, polars as pl
import step254_features as F                       # noqa: E402  (module-level layout + helpers)
import step254_eval7 as E                          # noqa: E402
from step237_features import hand_tables, pairhand_features, FEATURES as F237  # noqa: E402
from pu_pairs.build_allpairs_fold import fold_pools  # noqa: E402

OUT = C + 'experiments/step258_mech_block_view_20260918/'
S254 = C + 'experiments/step254_mechanical_tensor_20260917/'
VD = C + 'experiments/viewdis_20260911/'
PREFIX = .6


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ hand universes
def prefix_hands(tables):
    """step40_prefix_audit / step61 rule: first 60% of each table's development hands by started_at."""
    hd = pl.read_parquet(DATA + 'hands.parquet').filter(
        (pl.col('phase') == 'development') & pl.col('table_id').is_in(tables))
    hd = hd.sort('table_id', 'started_at').with_columns(
        (pl.int_range(pl.len()).over('table_id') / pl.len().over('table_id')).alias('_pos'))
    return hd.filter(pl.col('_pos') < PREFIX).drop('_pos')


def universe(mode):
    """returns (keys, hd, prod163) -- keys: pair_id/hand_id/table_id/player_1/player_2;
    hd: the table hand history the chronology may see; prod163: the 163-column per-hand frame."""
    tb = pl.read_parquet(OUT + 'testbed32.parquet').select('pair_id', 'fold', 'table_id', 'shared_hands', 'label')
    members = pl.concat([pl.read_csv(C + f'experiments/pu_pairs_20260911/fold{f}/pairs.csv')
                         .select('pair_id', 'player_1', 'player_2') for f in (1, 2)]).unique(subset=['pair_id'])
    if mode == 'prefix':
        raws = []
        for f in (1, 2):
            ids = tb.filter(pl.col('fold') == f).select('pair_id')
            raws.append(pl.read_parquet(VD + f'prefix_fold{f}_raw.parquet').join(ids, on='pair_id', how='semi'))
        prod = pl.concat(raws)
        assert prod['pair_id'].n_unique() == tb.height, (prod['pair_id'].n_unique(), tb.height)
        # the chronology must see the pair's whole PREFIX table history (as production sees the whole
        # development table history), not only the hands in which the pair is co-dealt: the session /
        # rebuy / stack-change running quantities are per player over the table's hand sequence.
        hd = prefix_hands(sorted(set(fold_pools(1)) | set(fold_pools(2))))
        hd = hd.filter(pl.col('table_id').is_in(prod['table_id'].unique().to_list()))
    else:
        lab = tb.filter(pl.col('label') >= 0).select('pair_id')
        assert lab.height == 149, lab.height
        prod = pl.read_parquet(ROOT + 'outputs_v2/hands_development/batch_*.parquet').join(
            lab, on='pair_id', how='semi')
        tabs = prod['table_id'].unique().to_list()
        hd = pl.read_parquet(DATA + 'hands.parquet').filter(
            (pl.col('phase') == 'development') & pl.col('table_id').is_in(tabs))
    keys = prod.select('pair_id', 'hand_id', 'table_id').join(members, on='pair_id', how='left')
    assert keys['player_1'].null_count() == 0
    # single-phase gate
    ph = pl.read_parquet(DATA + 'hands.parquet', columns=['hand_id', 'phase'])
    chk = keys.select('hand_id').unique().join(ph, on='hand_id', how='left')
    assert (chk['phase'] == 'development').all(), 'non-development hand in the STEP258 build'
    assert (hd['phase'] == 'development').all()
    return keys, hd.drop('phase'), prod


# ------------------------------------------------------------------ family 4 chronology (step254_prep)
def build_chrono(keys, hd):
    """step254_prep.build_chrono verbatim, parameterised by the (keys, hd) universe."""
    hd = hd.select('hand_id', 'table_id', 'started_at', 'big_blind').sort('table_id', 'started_at', 'hand_id')
    hd = hd.with_columns(pl.int_range(pl.len()).over('table_id').alias('hidx'))
    assert keys.select('hand_id').unique().join(hd.select('hand_id'), on='hand_id').height == \
        keys['hand_id'].n_unique(), 'a pair hand is outside the table history given to the chronology'
    members = pl.concat([keys.select(pl.col('player_1').alias('player_id')),
                         keys.select(pl.col('player_2').alias('player_id'))]).unique()
    s = (pl.scan_parquet(DATA + 'seats.parquet')
         .select('hand_id', 'player_id', 'starting_stack', 'net_chips', 'total_contribution')
         .join(members.lazy(), on='player_id')
         .join(hd.lazy().select('hand_id', 'table_id', 'hidx', 'big_blind'), on='hand_id')
         .collect())
    s = s.sort('player_id', 'table_id', 'hidx')
    g = ['player_id', 'table_id']
    s = s.with_columns([
        pl.col('starting_stack').shift(1).over(g).alias('_pstack'),
        pl.col('net_chips').shift(1).over(g).alias('_pnet'),
        pl.col('hidx').shift(1).over(g).alias('_phidx'),
    ])
    s = s.with_columns((pl.col('starting_stack') - (pl.col('_pstack') + pl.col('_pnet'))).alias('topup'))
    s = s.with_columns([
        (pl.col('topup').is_not_null() & (pl.col('topup') != 0)).cast(pl.Int8).alias('rebuy'),
        (pl.col('_phidx').is_null() | (pl.col('hidx') - pl.col('_phidx') != 1)
         | (pl.col('topup') < 0)).cast(pl.Int8).alias('new_sit'),
    ])
    s = s.with_columns(pl.col('new_sit').cum_sum().over(g).alias('sess'))
    g2 = ['player_id', 'table_id', 'sess']
    s = s.with_columns([
        pl.int_range(pl.len()).over(g2).alias('session_age'),
        (pl.col('net_chips').cum_sum().over(g2) - pl.col('net_chips')).alias('sess_profit_before'),
        pl.col('rebuy').shift(1).fill_null(0).rolling_sum(5, min_periods=1).over(g).alias('rebuy_prev5'),
        (pl.col('starting_stack') - pl.col('starting_stack').shift(5)).over(g).alias('stackchg5'),
    ])
    bb = pl.col('big_blind')
    s = s.select('hand_id', 'player_id',
                 pl.col('session_age').cast(pl.Float32),
                 (pl.col('sess_profit_before') / bb).cast(pl.Float32).alias('sess_profit_bb'),
                 pl.col('rebuy').cast(pl.Float32),
                 (pl.col('topup').fill_null(0) / bb).cast(pl.Float32).alias('topup_bb'),
                 pl.col('rebuy_prev5').cast(pl.Float32),
                 (pl.col('stackchg5').fill_null(0) / bb).cast(pl.Float32).alias('stackchg5_bb'),
                 (pl.col('starting_stack') / bb).cast(pl.Float32).alias('stack_bb'),
                 pl.col('net_chips'), pl.col('total_contribution'))
    COLS = ['session_age', 'sess_profit_bb', 'rebuy', 'topup_bb', 'rebuy_prev5', 'stackchg5_bb', 'stack_bb']
    k = keys.select('pair_id', 'hand_id', 'table_id', 'player_1', 'player_2').join(
        hd.select('hand_id', 'started_at', 'big_blind'), on='hand_id')
    for who, pcol in (('1', 'player_1'), ('2', 'player_2')):
        k = k.join(s.rename({c: c + '_' + who for c in COLS + ['net_chips', 'total_contribution']}),
                   left_on=['hand_id', pcol], right_on=['hand_id', 'player_id'], how='left')
    for c in COLS:
        assert k[c + '_1'].null_count() == 0 and k[c + '_2'].null_count() == 0, c
    n1, n2 = pl.col('net_chips_1'), pl.col('net_chips_2')
    k = k.with_columns(
        pl.when((n1 < 0) & (n2 > 0)).then(pl.min_horizontal(-n1, n2))
        .when((n2 < 0) & (n1 > 0)).then(-pl.min_horizontal(-n2, n1))
        .otherwise(0).cast(pl.Float64).alias('flow_1to2'))
    k = k.sort('pair_id', 'started_at', 'hand_id')
    k = k.with_columns([
        ((pl.col('flow_1to2').cum_sum().over('pair_id') - pl.col('flow_1to2'))
         / pl.col('big_blind')).alias('cumflow_before_1to2'),
        ((pl.col('flow_1to2').abs().cum_sum().over('pair_id') - pl.col('flow_1to2').abs())
         / pl.col('big_blind')).alias('cumabsflow_before'),
        pl.int_range(pl.len()).over('pair_id').alias('shared_idx'),
        (pl.col('flow_1to2') / pl.col('big_blind')).alias('flow_bb_1to2'),
    ])
    role = k.group_by('pair_id').agg(pl.col('net_chips_1').sum().alias('tot1'),
                                     pl.col('net_chips_2').sum().alias('tot2'))
    role = role.with_columns((pl.col('tot1') <= pl.col('tot2')).cast(pl.Int8).alias('p1_is_A'))
    k = k.join(role.select('pair_id', 'p1_is_A'), on='pair_id')
    A = pl.col('p1_is_A') == 1

    def ab(name):
        a = pl.when(A).then(pl.col(name + '_1')).otherwise(pl.col(name + '_2')).cast(pl.Float32)
        b = pl.when(A).then(pl.col(name + '_2')).otherwise(pl.col(name + '_1')).cast(pl.Float32)
        return a, b

    exprs = [pl.col('p1_is_A')]
    for c in COLS:
        a, b = ab(c)
        exprs += [a.alias(f't4_{c}_A'), b.alias(f't4_{c}_B'),
                  pl.max_horizontal(a, b).alias(f't4_{c}_mx'),
                  pl.min_horizontal(a, b).alias(f't4_{c}_mn'),
                  (a + b).alias(f't4_{c}_sm')]
    sgn = pl.when(A).then(1.0).otherwise(-1.0)
    exprs += [
        (pl.col('cumflow_before_1to2') * sgn).cast(pl.Float32).alias('t4_cumflow_before_A2B'),
        pl.col('cumabsflow_before').cast(pl.Float32).alias('t4_cumabsflow_before'),
        pl.col('shared_idx').cast(pl.Float32).alias('t4_shared_idx'),
        (pl.col('flow_bb_1to2') * sgn).cast(pl.Float32).alias('t4_flow_bb_A2B'),
        (pl.col('cumflow_before_1to2') * sgn / (pl.col('shared_idx') + 1)).cast(pl.Float32)
        .alias('t4_cumflow_rate_A2B'),
    ]
    out = k.select('pair_id', 'hand_id', *exprs)
    assert out.height == keys.height
    return out


# ------------------------------------------------------------------ the 662 mechanical columns
def build_mech(keys, chrono, chunk_hands=6000):
    """The chunk loop of evidence_aug/step254_features.main(), parameterised by the key set."""
    pf = np.load(S254 + 'step254_pf169.npz')
    PCT169, EQ169 = pf['pct169'], pf['eq169x169']
    k = keys.select('pair_id', 'hand_id', 'player_1', 'player_2').join(
        chrono.select('pair_id', 'hand_id', 'p1_is_A'), on=['pair_id', 'hand_id'])
    assert k.height == keys.height
    hid = k.select('hand_id').unique()
    meta_df = (pl.scan_parquet(DATA + 'hands.parquet').join(hid.lazy(), on='hand_id')
               .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot', 'players_dealt',
                       'players_at_showdown', 'button_seat').collect())
    assert (meta_df['phase'] == 'development').all(), 'non-development hand in the STEP258 mech build'
    assert meta_df.height == hid.height
    hand_list = meta_df['hand_id'].to_list()
    nchunk = max(1, int(np.ceil(len(hand_list) / chunk_hands)))
    bounds = np.linspace(0, len(hand_list), nchunk + 1).astype(int)
    parts = []
    t_start = time.time()
    for c in range(nchunk):
        t0 = time.time()
        part = pl.DataFrame({'hand_id': hand_list[bounds[c]:bounds[c + 1]]})
        m = meta_df.join(part, on='hand_id')
        seats = (pl.scan_parquet(DATA + 'seats.parquet').join(part.lazy(), on='hand_id')
                 .select('hand_id', 'player_id', 'seat_no', 'starting_stack', 'hole_card_1',
                         'hole_card_2', 'total_contribution', 'net_chips', 'folded',
                         'went_to_showdown', 'won_share').collect())
        acts = (pl.scan_parquet(DATA + 'actions.parquet').join(part.lazy(), on='hand_id')
                .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                        'amount_to', 'pot_before', 'stack_before', 'to_call').collect()
                .sort('hand_id', 'action_no'))
        hidx = {h: i for i, h in enumerate(m['hand_id'].to_list())}
        nh = m.height
        board = np.full((nh, 5), -1, np.int64)
        nboard = np.zeros(nh, np.int64)
        for i, bc in enumerate(m['board_cards'].to_list()):
            cs = (bc or '').split()
            nboard[i] = len(cs)
            for j, cc in enumerate(cs):
                board[i, j] = E.CARD_ID[cc]
        si = np.array([hidx[h] for h in seats['hand_id'].to_list()], np.int64)
        c1 = np.array([E.CARD_ID[x] for x in seats['hole_card_1'].to_list()], np.int64)
        c2 = np.array([E.CARD_ID[x] for x in seats['hole_card_2'].to_list()], np.int64)
        ns = len(si)
        sarr = np.full((ns, 4), np.nan, np.float32)
        clsarr = np.full((ns, 4), -1., np.float32)
        drw = np.zeros((ns, 4), np.float32)
        cls169 = np.where(((c1 & 3) == (c2 & 3)) | ((c1 >> 2) == (c2 >> 2)),
                          np.maximum(c1 >> 2, c2 >> 2) * 13 + np.minimum(c1 >> 2, c2 >> 2),
                          np.minimum(c1 >> 2, c2 >> 2) * 13 + np.maximum(c1 >> 2, c2 >> 2))
        sarr[:, 0] = PCT169[cls169]
        nbs = nboard[si]
        for kk, nb in ((1, 3), (2, 4), (3, 5)):
            sel = np.flatnonzero(nbs >= nb)
            if not len(sel):
                continue
            cards = np.concatenate([c1[sel, None], c2[sel, None], board[si[sel], :nb]], axis=1)
            rk = E.evaluate(cards)
            sarr[sel, kk] = (7463.0 - rk) / 7462.0
            clsarr[sel, kk] = F.made_class(rk)
            if nb in (3, 4):
                mask = np.bitwise_or.reduce(np.int64(1) << (cards >> 2), axis=1)
                sc = np.stack([((cards & 3) == su).sum(axis=1) for su in range(4)], axis=1)
                base = 0 if nb == 3 else 2
                drw[sel, base] = (sc.max(axis=1) == 4).astype(np.float32)
                drw[sel, base + 1] = (F.HAS4[mask] & ~F.HASS[mask]).astype(np.float32)
        drw2 = drw
        srows, stv, clsv, drwv, holev = {}, {}, {}, {}, {}
        for j, r in enumerate(seats.iter_rows(named=True)):
            h = r['hand_id']
            p = r['player_id']
            srows.setdefault(h, {})[p] = r
            stv.setdefault(h, {})[p] = sarr[j]
            clsv.setdefault(h, {})[p] = clsarr[j]
            drwv.setdefault(h, {})[p] = drw2[j]
            holev.setdefault(h, {})[p] = (int(c1[j]), int(c2[j]), int(cls169[j]))
        arows = {}
        for r in acts.iter_rows(named=True):
            arows.setdefault(r['hand_id'], []).append(r)
        mrows = {r['hand_id']: r for r in m.iter_rows(named=True)}
        sub = k.join(part, on='hand_id')
        n = sub.height
        Vout = np.zeros((n, F.NCOL), np.float32)
        rowsA = np.zeros((n, 2), np.int64)
        rowsB = np.zeros((n, 2), np.int64)
        rowh = np.zeros(n, np.int64)
        pids, hids = [], []
        eq_need = {1: [], 2: []}
        for i, r in enumerate(sub.iter_rows(named=True)):
            h = r['hand_id']
            mA = r['player_1'] if r['p1_is_A'] == 1 else r['player_2']
            mB = r['player_2'] if r['p1_is_A'] == 1 else r['player_1']
            pids.append(r['pair_id'])
            hids.append(h)
            mr = mrows[h]
            bb = mr['big_blind']
            seat = srows[h]
            st = stv[h]
            fin_best = max((st[q][3] if st[q][3] == st[q][3] else st[q][2] if st[q][2] == st[q][2]
                            else st[q][1] if st[q][1] == st[q][1] else st[q][0])
                           for q in seat if not seat[q]['folded'])
            stf = {q: np.where(np.isnan(st[q]), -1., st[q]) for q in seat}
            for q in seat:
                a4 = stf[q]
                for kk in range(1, 4):
                    if a4[kk] < 0:
                        a4[kk] = a4[kk - 1]
            v = np.zeros((2, F.NPM), np.float32)
            v[:, F.B_FSP:F.B_FSP + 6] = 9.
            v[:, F.B_FSA:F.B_FSA + 5] = 9.
            v[:, F.B_ARNK:F.B_ARNK + 3] = 9.
            v[:, F.B_SPRS:F.B_SPRS + 4] = -1.
            hl = np.zeros(F.NHL, np.float32)
            button = mr['button_seat']
            o_pre = {s: (s - button - 3) % 6 for s in range(6)}
            o_post = {s: (s - button - 1) % 6 for s in range(6)}
            F.process_hand(mA, mB, arows.get(h, []), seat, stf, fin_best,
                           mr['final_pot'] / bb, bb, v, hl, button, (o_pre, o_post))
            for kk, p in enumerate((mA, mB)):
                a, b, cl = holev[h][p]
                r1, r2 = a >> 2, b >> 2
                v[kk, F.B_HOLE + 0] = max(r1, r2) + 2
                v[kk, F.B_HOLE + 1] = min(r1, r2) + 2
                v[kk, F.B_HOLE + 2] = 1. if r1 == r2 else 0.
                v[kk, F.B_HOLE + 3] = 1. if (a & 3) == (b & 3) else 0.
                v[kk, F.B_HOLE + 4] = abs(r1 - r2)
                v[kk, F.B_HOLE + 5] = PCT169[cl]
                v[kk, F.B_STR:F.B_STR + 3] = np.where(np.isnan(st[p][1:]), -1., st[p][1:])
                v[kk, F.B_CLS:F.B_CLS + 3] = clsv[h][p][1:]
                v[kk, F.B_DRAW:F.B_DRAW + 4] = drwv[h][p]
            for kk, p in enumerate((mA, mB)):
                for s in range(1, 4):
                    if np.isnan(st[p][s]):
                        v[kk, F.B_RNK + s - 1] = -1.
                        continue
                    nb2 = sum(1 for q in seat if q != p and not seat[q]['folded']
                              and st[q][s] > st[p][s])
                    v[kk, F.B_RNK + s - 1] = nb2 + 1
                    v[kk, F.B_BEST + s - 1] = 1. if nb2 == 0 else 0.
            nbd = int(nboard[hidx[h]])
            for s in range(1, 4):
                if nbd >= s + 2:
                    tex = F.board_texture(board[hidx[h]], s + 2)
                    hl[F.H_BRD + (s - 1) * 5:F.H_BRD + s * 5] = tex
                else:
                    hl[F.H_BRD + (s - 1) * 5:F.H_BRD + s * 5] = -1.
            for s in range(4):
                a_s, b_s = st[mA][s], st[mB][s]
                if a_s != a_s or b_s != b_s:
                    hl[F.H_AHEAD + s] = -9.
                    hl[F.H_SGAP + s] = -9.
                else:
                    hl[F.H_AHEAD + s] = 1. if a_s > b_s else (0. if a_s == b_s else -1.)
                    hl[F.H_SGAP + s] = a_s - b_s
            hl[F.H_EQ:F.H_EQ + 4] = -1.
            hl[F.H_EQM:F.H_EQM + 4] = -1.
            ca1, ca2, cla = holev[h][mA]
            cb1, cb2, clb = holev[h][mB]
            hl[F.H_EQ] = EQ169[cla, clb]
            if nbd == 5:
                a5, b5 = st[mA][3], st[mB][3]
                hl[F.H_EQ + 3] = 1. if a5 > b5 else (.5 if a5 == b5 else 0.)
            rowsA[i] = (ca1, ca2)
            rowsB[i] = (cb1, cb2)
            rowh[i] = hidx[h]
            if hl[F.H_BOTHLIVE + 1] > 0 and nbd >= 3:
                eq_need[1].append(i)
            if hl[F.H_BOTHLIVE + 2] > 0 and nbd >= 4:
                eq_need[2].append(i)
            hl[F.H_SDN] = float(mr['players_at_showdown'])
            row = Vout[i]
            row[F.C_A] = v[0, F.P_A]
            row[F.C_B] = v[1, F.P_B]
            row[F.C_SM] = v[0, F.P_SM] + v[1, F.P_SM]
            row[F.C_MX] = np.maximum(v[0, F.P_MX], v[1, F.P_MX])
            row[F.C_MN] = np.minimum(v[0, F.P_MN], v[1, F.P_MN])
            row[F.N_EMIT:] = hl
        for street, nb in ((1, 3), (2, 4)):
            idx = np.array(eq_need[street], np.int64)
            for a0 in range(0, len(idx), 400):
                sel = idx[a0:a0 + 400]
                R = len(sel)
                bd = board[rowh[sel], :nb]
                used = np.concatenate([rowsA[sel], rowsB[sel], bd], axis=1)
                mask = np.ones((R, 52), bool)
                mask[np.arange(R)[:, None], used] = False
                avail = np.nonzero(mask)[1].reshape(R, 52 - 4 - nb)
                run = avail[:, F.PAIRS45] if nb == 3 else avail[:, :, None]
                nr = run.shape[1]
                base = np.concatenate([bd, np.zeros((R, 0), np.int64)], axis=1)
                bb_ = np.repeat(base[:, None, :], nr, axis=1)
                ha = np.repeat(rowsA[sel][:, None, :], nr, axis=1)
                hb = np.repeat(rowsB[sel][:, None, :], nr, axis=1)
                ca = np.concatenate([ha, bb_, run], axis=2).reshape(R * nr, -1)
                cb = np.concatenate([hb, bb_, run], axis=2).reshape(R * nr, -1)
                va = E.evaluate(ca).reshape(R, nr)
                vb = E.evaluate(cb).reshape(R, nr)
                eq = (np.where(va < vb, 1., np.where(va == vb, .5, 0.))).mean(axis=1)
                Vout[sel, F.N_EMIT + F.H_EQ + street] = eq
        for s in range(4):
            e = Vout[:, F.N_EMIT + F.H_EQ + s]
            Vout[:, F.N_EMIT + F.H_EQM + s] = np.where(e < 0, -1., np.maximum(e, 1. - e))
        parts.append(pl.DataFrame({'pair_id': pids, 'hand_id': hids}).hstack(pl.DataFrame(Vout, schema=F.COLS)))
        log(f'  mech chunk {c + 1}/{nchunk} rows {parts[-1].height} {time.time() - t0:.1f}s '
            f'total {time.time() - t_start:.0f}s')
    full = pl.concat(parts)
    assert full.height == keys.height
    full = full.join(chrono.drop('p1_is_A'), on=['pair_id', 'hand_id'], how='left')
    # the PRODUCTION kept-column list; no new pruning
    keep = [c for c in pl.read_parquet(S254 + 'step254_features.parquet', n_rows=1).columns
            if c not in ('pair_id', 'hand_id')]
    assert len(keep) == 662, len(keep)
    missing = [c for c in keep if c not in full.columns]
    assert not missing, missing[:5]
    out = full.select('pair_id', 'hand_id', *keep).with_columns(pl.col(keep).cast(pl.Float32))
    assert out.null_count().to_numpy().sum() == 0, 'nulls in the mechanical block'
    return out, keep


# ------------------------------------------------------------------ the 25 STEP237 columns
def build_237(keys, chunk_hands=6000):
    hid = keys.select('hand_id').unique()
    meta_df = (pl.scan_parquet(DATA + 'hands.parquet').join(hid.lazy(), on='hand_id')
               .select('hand_id', 'phase', 'big_blind', 'board_cards', 'final_pot', 'players_dealt').collect())
    assert (meta_df['phase'] == 'development').all(), 'non-development hand in the STEP258 237 build'
    hand_list = meta_df['hand_id'].to_list()
    nchunk = max(1, int(np.ceil(len(hand_list) / chunk_hands)))
    bounds = np.linspace(0, len(hand_list), nchunk + 1).astype(int)
    rows = []
    t0 = time.time()
    for c in range(nchunk):
        part = pl.DataFrame({'hand_id': hand_list[bounds[c]:bounds[c + 1]]})
        m = meta_df.join(part, on='hand_id')
        meta = {r['hand_id']: {'bb': r['big_blind'], 'board': (r['board_cards'] or '').split(),
                               'final_pot': r['final_pot'], 'players_dealt': r['players_dealt']}
                for r in m.iter_rows(named=True)}
        seats = (pl.scan_parquet(DATA + 'seats.parquet').join(part.lazy(), on='hand_id')
                 .select('hand_id', 'player_id', 'hole_card_1', 'hole_card_2', 'total_contribution',
                         'net_chips', 'folded', 'went_to_showdown', 'won_share').collect())
        acts = (pl.scan_parquet(DATA + 'actions.parquet').join(part.lazy(), on='hand_id')
                .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount',
                        'amount_to', 'pot_before', 'to_call').collect().sort('hand_id', 'action_no'))
        strength, final = hand_tables(seats, meta)
        srows, arows = {}, {}
        for r in seats.iter_rows(named=True):
            srows.setdefault(r['hand_id'], {})[r['player_id']] = r
        for r in acts.iter_rows(named=True):
            arows.setdefault(r['hand_id'], []).append(r)
        sub = keys.join(part, on='hand_id')
        for r in sub.iter_rows(named=True):
            h = r['hand_id']
            if h not in arows:
                continue
            f = pairhand_features(r['pair_id'], h, r['player_1'], r['player_2'], meta[h],
                                  srows[h], arows[h], strength[h], final[h])
            rows.append((r['pair_id'], h, *[f[kk] for kk in F237]))
        log(f'  237 chunk {c + 1}/{nchunk}: {len(rows)} rows {time.time() - t0:.0f}s')
    df = pl.DataFrame(rows, schema=['pair_id', 'hand_id'] + F237, orient='row')
    return df.with_columns(pl.col(F237).cast(pl.Float32))


def main():
    mode = sys.argv[1]
    assert mode in ('prefix', 'full149')
    os.makedirs(OUT, exist_ok=True)
    dest = OUT + f'mechplus_{mode}.parquet'
    assert not os.path.exists(dest), dest
    t0 = time.time()
    keys, hd, prod = universe(mode)
    log(mode, 'keys', keys.shape, 'hand history rows', hd.height, 'pairs', keys['pair_id'].n_unique())
    chrono = build_chrono(keys, hd)
    log('chrono', chrono.shape, round(time.time() - t0, 1), 's')
    mech, MECH = build_mech(keys, chrono)
    log('mech', mech.shape, round(time.time() - t0, 1), 's')
    f237 = build_237(keys)
    log('237', f237.shape, round(time.time() - t0, 1), 's')
    feats = [c for c in prod.columns if c not in ('pair_id', 'hand_id', 'table_id')]
    assert len(feats) == 163, len(feats)
    out = (prod.select('pair_id', 'hand_id', 'table_id', *feats)
           .join(f237, on=['pair_id', 'hand_id'], how='left')
           .join(mech, on=['pair_id', 'hand_id'], how='left'))
    assert out.height == keys.height
    assert out.select(F237 + MECH).null_count().to_numpy().sum() == 0, 'join left nulls'
    out.write_parquet(dest)
    meta = {'mode': mode, 'rows': int(out.height), 'pairs': int(out['pair_id'].n_unique()),
            'n_163': len(feats), 'n_237': len(F237), 'n_mech': len(MECH),
            'column_order_mechplus': feats + F237 + MECH,
            'dest': dest, 'seconds': round(time.time() - t0, 1)}
    json.dump(meta, open(OUT + f'mechplus_{mode}_meta.json', 'w', encoding='utf-8'), indent=1)
    log('wrote', dest, out.shape, round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
