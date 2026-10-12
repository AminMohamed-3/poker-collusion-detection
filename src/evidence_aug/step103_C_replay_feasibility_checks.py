"""Step 103 (Claude): label-free checks of the IDEA_BANK #2 feasibility investigation. No fits, no metrics, and no label,
relevance, family, card or outcome column. Writes experiments/legal_replay_feasibility_C_20260913/verification-C.json."""
import sys
sys.dont_write_bytecode = True
import io, json, hashlib, os
from collections import defaultdict
from datetime import datetime, timezone
import polars as pl
ROOT = 'C:/Data/Projects/Money Making/work/poker/'
CV = ROOT + 'Claude Version/'
EXP = CV + 'experiments/legal_replay_feasibility_C_20260913/'
INV = EXP + 'investigation/'
D = ROOT + 'data/'
OUT = EXP + 'verification-C.json'
RULES = ['DT_single_response_to_partner_aggression', 'SP_single_passive_vs_partner', 'CI_member_open_ge3_outsider_folds']
fsha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
text = lambda p, a, b: ''.join(io.open(p, encoding='utf-8').readlines()[a - 1:b])


def coverage(E, elig, is_agg, is_callish):
    ids = sorted(set(E['hand_id']))
    S = pl.scan_parquet(D + 'seats.parquet').filter(pl.col('hand_id').is_in(ids)).select('hand_id', 'player_id').collect()
    A = (pl.scan_parquet(D + 'actions.parquet').filter(pl.col('hand_id').is_in(ids))
         .select('hand_id', 'action_no', 'street', 'player_id', 'action', 'amount', 'to_call').collect().sort('hand_id', 'action_no'))
    smap, amap = defaultdict(list), defaultdict(list)
    for r in S.iter_rows(named=True):
        smap[r['hand_id']].append(r)
    for r in A.iter_rows(named=True):
        amap[r['hand_id']].append(r)
    pm = {a: (b, c) for a, b, c in pl.read_csv(D + 'development_labels.csv', columns=['pair_id', 'player_1', 'player_2'])
          .select('pair_id', 'player_1', 'player_2').iter_rows()}
    cnt, tot = {k: defaultdict(int) for k in RULES}, defaultdict(int)
    for pid, hid in E.select('pair_id', 'hand_id').iter_rows():
        ev = amap[hid]
        _, foc = elig(smap[hid], ev, *pm[pid])
        last, brow, xrow = len(ev) - 1, False, False
        for k, f in foc.items():
            e, c = ev[f], cnt[k]
            c['eligible'] += 1
            term = f == last and (is_callish(e) or (e['action'] == 'fold' and e['street'] == 'river'))
            c['terminal_exact'] += term
            if k.startswith('CI'):
                allfold = all(x['action'] == 'fold' for j, x in enumerate(ev) if j != f)
                c['open_to_fold_exact'] += allfold
                bound = exact = allfold
            else:
                nonfold = e['action'] != 'fold'
                later = any(is_agg(x) for x in ev[f + 1:] if x['player_id'] == e['player_id'])
                c['focal_nonfold'] += nonfold
                c['nonfold_no_later_member_aggression'] += nonfold and not later
                bound, exact = (nonfold and not later) or term, term
            c['policy_free_upper_bound'] += bound
            brow, xrow = brow or bound, xrow or exact
        tot['distinct_rows_eligible'] += bool(foc)
        tot['distinct_rows_upper_bound'] += brow
        tot['distinct_rows_exact_only'] += xrow
    tot['rule_rows_eligible'] = sum(cnt[k]['eligible'] for k in RULES)
    tot['rule_rows_upper_bound'] = sum(cnt[k]['policy_free_upper_bound'] for k in RULES)
    assert [cnt[k]['eligible'] for k in RULES] == list(E.select([pl.col(k).fill_null(0).sum() for k in RULES]).row(0))
    assert [cnt[k]['terminal_exact'] for k in RULES] == [int(E[k + '__swap_exact'].fill_null(0).sum()) for k in RULES]
    return {k: {x: int(y) for x, y in cnt[k].items()} for k in RULES}, {x: int(y) for x, y in tot.items()}


def pools_and_code(E):
    sys.path.insert(0, ROOT + 'Codex Resume/src')
    sys.path.insert(0, CV + 'src')
    from evidence_aug.fold import fold_pools
    held, audit = {f: set(map(str, fold_pools(f))) for f in (0, 1, 2)}, set(E['table_id'])
    pools = {}
    for n in ('inner0', 'inner1', 'inner2', 'held'):
        j = json.load(io.open(ROOT + 'reports/conditional-action-experiment/policy-screen/' + n + '.json', encoding='utf-8'))
        tp, sp = set(map(str, j['training_pools'])), set(map(str, j['scored_pools']))
        pools[n] = {'training_pools': len(tp), 'scored_pools': len(sp), 'training_and_audit_tables': len(tp & audit),
                    'training_and_outer_held_0_1_2': [len(tp & held[f]) for f in (0, 1, 2)],
                    'scored_and_audit_tables': len(sp & audit)}
    code = {
        'Claude Version/src/features.py:113 batch-level build': ('def build(phase: str', CV + 'src/features.py', 113, 113),
        'Claude Version/src/features.py:190-210 batch player baselines': ("player_base = s.group_by('player_id')", CV + 'src/features.py', 190, 210),
        'Codex Resume/src/symmetry_censor_audit.py:85-97 stage-1 refit reproduces saved OOF': ('Original OOF not reproduced', ROOT + 'Codex Resume/src/symmetry_censor_audit.py', 85, 97),
        'Claude Version/src/run_conditional_policy_screen.py:63 policy trains on label-0 actions': ("(pl.col('label') == 0)", CV + 'src/run_conditional_policy_screen.py', 63, 63),
        'Claude Version/src/run_conditional_policy_screen.py:68 scored tables excluded from training': ("assert not set(train['table_id']) & set(scored['table_id'])", CV + 'src/run_conditional_policy_screen.py', 68, 68),
        'Claude Version/src/build_conditional_action_cache.py:27 style from all other development hands': ('from all other development hands', CV + 'src/build_conditional_action_cache.py', 27, 27)}
    ok = {k: s in text(p, a, b) for k, (s, p, a, b) in code.items()}
    assert all(ok.values()), ok
    return pools, sorted(ok)


def main():
    assert not os.path.exists(OUT) and fsha(EXP + 'feasibility-criteria.json').startswith('6ebe5829')
    man = json.load(io.open(INV + 'MANIFEST.json', encoding='utf-8'))
    assert all(fsha(INV + f) == m['sha256'] for f, m in man['files'].items())
    src = io.open(INV + 'probe.py', encoding='utf-8').read()
    seg = src[src.index('def is_callish'):src.index('FOC = [')]
    assert seg.count('    return row\n') == 1
    ns = {}
    exec(seg.replace('    return row\n', '    return row, foc\n'), ns)
    E = pl.read_parquet(INV + 'eligibility_audit_top20.parquet')
    cov, tot = coverage(E, ns['elig'], ns['is_agg'], ns['is_callish'])
    pools, code = pools_and_code(E)
    pr = json.load(io.open(INV + 'probe_results.json', encoding='utf-8'))
    B = pr['B_audit_top20']
    out = {'step': 103, 'author': 'Claude', 'created_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat(), 'fits': 0,
           'metrics_computed': False, 'label_columns_read': False, 'development_labels_columns_read': ['pair_id', 'player_1', 'player_2'], 'criteria_sha256': fsha(EXP + 'feasibility-criteria.json'),
           'investigation_manifest_sha256': fsha(INV + 'MANIFEST.json'), 'probe_sha256': fsha(INV + 'probe.py'),
           'script_sha256': fsha(os.path.abspath(__file__)), 'probe_times_s': pr['times'],
           'public_state_dev_sample': pr['A_dev_sample_full'], 'public_state_audit_top20': pr['A_audit_top20_public_only'],
           'eligibility_probe': {k: B[k] for k in RULES},
           'n_actions_quantiles': {'audit_top20': B['n_actions_quantiles_10_25_50_75_90'],
                                   'donors': pr['C_donor_eligibility']['n_actions_quantiles_10_25_50_75_90']},
           'policy_free_coverage': cov, 'coverage_totals': tot, 'ca_policy_pools': pools, 'code_facts_checked': code}
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write(json.dumps(out, indent=2) + '\n')
    print(json.dumps(tot), json.dumps(cov), 'sha', fsha(OUT))


if __name__ == '__main__':
    main()
