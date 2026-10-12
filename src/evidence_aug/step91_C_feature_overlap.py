"""Step 91 [C] (author: claude).  CODE-READING INVENTORY, no model fitting, no experiment.

Question: are Codex's evidence-side features a genuinely different view of the same hands, or
largely a re-derivation of what our stage 2 already sees?

Our side
  stage 1 prod design = 382 base ranker columns + chrono_frac = 383 (base382 is the audit OOF surface)
  stage 2 (r382DTSP / structCI) sees, per audit_common.load_current / tabpfn3_relative_prepare:
      shared = struct(70) + [chrono_rank, chrono_frac, n_hands, oof, oof_rank]   -> 75
      extra  = the 382 minus family_* minus the struct name collision              -> 378
      DT/SP width 453, CI width 75.

Codex side
  Nine distinct new-named feature families built in Codex Resume/src, each frozen with a manifest.
  Every one of them was measured on the SAME 162-pair training-only audit against the SAME
  incumbent.  Their gate outcomes are read from their own results.json files here.

Everything below is read-only outside Claude Version/.  No labels, no held folds, no submission.
Output: Claude Version/experiments/feature_overlap_C_20260913/feature-overlap-inventory.json
"""
import os

for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '2')

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import polars as pl

ROOT = Path('C:/Data/Projects/Money Making/work/poker')
CV = ROOT / 'Claude Version'
CX = ROOT / 'Codex Resume'
OUT = CV / 'experiments' / 'feature_overlap_C_20260913'
BUNDLE = ROOT / 'reports' / 'conditional-action-experiment' / 'screen-results' / 'conditional-evidence.joblib'
STRUCTF = CV / 'experiments' / 'label_process_20260911' / 'audit-structural-features.parquet'

PROVENANCE = [
    {'issue': 'arm double-counting, caught mid-run before any number was written',
     'detail': 'My first pass was going to multiply each family’s column count by its number of arms. Six of the nine '
               'families (mp, rs, os, od, nw, wh) write the SAME column names to two or three arm parquets, so that would '
               'have inflated Codex’s column count roughly 2x. Distinct names are now counted once per family and '
               'arm-level differences (notably the rs_ terminal arm) are reported as unique QUANTITIES with new_names=0.'},
    {'issue': 'cs_ prefix collision between the two trees',
     'detail': 'Our own unwired src/build_conditional_sizing_screen.py emits cs_size_residual / cs_absolute_size_residual / '
               'cs_weak_excess_size / ... while Codex’s contestable-state features also use a cs_ prefix. They are '
               'different quantities. No name in either set actually collides (checked: the literal-overlap set is entirely '
               'eq_*), but the prefix must not be used as an identity test.'},
    {'issue': 'pot_bb name collision inside our own stack',
     'detail': 'pot_bb appears both in the 382 ranker columns and in the 70 structural columns, so stage 2’s extra list is '
               '378 rather than 379. Codex hit the same thing and recorded "struct_overlap": ["pot_bb"] in '
               'outputs_contestable_exploratory_20260912/fit.json. The script asserts the collision list equals exactly '
               '["pot_bb"] rather than assuming it.'},
    {'issue': 'classification bias',
     'detail': 'Per the brief, every genuinely ambiguous pair was pushed into the SEMANTICALLY OVERLAPPING bucket, not the '
               'unique bucket. The clearest instances are cs_effective_remaining_pot (a stack-to-pot ratio we never actually '
               'form) and the whole rs_ prefix arm (a different weighting of a quantity we do have). Reading those the other '
               'way would move 34 columns from overlapping to unique.'},
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


# --------------------------------------------------------------------------------------- OUR SIDE

def our_side():
    features = list(joblib.load(BUNDLE)['features'])
    assert len(features) == 382, len(features)
    prod = features + ['chrono_frac']

    def pick(test):
        return [c for c in features if test(c)]

    blocks = [
        {'name': 'base_pair_aggregate', 'n': len(pick(lambda c: not c.startswith(('v2_', 'context_', 'family_', 'seq_', 'eq_', 'ca_')))),
         'constructed_in': 'src/features.py build() lines ~155-240 (25 per-seat quantities x {sum,gap,min,max} + 7 pair extras)',
         'view': 'absolute chip accounting + hand-level outcome + crude cross-hand player baseline'},
        {'name': 'v2_events', 'n': len(pick(lambda c: c.startswith('v2_'))),
         'constructed_in': 'src/advanced_features.py directional_events() event_exprs (28 events x {sum,max})',
         'view': 'card-strength-weighted relational action events, partner-directed vs other-directed'},
        {'name': 'context_v2_events', 'n': len(pick(lambda c: c.startswith('context_v2_'))),
         'constructed_in': 'src/rank_evidence.py context(): within-pair z-score of each v2_ column',
         'view': 'within-pair standardisation of the above'},
        {'name': 'family_onehot', 'n': len(pick(lambda c: c.startswith('family_'))),
         'constructed_in': 'src/rank_evidence.py family_columns()', 'view': 'routing'},
        {'name': 'seq', 'n': len(pick(lambda c: c.startswith('seq_'))),
         'constructed_in': 'src/sequence_evidence_features.py NAMES (23), prefixed seq_ in evidence_aug/frame.py',
         'view': 'within-hand ordering of pair aggression / outsider capitulation'},
        {'name': 'context_seq', 'n': len(pick(lambda c: c.startswith('context_seq_'))),
         'constructed_in': 'evidence_aug/frame.py context(.., SEQ)', 'view': 'within-pair standardisation'},
        {'name': 'eq', 'n': len(pick(lambda c: c.startswith('eq_') and not c.startswith('context_'))),
         'constructed_in': 'src/equity_evidence_features.py FEATURES (12 NAMES x {_sum,_max}); heads-up equity, 256 preflop samples',
         'view': 'card/equity information, heads-up proxy, side pots explicitly excluded'},
        {'name': 'context_eq', 'n': len(pick(lambda c: c.startswith('context_eq_'))),
         'constructed_in': 'evidence_aug/frame.py context(.., EQ)', 'view': 'within-pair standardisation'},
        {'name': 'ca', 'n': len(pick(lambda c: c.startswith('ca_') and not c.startswith('context_'))),
         'constructed_in': 'src/run_conditional_policy_screen.py hand_features() over src/conditional_action_features.py INPUTS',
         'view': 'policy-deviation: action-class residual vs a pool-separated normal-action reference'},
        {'name': 'context_ca', 'n': len(pick(lambda c: c.startswith('context_ca_'))),
         'constructed_in': 'evidence_aug/frame.py context(.., caf)', 'view': 'within-pair standardisation'},
    ]
    assert sum(b['n'] for b in blocks) == 382

    struct = [c for c in pl.read_parquet(STRUCTF).columns if c not in ('pair_id', 'hand_id')]
    chrono = ['chrono_rank', 'chrono_frac', 'n_hands']
    s1 = ['oof', 'oof_rank']
    shared = struct + chrono + s1
    extra = [f for f in features if not f.startswith('family_') and f not in struct]
    collisions = sorted(set(features) & set(struct))
    universe = sorted(set(prod) | set(shared))
    return {
        'prod_design_n': len(prod),
        'base382_n': len(features),
        'reconciles_to_383': len(prod) == 383 and len(features) == 382,
        'blocks': blocks,
        'columns_382': features,
        'stage2': {
            'struct_n': len(struct), 'struct_columns': struct,
            'chrono': chrono, 'stage1_score': s1,
            'shared_n': len(shared), 'extra_n': len(extra),
            'dtsp_width': len(shared) + len(extra), 'ci_width': len(shared),
            'name_collisions_struct_vs_382': collisions,
            'constructed_in': 'Claude Version/src/evidence_aug/step65_structural_evidence.py hand_features(); '
                              'routed by step65i/65k and re-derived verbatim by Codex Resume/src/audit_common.py load_current()',
        },
        'universe_n': len(universe),
        'universe': universe,
    }


# ------------------------------------------------------------------------------------- CODEX SIDE

CODEX_SOURCES = [
    ('cs', 'contestable_state', 'src/contestable_state_features.py', 'research/contestable_state_20260912/feature_manifest.json', 'features',
     ['single'], 'experiments/contestable_state_20260912/results.json', 'contestable_state'),
    ('mp', 'marginal_payout', 'src/marginal_payout_features.py', 'research/marginal_payout_20260912/full-freeze.json', 'features',
     ['single', 'layered'], 'experiments/marginal_payout_20260912/results.json', 'layered_payout'),
    ('rs', 'retrospective_strength', 'src/retrospective_strength_features.py', 'research/retrospective_strength_20260912/freeze.json', 'features',
     ['prefix', 'terminal'], 'experiments/retrospective_strength_20260912/results.json', 'terminal_strength'),
    ('os', 'ordered_surprise', 'src/ordered_surprise_features.py', 'research/ordered_surprise_20260912/feature-freeze.json', 'features',
     ['ordered', 'marginal_control'], 'experiments/ordered_surprise_20260912/results.json', 'ordered'),
    ('od', 'outsider_card', 'src/outsider_card_features.py', 'research/outsider_decisions_20260912/feature-freeze.json', 'features',
     ['linked', 'card_control'], 'experiments/outsider_cards_20260912/results.json', 'linked'),
    ('nw', 'nori_wager', 'src/nori_wager_feature_build.py', 'research/nori_wager_20260912/features-receipt.json', 'columns',
     ['tree_quantiles', 'nori'], 'experiments/nori_wager_20260912/results.json', 'nori'),
    ('wh', 'within_hand_sequence', 'src/within_hand_feature_build.py', 'research/within_hand_sequence_20260912/features-receipt.json', 'columns',
     ['query_only', 'shuffled', 'ordered'], 'experiments/within_hand_sequence_20260912/results.json', 'ordered'),
    ('eq4096', 'preflop_equity_precision', 'src/preflop_equity_precision_features.py', 'research/preflop_equity_precision_20260912/freeze.json', 'features',
     ['equity4096'], 'experiments/preflop_equity_precision_20260912/results.json', 'equity4096'),
]


def codex_side(our_struct, our_chrono):
    families = []
    for prefix, name, module, manifest, key, arms, results, primary in CODEX_SOURCES:
        cols = list(read_json(CX / manifest)[key])
        r = read_json(CX / results)
        comps = r.get('comparisons', {})
        families.append({
            'prefix': prefix, 'name': name, 'module': 'Codex Resume/' + module,
            'manifest': 'Codex Resume/' + manifest,
            'n_distinct_columns': len(cols), 'columns': cols, 'arms': arms,
            'audit_arm_deltas': {k: round(v['mean_delta'], 5) for k, v in comps.items() if isinstance(v, dict)},
            'audit_gate_pass': {k: bool(v.get('gate_pass')) for k, v in comps.items() if isinstance(v, dict)},
            'primary_arm': primary,
        })

    # RZ within-pair relative encoding: tabpfn3_relative_prepare.relativise() over struct + chrono.
    rel_base = list(our_struct) + list(our_chrono)
    rel_cols = [c + s for c in rel_base for s in ['__wr', '__wz']]
    assert len(rel_cols) == 146, len(rel_cols)  # matches the source's own `assert len(rcols)==146`
    families.append({
        'prefix': '__wr/__wz', 'name': 'tabpfn3_relative_encoding',
        'module': 'Codex Resume/src/tabpfn3_relative_prepare.py relativise()',
        'manifest': 'Codex Resume/experiments/tabpfn3_relative_20260912/context-receipt.json',
        'n_distinct_columns': len(rel_cols), 'columns': rel_cols,
        'arms': ['RZ'],
        'audit_arm_deltas': {}, 'audit_gate_pass': {'RZ_vs_T': False},
        'primary_arm': 'RZ',
    })
    return families


# ------------------------------------------------------------------- SEMANTIC JUDGEMENTS (by hand)

SEMANTIC = [
    {'codex': 'eq_* (24, 4096 preflop samples)', 'ours': 'eq_* (24, 256 preflop samples)',
     'quantity': 'heads-up equity-based decision surplus/deficit; identical module, identical names',
     'codex_code': "preflop_equity_precision_features.py: FEATURES = ['eq_'+name+suffix ...]; assert len==24; regimes 256 vs 4096 sample_indices",
     'our_code': "equity_evidence_features.py: PREFLOP_SAMPLES=256; FEATURES = ['eq_'+name+suffix for suffix in ['_sum','_max'] for name in NAMES]",
     'verdict': 'literal duplicate, differs only in Monte-Carlo precision', 'n_codex_columns': 24},

    {'codex': '*__wr / *__wz (146)', 'ours': 'oof_rank (wired) and the full step-78 within-pair rank/z design (built, not wired)',
     'quantity': 'within-pair (within candidate set) rank and z-score of each structural/chronology column',
     'codex_code': "tabpfn3_relative_prepare.relativise(): v.rank('average').over('pair_id') and ((v-v.mean().over('pair_id'))/(v.std().over('pair_id')+1e-9))",
     'our_code': "step78_C_stage2_selection.py: 'For each structural column, add its rank and z-score computed within the pair's own candidate set.'",
     'verdict': 'same construction, independently reinvented on both sides', 'n_codex_columns': 146},

    {'codex': 'cs_contribution_gap_partner_bb_{mean,max}', 'ours': 'contribution_bb_gap',
     'quantity': 'imbalance between the two pair members\u2019 chip contributions, in big blinds',
     'codex_code': "contestable_state_features: 'gap_bb': abs(contribution[actor]-contribution[partner])/big  (per pair decision, then mean/max)",
     'our_code': "features.py: contribution_bb = total_contribution/big_blind; symmetric += (c1-c2).abs().alias(c+'_gap')  (end of hand)",
     'verdict': 'same quantity, decision-time vs end-of-hand', 'n_codex_columns': 2},

    {'codex': 'cs_effective_remaining_bb_{mean,max}, cs_effective_remaining_pot_{mean,max}', 'ours': 'stack_bb_{sum,gap,min,max}, pot_bb, fold_pot_odds_*',
     'quantity': 'effective pair stack, absolute and relative to the pot',
     'codex_code': "effective = min(actor_remaining, partner_remaining) if both_can_wager else 0; /big and /max(pot_before,1)",
     'our_code': "stack_bb = starting_stack/big_blind, then min/max/sum/gap; pot_bb = final_pot/big_blind",
     'verdict': 'same quantity family, decision-time vs hand-start/hand-end', 'n_codex_columns': 4},

    {'codex': 'cs_live_opponents_can_bet_{mean,max}', 'ours': 'min_active_{sum,gap,min,max}, players_at_showdown',
     'quantity': 'how many opponents are still live and able to wager',
     'codex_code': "live_opponents = sum(1 for p in by_player if p!=actor and p not in folded and p not in all_in and starting[p]>contribution[p])",
     'our_code': "features.py: pl.col('players_active').min().alias('min_active')",
     'verdict': 'same quantity, Codex excludes all-in players', 'n_codex_columns': 2},

    {'codex': 'cs_decisions', 'ours': 'n_actions_{sum,gap,min,max}, ca_actions',
     'quantity': 'count of pair-member decisions in the hand',
     'codex_code': "feature['cs_decisions'] += 1  (actor in pair and partner not folded)",
     'our_code': "features.py: pl.len().alias('n_actions'); run_conditional_policy_screen: pl.len().alias('ca_actions')",
     'verdict': 'same quantity', 'n_codex_columns': 1},

    {'codex': 'cs_{fold,check,call,aggr}_partner_active_{count,bb_sum} (6)', 'ours': 'n_check/n_call/n_aggressive/agg_bb/call_bb (x4 stats), v2_partner_call_count_*, v2_partner_fold_count_*',
     'quantity': 'counts and big-blind sums of each action type taken while the partner is still in the hand',
     'codex_code': "feature[f'cs_call_{suffix}_count'] += 1; feature[f'cs_call_{suffix}_bb_sum'] += amount/big  (suffix='partner_active')",
     'our_code': "features.py action_stats: n_check, n_call, n_aggressive, agg_bb, call_bb; advanced_features: 'partner_call_count': call*to_partner",
     'verdict': 'same quantities', 'n_codex_columns': 6},

    {'codex': 'mp_{partner,outsider}_{call,fold}_own_{loss,gain}_bb_{sum,max} (16)', 'ours': 'eq_partner_call_deficit_bb_*, eq_partner_call_surplus_bb_*, eq_partner_fold_surplus_pot_bb_*',
     'quantity': 'marginal chip value to the actor of calling versus folding, in big blinds',
     'codex_code': "delta = cp[p]-fp[p]-(cost if p==actor else 0); own_loss=max(0.,-delta[actor]); own_gain=max(0.,delta[actor]); /big_blind",
     'our_code': "equity_evidence_features: surplus = max(0., q*(pot+cost)-cost)/big_blind; deficit = max(0., cost-q*(pot+cost))/big_blind",
     'verdict': 'same quantity: exact multiway showdown payout vs a heads-up equity proxy of the same call/fold delta',
     'n_codex_columns': 16},

    {'codex': 'rs_* (32 names, prefix arm)', 'ours': 'v2_soft_strength_fold_*, v2_soft_ahead_fold_*, v2_soft_late_check_*, v2_soft_ahead_check_*, v2_late_aggressive_strength_*, v2_weak_call_*, v2_behind_partner_call_*',
     'quantity': 'own card strength (and strength relative to the partner) weighted onto each action type, split by whether the action faces the partner',
     'codex_code': "accum[mode][actor][f'{context}_{action}_strength'] += s[actor]*w ; ..._ahead_gap += max(0., s[actor]-s[partner])*w ; w=log1p(pot_before/bb) ; context = 'partner' if previous==partner else 'other'",
     'our_code': "advanced_features event_exprs: 'soft_strength_fold': fold*to*price*strength ; 'soft_ahead_fold': fold*to*price*ahead ; 'late_aggressive_strength': agg*late*strength ; ahead = actor_rank < partner_rank",
     'verdict': 'same quantity family; Codex has a continuous ahead_gap where we have a binary ahead flag, and a different pot-size weight',
     'n_codex_columns': 32},

    {'codex': 'wh_summary mean/max blocks over residual dims 0-3 and -log p_obs (10 of 26), and the summed count field',
     'ours': 'ca_residual_{0..3}_{mean,absmax}, ca_surprise_{mean,absmax}, ca_actions',
     'quantity': 'per-action policy residual against a normal-action reference, and the negative log probability of the action actually taken',
     'codex_code': "within_hand_feature_build: residual=np.eye(4)[y]-p ; v=[residual, -log(max(obs,1e-7)), sum(residual**2), entropy, obs]",
     'our_code': "run_conditional_policy_screen.hand_features: residual=(np.eye(4)[y]-probability)/sqrt(probability*(1-probability)+.05), clip(-4,4); ca_surprise=-log(max(p_obs,1e-5))",
     'verdict': 'same quantity; Codex uses a neural prefix-conditioned reference where ours is a LightGBM pointwise reference over hand-crafted prefix state (own_prior_actions, partner_prior_actions, ...)',
     'n_codex_columns': 11},
]

UNIQUE = [
    {'codex': 'cs_actor_unreachable_frac_{mean,max}, cs_partner_unreachable_frac_{mean,max}', 'n': 4, 'new_names': 4,
     'measures': 'fraction of the current pot the actor (or the all-in partner) cannot win because of side-pot capping',
     'view': 'absolute chip accounting / side-pot structure',
     'codex_code': "contestable_after_call = sum(min(value + (call_paid if p==actor else 0), actor_target) for p,value in contribution.items()); actor_unreachable = (total_after_call - contestable_after_call)/max(total_after_call,1)",
     'counterpart_in_our_codebase': None,
     'computable_from_data_we_already_load': True,
     'needs': 'data/hands.parquet + data/actions.parquet only (already loaded by step65_structural_evidence.build)'},

    {'codex': 'cs_decisions_{partner_allin,outsider_allin,any_allin,both_can_wager}', 'n': 4, 'new_names': 4,
     'measures': 'how many pair decisions were taken with the partner / an outsider / anyone already all-in, and with both members still able to wager',
     'view': 'absolute chip accounting / all-in state',
     'codex_code': "all_in.add(actor) when contribution[actor]==starting[actor]; both_can_wager = actor_remaining>0 and partner_remaining>0 and not partner_allin",
     'counterpart_in_our_codebase': None,
     'computable_from_data_we_already_load': True,
     'needs': 'same three parquet files; our builders currently fold all_in into call/raise and never track the state'},

    {'codex': 'cs_{fold,check,call,aggr}_partner_allin_{count,bb_sum}', 'n': 6, 'new_names': 6,
     'measures': 'action counts and big-blind sums taken specifically while the partner is all-in',
     'view': 'absolute chip accounting / all-in state',
     'codex_code': "suffix = 'partner_allin' if partner_allin else 'partner_active'; feature[f'cs_aggr_{suffix}_bb_sum'] += amount/big",
     'counterpart_in_our_codebase': None,
     'computable_from_data_we_already_load': True,
     'needs': 'same three parquet files'},

    {'codex': 'mp_{partner,outsider}_{call,fold}_{partner_gain,outsider_gain}_bb_{sum,max}', 'n': 16, 'new_names': 16,
     'measures': 'counterfactual chip transfer to the PARTNER and to the OUTSIDERS caused by the actor calling instead of folding (exact multiway showdown, single-pot and layered side-pot arms)',
     'view': 'card/equity information + explicit transfer accounting',
     'codex_code': "delta={p:cp[p]-fp[p]-(cost if p==actor else 0) for p in players}; partner_gain=max(0.,delta[partner]); outsider_gain=max(0.,sum(delta[p] for p not in pair))",
     'counterpart_in_our_codebase': 'src/multiway_evidence_features.py mw_partner_fold_partner_gain / mw_partner_fold_outsider_gain / mw_partner_call_outsider_share_cost (24 columns, BUILT BUT NOT WIRED into the 383/453)',
     'computable_from_data_we_already_load': True,
     'needs': 'seats.parquet hole cards for all six seats + phevaluator; already loaded'},

    {'codex': 'rs_* terminal arm (32 columns, same names as the prefix arm, separate parquet)', 'n': 32, 'new_names': 0,
     'measures': 'the same strength-weighted action summaries but with card strength evaluated on the FINAL board, i.e. hindsight strength attached to an earlier-street action',
     'view': 'card information / retrospective (hindsight)',
     'codex_code': "terminal={p:strength(cards[p],board) for p in pair}; for mode,s in [('prefix',cache[n]),('terminal',terminal)]",
     'counterpart_in_our_codebase': None,
     'counterpart_note': 'partial only: won_share_*, net_bb_*, went_to_showdown_* are hindsight OUTCOMES; '
                         'advanced_features.street_rank() deliberately uses the street prefix, never the final board. '
                         'This arm contributes NO new column names (it reuses the 32 rs_ names), so it is counted as a '
                         'unique QUANTITY but not as a unique NAME.',
     'computable_from_data_we_already_load': True,
     'needs': 'hands.parquet board_cards + seats.parquet hole cards; already loaded'},

    {'codex': 'os_r{i}_r{j}_{mean,absmean} (32)', 'n': 32, 'new_names': 32,
     'measures': 'outer product of the policy residual vectors of two ADJACENT, cross-partner actions inside the same street, against an exact within-actor/street permutation control',
     'view': 'within-hand sequence / ordering, cross-partner coupling',
     'codex_code': "edges = [i for i in range(n-1) if street[i]==street[i+1] and direction[i]!=direction[i+1]]; ordered_signed += np.outer(residual[i], residual[i+1])",
     'counterpart_in_our_codebase': 'src/directional_conditional_features.py cd_<name>_product and cd_response_{0,1,2}_pressure (39 columns, BUILT BUT NOT WIRED); those are pair-hand-level products of direction means, not adjacent-action products',
     'computable_from_data_we_already_load': True,
     'needs': 'the existing ca_ action-level cache (reports/conditional-action-experiment/action-cache) plus the saved reference policy'},

    {'codex': 'od_{one,both}_{fold,call,aggr}_{strength,ahead}_{sum,max} (24)', 'n': 24, 'new_names': 24,
     'measures': 'the responding OUTSIDER\u2019s own disclosed card strength (and how far it beats the aggressor\u2019s) weighted onto that outsider\u2019s fold/call/raise, split by whether one or both pair members applied pressure; plus a 24-permutation shared-card control',
     'view': 'card information about third parties',
     'codex_code': "strength = matrix[i, actor[i]] * weight[i]; ahead = max(0., matrix[i, actor[i]] - aggressor[i]) * weight[i]",
     'counterpart_in_our_codebase': None,
     'counterpart_note': 'none for outsider CARDS. We have outsider ACTION counts only: seq_outsider_fold_to_pair_n, '
                         'seq_outsider_call_to_pair_bb, opp_fold_*, opp_call_*, v2_other_strength_fold_* '
                         '(that last one uses the PAIR MEMBER\u2019s strength, not the outsider\u2019s)',
     'computable_from_data_we_already_load': True,
     'needs': 'seats.parquet hole_card_1/2 for the non-pair seats; already read by conditional_action_features.extract_hand'},

    {'codex': 'nw_summary_0..25 (26)', 'n': 26, 'new_names': 26,
     'measures': 'deviation of the actual WAGER SIZE from a conditional q10/q50/q90 quantile model of the bet size, standardised, clipped, aggregated per direction with mean / direction-gap / max / counts',
     'view': 'opponent-relative / policy deviation, on action MAGNITUDE rather than action CLASS',
     'codex_code': "lower=clip(actual-q50,-3,3); standardized=clip((actual-q50)/max(q90-q10,.05),-5,5); below/above/outside/width; multiway_lower=above*(1-strength)*(players_active>=3); partner_upper=below*strength*(facing_partner>0)",
     'counterpart_in_our_codebase': 'src/build_conditional_sizing_screen.py cs_size_residual / cs_absolute_size_residual / cs_weak_excess_size / cs_partner_facing_excess_size / cs_partner_size_covariance / cs_preflop_size_information / cs_weak_multiway_size (BUILT BUT NOT WIRED into the 383/453)',
     'computable_from_data_we_already_load': True,
     'needs': 'the existing action cache; a quantile regressor must be fitted, but on data we already have'},

    {'codex': 'wh_summary direction-asymmetry block (8 of the |a-b| fields) + the count-asymmetry field (1)', 'n': 9, 'new_names': 9,
     'measures': 'per-direction split of the policy-residual summaries, so that |member A mean - member B mean| becomes a feature in its own right',
     'view': 'opponent-relative / policy deviation, made asymmetric between the two pair members',
     'codex_code': "pair_summary: np.column_stack([(means[:,0]+means[:,1])/2, np.abs(means[:,0]-means[:,1]), np.maximum(maxima[:,0],maxima[:,1]), counts.sum(1)/24, np.abs(counts[:,0]-counts[:,1])/24])",
     'counterpart_in_our_codebase': 'src/directional_conditional_features.py cd_<name>_gap = |a-b| over per-direction ca_ means and cd_both_actors (BUILT BUT NOT WIRED). Our wired ca_ pools both directions before aggregating, so the asymmetry is destroyed.',
     'computable_from_data_we_already_load': True,
     'needs': 'the existing ca_ action-level cache; a two-line change to the group_by in run_conditional_policy_screen.hand_features'},

    {'codex': 'wh_summary uncertainty dims (sum of squared residual, predictive entropy, observed action probability) in the mean and max blocks', 'n': 6, 'new_names': 6,
     'measures': 'how UNCERTAIN the reference policy was at each decision, not just how far the action deviated from it',
     'view': 'opponent-relative / policy deviation, second moment',
     'codex_code': "v = [residual(4), -log(max(obs,1e-7)), float((residual**2).sum()), float(-(p*np.log(p+1e-9)).sum()), obs]",
     'counterpart_in_our_codebase': None,
     'counterpart_note': 'ca_surprise is -log p_obs (first moment of surprise). We never emit the entropy of the reference '
                         'distribution nor the Brier term, so we cannot distinguish "the model was confident and wrong" '
                         'from "the model had no idea".',
     'computable_from_data_we_already_load': True,
     'needs': 'the existing ca_ action-level cache and the already-saved reference policy; three extra aggregations'},
]

VIEW_TYPES = {
    'absolute_chip_accounting': {
        'ours': 'STRONG at hand level: contribution_bb/net_bb/stack_bb/agg_bb/call_bb/fold_pot_odds x {sum,gap,min,max}, pot_bb, plus struct A_/B_stack_bb, put_bb, net_bb, pot_bb, pf_open_size_bb',
        'codex_addition': 'decision-time versions, plus side-pot reachability and all-in state (cs_ 14 unique columns)',
        'we_lack': 'side-pot / uncontestable-money structure and all-in state; everything else is present at a different time point'},
    'card_equity_information': {
        'ours': 'STRONG for the pair: hole_strength_*, A_/B_chen, chen_max/min, rank per street, eq_ heads-up equity (24), ca_ own_strength/partner_strength/board_* features',
        'codex_addition': 'outsider hole cards (od_ 24), exact multiway showdown payouts with transfer decomposition (mp_ 16 unique), hindsight terminal-board strength (rs_ terminal arm)',
        'we_lack': 'THIRD-PARTY CARD INFORMATION entirely, side-pot-aware payout, and hindsight strength'},
    'within_hand_sequence_ordering': {
        'ours': 'MODERATE: seq_ 23 ordering events, v2_ to_partner (previous_aggressor), ca_ prior-action state (prior_actions, own_prior_actions, partner_prior_actions, own_street_aggression, ...)',
        'codex_addition': 'os_ adjacent cross-partner residual outer products with an exact permutation control (32); wh_ neural prefix-conditioned reference',
        'we_lack': 'explicit ORDER-INTERACTION terms between consecutive partner actions; we have order-dependent indicators but no coupling term in the wired set'},
    'cross_hand_chronological': {
        'ours': 'THIN: chrono_rank / chrono_frac / n_hands, plus aggression_residual / check_residual / contribution_residual / net_residual (deviation from that player\u2019s mean over all their hands in the phase)',
        'codex_addition': 'NONE. Codex built no cross-hand feature at all; research/recent_player_history_20260912/coverage.json is a data-availability count only',
        'we_lack': 'both sides lack a proper prior-hands-only rolling player history; neither side can supply the other here'},
    'opponent_relative_policy_deviation': {
        'ours': 'STRONG on action CLASS: ca_ 33 + context_ca_ 33 from a pool-separated normal-action reference; the residual formula is identical to Codex\u2019s',
        'codex_addition': 'action MAGNITUDE deviation (nw_ 26), per-direction asymmetry and uncertainty summaries (wh_ 15 unique)',
        'we_lack': 'conditional bet-SIZE deviation, and the A-vs-B asymmetry of the policy residual'},
}

REVERSE = {
    'headline': 'Codex\u2019s live evidence path imports our frame verbatim, so there is no wired column we have that Codex lacks.',
    'evidence': [
        'Codex Resume/src/audit_common.py load_current(): imports evidence_aug.fold.load_fold, evidence_aug.frame.FAMILIES, '
        'evidence_aug.step14_audit_truncation, evidence_aug.step65_structural_evidence.summarize, evidence_aug.step65b_family_stage2.boot, '
        'and reads Claude Version/experiments/label_process_20260911/audit-structural-features.parquet.',
        'Codex Resume/src/contestable_state_production.py imports step65k_r382_candidate wholesale and adds the 29 cs_ fields to it.',
        'Codex Resume/src/tabpfn3_confirm_prepare.py asserts xtr.shape[1] in (75, 453): exactly our shared/shared+extra widths.',
    ],
    'what_we_have_that_they_have_not_re_derived': [
        'the 382-column stage-1 ranker itself (base pair aggregates, v2_ relational events, seq_, eq_, ca_) - they consume it, they did not rebuild it',
        'the 70 structural columns of step65 (preflop open / limpers / isolation shape / c-bet / who-reached-flop) - consumed, not rebuilt',
        'chrono_frac and the chronology stage-1 recipe - consumed, not rebuilt',
        'the listwise/candidate-set-sweep formulation of step 78 - Codex explicitly ceded that axis ("Claude owns relative features/listwise trials")',
    ],
    'ours_built_but_not_wired_that_would_be_worth_offering': [
        'src/multiway_evidence_features.py mw_* (24): multiway equity with partner_gain / outsider_gain terms - the same idea as mp_',
        'src/directional_conditional_features.py cd_* (39): per-direction ca_ means with |gap| and cross products - the same idea as os_ and the wh_ asymmetry block',
        'src/build_conditional_sizing_screen.py cs_size_* : conditional bet-size residual - the same idea as nw_',
        'src/conditional_bucket_features.py cb_cross_{a}_{b} / cb_unseen_{a}_{b}: action x bucket residual cross terms',
    ],
}

UNCERTAIN = [
    {'item': 'rs_ prefix vs terminal arms share the same 32 column names',
     'note': 'The two arms are written to two parquet files (prefix.parquet, terminal.parquet) with identical column names. '
             'For the Jaccard denominator I count 32 DISTINCT names once. The terminal arm is listed as a unique QUANTITY '
             '(32 columns of hindsight strength) even though it contributes no new names. Same pattern for mp_ (single/layered), '
             'os_ (ordered/marginal_control), od_ (linked/card_control), nw_ (tree_quantiles/nori x 5 folds), wh_ (3 arms x 5 folds).'},
    {'item': 'wh_summary_0..25 and nw_summary_0..25 are positionally named',
     'note': 'The 26 fields are index-named, so the mapping from index to quantity is read off pair_summary() and the v/values '
             'column_stack. I decomposed them as 8 base dimensions x {mean, |direction gap|, max} = 24 plus 2 count fields. '
             'My split of the 26 into overlapping (11) and unique (15) for wh_ follows that decomposition; the exact index-to-name '
             'assignment inside each block is inferred from column order, not from a written manifest.'},
    {'item': 'whether cs_effective_remaining_pot (stack-to-pot ratio at decision) should count as overlapping',
     'note': 'We have stack_bb and pot_bb separately but never their ratio at a decision point. Per the instruction to be '
             'conservative I classified it as OVERLAPPING. If you want the aggressive reading, add 2 columns to unique.'},
    {'item': 'contestable_state_production.py wiring status',
     'note': 'The 29 cs_ fields ARE in the currently best public submission (submission_contestable_exploratory.csv, public .89457). '
             'That build ran under an explicit one-off user exception with strict_confirmation_gate_passed=false. I record it as '
             '"in a submitted artefact but gate-failed", separate from "wired into a confirmed pipeline".'},
    {'item': 'tabpfn3_relative RZ arm',
     'note': 'results.json has primary_eligible=false and no top-level comparisons block in the shape the others use, so I record '
             'gate_pass=False from primary_eligible rather than from a mean_delta. RESULTS_20260912.md states the RZ arm failed admission.'},
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ours = our_side()
    struct = ours['stage2']['struct_columns']
    chrono = ours['stage2']['chrono']
    families = codex_side(struct, chrono)

    codex_names = []
    for f in families:
        codex_names.extend(f['columns'])
    assert len(codex_names) == len(set(codex_names)), 'unexpected duplicate column name across Codex families'
    codex_set = set(codex_names)
    prod_set = set(ours['columns_382'] + ['chrono_frac'])
    universe_set = set(ours['universe'])

    literal_vs_prod = sorted(codex_set & prod_set)
    literal_vs_universe = sorted(codex_set & universe_set)
    jac_prod = len(literal_vs_prod) / len(codex_set | prod_set)
    jac_universe = len(literal_vs_universe) / len(codex_set | universe_set)

    n_semantic = sum(s['n_codex_columns'] for s in SEMANTIC)
    n_literal_in_semantic = next(s['n_codex_columns'] for s in SEMANTIC if s['verdict'].startswith('literal'))
    n_semantic_nonliteral = n_semantic - n_literal_in_semantic
    n_unique = sum(u['n'] for u in UNIQUE)
    # the rs_ terminal arm contributes no new NAMES; it is a unique quantity carried on already-counted names.
    n_unique_names = sum(u['new_names'] for u in UNIQUE)
    total_named = len(codex_set)
    assert n_literal_in_semantic + n_semantic_nonliteral + n_unique_names == total_named, (
        n_literal_in_semantic, n_semantic_nonliteral, n_unique_names, total_named)

    with_unwired_counterpart = sum(u['new_names'] for u in UNIQUE if u['counterpart_in_our_codebase'])
    no_counterpart_anywhere = sum(u['new_names'] for u in UNIQUE if not u['counterpart_in_our_codebase'])
    all_computable = all(u['computable_from_data_we_already_load'] for u in UNIQUE)

    gate = {f['name']: f['audit_gate_pass'] for f in families}
    n_confirmed = 0  # families that cleared a larger confirmation gate
    submitted_gate_failed = ['contestable_state']

    artefact = {
        'step': 91,
        'author': 'claude',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'kind': 'source-inspection inventory; no model fitted, no experiment run, no held fold or evaluation label read',
        'question': 'Are Codex evidence features a different view of the same hands, or a re-derivation of what our stage 2 already sees?',
        'ours': ours,
        'codex': {
            'families': families,
            'total_distinct_new_column_names': total_named,
            'arm_note': 'Several families emit the same column names under two or three arms; distinct NAMES are counted once.',
            'wired_into_live_evidence_path': {
                'confirmed_and_shipping': n_confirmed,
                'explanation': 'Codex\u2019s live evidence path is our own stage 1 (383 chronology ranker) plus our stage 2 '
                               '(r382DTSP / structCI). Every new Codex feature family was measured as an ADDITION to that '
                               'incumbent and none cleared a confirmation gate.',
                'arms_that_passed_the_small_audit_gate_only': {
                    'contestable_state.contestable_state': '+0.03393 on the 162-pair audit; the larger confirmation then gave '
                                                           '+0.00915 with pool CI [-0.000140, +0.019152] and FAILED',
                    'retrospective_strength.prefix_strength': '+0.02373 on the audit, but the study’s primary arm was '
                                                              'terminal_strength (+0.01781, FAIL); prefix was never carried to a '
                                                              'confirmation run',
                },
                'stage1_only_arm_note': 'every results.json also carries a shared stage1_only baseline arm at -0.05178; it is a '
                                        'reference point, not a feature arm',
                'in_a_submitted_artefact_but_gate_failed': submitted_gate_failed,
                'submitted_artefact': 'Codex Resume/outputs_contestable_exploratory_20260912/submission_contestable_exploratory.csv, '
                                      'public .89457 vs .89447 base, one-off authorised exploratory exception, '
                                      'strict_confirmation_gate_passed=false, larger confirmation +.00915 CI [-.00014, +.01915]',
                'passed_but_not_a_feature': 'TabPFN-3 confirmation (+.0140873, CI [+.0004109, +.0282275]) is a LEARNER swap on '
                                            'the identical 75/453 column route (tabpfn3_confirm_prepare.py asserts that width).',
                'built_but_not_wired': [f['name'] for f in families if f['name'] != 'contestable_state'],
            },
            'audit_gate_summary': gate,
        },
        'overlap': {
            'a_literal_name_overlap': {
                'codex_distinct_names': total_named,
                'our_prod_383': len(prod_set),
                'our_stage2_universe': len(universe_set),
                'exact_matches_vs_prod383': len(literal_vs_prod),
                'exact_matches_vs_universe': len(literal_vs_universe),
                'matched_names': literal_vs_prod,
                'jaccard_vs_prod383': round(jac_prod, 6),
                'jaccard_vs_stage2_universe': round(jac_universe, 6),
            },
            'b_semantic_overlap': {
                'pairs': SEMANTIC,
                'codex_columns_literally_duplicated': n_literal_in_semantic,
                'codex_columns_semantically_overlapping_non_literal': n_semantic_nonliteral,
                'codex_columns_covered_by_a_and_b': n_semantic,
                'share_of_codex_names_covered': round(n_semantic / total_named, 4),
            },
            'c_unique_to_codex': {
                'items': UNIQUE,
                'total_columns_including_repeated_names': n_unique,
                'total_distinct_new_names': n_unique_names,
                'share_of_codex_names_unique': round(n_unique_names / total_named, 4),
                'with_an_unwired_counterpart_in_our_codebase_names': with_unwired_counterpart,
                'with_no_counterpart_anywhere_in_our_codebase_names': no_counterpart_anywhere,
                'all_computable_from_data_we_already_load': all_computable,
                'none_needs_new_data': all_computable,
            },
        },
        'view_types': VIEW_TYPES,
        'view_types_we_lack': [
            'third-party (outsider) card information - zero representation in our 383 and in our 453',
            'side-pot reachability and all-in state at decision time - zero representation',
            'conditional bet-SIZE deviation - zero representation in the wired set (a counterpart exists unwired)',
            'hindsight / terminal-board card strength attached to an earlier action - zero representation',
            'cross-partner action-ORDER interaction terms - zero representation in the wired set (a counterpart exists unwired)',
        ],
        'view_types_both_sides_lack': [
            'prior-hands-only rolling player history (a true cross-hand view). Our *_residual columns use a whole-phase player mean, '
            'not a causal prior window; Codex built no cross-hand feature at all.',
        ],
        'reverse_direction': REVERSE,
        'uncertain': UNCERTAIN,
        'provenance': PROVENANCE,
    }

    # ---------------------------------------------------------------------------- self-check
    checks = {
        'our_382_from_saved_bundle': ours['base382_n'] == 382,
        'our_prod_383': ours['prod_design_n'] == 383,
        'our_blocks_sum_to_382': sum(b['n'] for b in ours['blocks']) == 382,
        'struct_70': ours['stage2']['struct_n'] == 70,
        'shared_75': ours['stage2']['shared_n'] == 75,
        'extra_378': ours['stage2']['extra_n'] == 378,
        'dtsp_453': ours['stage2']['dtsp_width'] == 453,
        'single_struct_collision_is_pot_bb': ours['stage2']['name_collisions_struct_vs_382'] == ['pot_bb'],
        'universe_456': ours['universe_n'] == 456,
        'codex_distinct_names_371': total_named == 371,
        'literal_matches_24': len(literal_vs_prod) == 24 and len(literal_vs_universe) == 24,
        'literal_matches_are_eq': all(c.startswith('eq_') for c in literal_vs_prod),
        'jaccard_prod_rounds_to_0_033': round(jac_prod, 3) == 0.033,
        'partition_covers_every_codex_name': n_literal_in_semantic + n_semantic_nonliteral + n_unique_names == total_named,
        'semantic_covered_244': n_semantic == 244,
        'semantic_non_literal_220': n_semantic_nonliteral == 220,
        'semantic_share_0_66': round(n_semantic / total_named, 2) == 0.66,
        'unique_share_0_34': round(n_unique_names / total_named, 2) == 0.34,
        'unique_names_127': n_unique_names == 127,
        'unique_columns_159': n_unique == 159,
        'no_codex_family_confirmed': n_confirmed == 0,
        'every_unique_item_computable': all_computable,
        'no_counterpart_anywhere_44': no_counterpart_anywhere == 44,
        'unwired_counterpart_83': with_unwired_counterpart == 83,
    }
    artefact['self_check'] = checks
    failed = [k for k, v in checks.items() if not v]
    write_json(OUT / 'feature-overlap-inventory.json', artefact)
    print(json.dumps({'written': str(OUT / 'feature-overlap-inventory.json'),
                      'failed_checks': failed,
                      'headline': {
                          'our_prod': ours['prod_design_n'], 'our_universe': ours['universe_n'],
                          'codex_names': total_named, 'literal': len(literal_vs_prod),
                          'jaccard_prod': round(jac_prod, 6), 'jaccard_universe': round(jac_universe, 6),
                          'semantic_non_literal': n_semantic_nonliteral, 'unique_names': n_unique_names,
                          'unique_columns': n_unique, 'no_counterpart_anywhere': no_counterpart_anywhere}},
                     indent=2))
    if failed:
        raise SystemExit('SELF-CHECK FAILED: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
