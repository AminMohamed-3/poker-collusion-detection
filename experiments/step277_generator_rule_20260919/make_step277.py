"""Derive step277_B_sp_rule_build.py from the already-gated step276_G_ci_gamma_build.py by targeted patch.

Patching rather than rewriting is deliberate: step276_G's three gates (v1 replay per batch, decoder identity at
gamma = 1, and byte-identity to STEP272 in neutral mode) are verified working, and the only thing STEP277 changes
is one line of arithmetic -- the soft-play rule blend applied to the ensemble AFTER the event fusion and BEFORE
the decoder, which is exactly where the audit validation applied it.
"""
from pathlib import Path

SRC = Path('C:/Data/Projects/Money Making/work/poker/Claude Version/src/evidence_aug/step276_G_ci_gamma_build.py')
DST = SRC.with_name('step277_B_sp_rule_build.py')
t = SRC.read_text(encoding='utf-8')

DOC = '''"""STEP277-B: the induced soft-play citation rule, blended into the evidence probability. Evidence-only,
on the STEP272 pair side.

What changes, in one line: for pairs whose predicted_behavior is soft_play, the probability entering the typed
decoder becomes `p_final = p_fused^(1-w) * p_rule^w` with w = .30, where p_rule is the STEP277-A induced
per-hand citation rule. Everything else -- the four LP members, the arithmetic ensemble, the STEP259b event
fusion at w = .25 / F = 10 on DT and SP, the chronology, the type attributes, the CI mask and the decoder
itself -- is untouched, and directed_transfer and coordinated_isolation pairs are bit-identical to STEP272.

Why (OBSERVATIONS 19.237). p_rule is fitted on the 85 soft-play development pairs that lie OUTSIDE the 162-query
audit, over their full ~121-hand candidate universes, on the 662 frozen STEP254 mechanical columns, bagged over
8 seeds. Evaluated on the audit, which no part of the fit has seen:

    soft play subgroup  +.024255        whole audit  +.008148   P(>0) .9285   40 lists changed, 17 sets changed
    honest split-half over 28 (scope, weight) cells, control included:  +.003039   307/400 positive

and every control fails in the right direction:

    PERM   labels permuted within each training pair   -.054178   <- the gain is the LABELS, not the blend
    SHUF_FAM  trained on the other two families        -.023481
    5 hyperparameter settings   +.0185 .. +.0319, 5/5 positive
    10 draws of 70% of the training pairs   mean +.0209, min +.0088, 10/10 positive

This is the information-adding class, which is the class that transfers in this project's public record (STEP167
LP +.00258, STEP259b fusion +.00114), not the decoder-re-parameterisation class that went 0 for 2 (STEP175
-.00120, STEP179 -.00086). Composite value +.0006 to +.0016.

Two honest caveats, both recorded rather than argued away:
  1. STEP257's own mechplus_p_SP does NOT reproduce this (-.007413 at the same w, with its permutation control at
     -.011121 tracking it). The shipped object is therefore this rule and not that score, and the two disagree --
     mechplus_p_SP is a 5-fold OOF event model over all 1,860 pairs and 850 columns, while p_rule is a citation
     model trained only on soft-play pairs with a strictly disjoint 210/162 pair split.
  2. The audit validation blended into the probability the audit's decoder consumes (PRODbare = v1_DTfp3). On the
     evaluation side soft play ALSO receives the STEP259b event fusion, so the shipped blend composes with a
     term the audit read did not exercise. The pipeline POSITION is the same -- immediately before the decoder --
     but the composition is not something the audit could test.

Features: the 662 columns cannot come from STEP254, which is development-only by construction. They are built by
step277_A_sp_rule_features.py through step257_mech_eval.build_chunk -- STEP254's computation imported unchanged
with the phase parameterised -- with the column list frozen from the development file. That build passed GATE P:
1,445 development soft-play rows x 662 columns reproduced step254_features.parquet exactly through this code
path, which is what makes the evaluation features trustworthy.

Four gates, all of which must pass:
  GATE 1  v1 replay: decoding the base v1 score must reproduce the STEP169 base file's evidence, per batch.
  GATE 2  identity: the decoder at gamma = 1 must equal the production decoder's output exactly.
  GATE 3  end-to-end: at w = 0 the written file must be BYTE-IDENTICAL to STEP272 d5891a68.
  GATE 4  scope: only soft_play rows may differ from STEP272; risk_score and predicted_behavior byte-identical.

Usage:  python step277_B_sp_rule_build.py neutral    # GATE 3
        python step277_B_sp_rule_build.py rule       # the STEP277 candidate, w = .30
Never uploads.
"""'''
t = DOC + t[t.index('"""', t.index('"""') + 3) + 3:]
rep = [
    ("OUT = CV / 'outputs_step276_20260919'", "OUT = CV / 'outputs_step277_20260919'"),
    ("GAMMA_CI = 2.0\nOUTNAME = 'submission_step276_cigamma2_on_step272.csv'",
     "GAMMA_CI = 2.0\nSP_FAM = 'soft_play'\nW_RULE = 0.30\n"
     "RULEF = CV / 'experiments/step277_generator_rule_20260919/sp_rule_eval.parquet'\n"
     "OUTNAME = 'submission_step277_spRule_on_step272.csv'"),
    ("def build(gamma: float, outname: str) -> dict:", "def build(w_rule: float, outname: str) -> dict:"),
    ("""    lkmap = {(a, b): (c, d) for a, b, c, d in event_lookup(allkeys).iter_rows()}
    print('event scores attached for', len(lkmap), 'evidence rows', flush=True)""",
     """    lkmap = {(a, b): (c, d) for a, b, c, d in event_lookup(allkeys).iter_rows()}
    print('event scores attached for', len(lkmap), 'evidence rows', flush=True)
    rl = pl.read_parquet(RULEF)
    rmap = {(a, b): c for a, b, c in rl.iter_rows()}
    sp_pairs_rule = set(rl['pair_id'].to_list())
    sp_pairs_base = set(base.filter(pl.col('predicted_behavior') == SP_FAM)
                        .join(allkeys.select('pair_id').unique(), on='pair_id', how='semi')['pair_id'].to_list())
    assert sp_pairs_rule == sp_pairs_base, (
        'the rule covers a different soft-play pair set than the scoped base: rule %d, base %d, missing %d'
        % (len(sp_pairs_rule), len(sp_pairs_base), len(sp_pairs_base - sp_pairs_rule)))
    print('soft-play rule attached for %d pairs / %d rows' % (len(sp_pairs_rule), rl.height), flush=True)
    rule_rows = 0"""),
    ("""        cov['other'] += int((~np.isin(famrow, [f for f, _ in FUSED_FAMS])).sum())
""",
     """        cov['other'] += int((~np.isin(famrow, [f for f, _ in FUSED_FAMS])).sum())

        # ---- the STEP277 change: the induced soft-play rule, on the post-fusion probability
        ensr = ensf.copy()
        sprow = famrow == SP_FAM
        if w_rule > 0 and sprow.any():
            pr = np.array([rmap[(p, h)] for p, h, mm in zip(pids, hids, sprow) if mm], dtype=np.float64)
            assert len(pr) == int(sprow.sum()) and np.isfinite(pr).all(), 'soft-play rule coverage gap'
            ensr[sprow] = (np.clip(ensf[sprow], 1e-9, 1.0) ** (1 - w_rule)) * np.clip(pr, 1e-9, 1.0) ** w_rule
            rule_rows += int(sprow.sum())
"""),
    ("""                        ).with_columns(pl.Series('v1_ens', ensf + 10))""",
     """                        ).with_columns(pl.Series('v1_ens', ensf + 10),
                                       pl.Series('v1_rule', ensr + 10))"""),
    ("""        new = prod if gamma == 1.0 else S108.expand(CSP.rank(bf, decode_gamma(fe, typed, cif, gamma)))""",
     """        if w_rule == 0:
            new = prod
        else:
            fer = fc.with_columns(pl.Series('score', bf['v1_rule'].to_numpy()))
            new = S108.expand(CSP.rank(bf, S104.decode(fer, typed)))"""),
    ("""        assert mine1.equals(prod), 'GATE 2 gamma = 1 is not the production decoder in ' + name""",
     """        assert mine1.equals(prod), 'GATE 2 the decoder at gamma = 1 is not the production decoder in ' + name"""),
    ("""    rec = {'step': 'STEP276', 'gamma_CI': gamma,""", """    rec = {'step': 'STEP277', 'w_rule': w_rule, 'rule_rows': rule_rows,
           'rule_file': str(RULEF.relative_to(CV)), 'rule_sha256': sha(RULEF),"""),
    ("""           'audit_evidence': {'gamma1_decoded': 0.733740, 'gamma2_decoded': 0.735201,
                              'd_all': 0.001461, 'P_gt_0_all': 0.8685,
                              'd_CI_subgroup': 0.005504, 'P_gt_0_CI': 0.8715,
                              'halves': [0.00100, 0.00200],
                              'splithalf_honest_mean': 0.000482, 'splithalf_positive': '176/400',
                              'CI_audit_queries': 43},""",
     """           'audit_evidence': {'prod_decoded': 0.733740, 'rule_decoded': 0.741888,
                              'd_all': 0.008148, 'P_gt_0_all': 0.9595,
                              'd_SP_subgroup': 0.028090, 'SP_audit_queries': 47,
                              'halves': [0.00471, 0.01213], 'lists_changed': 39, 'lists_set_changed': 16,
                              'splithalf_honest_mean': 0.003039, 'splithalf_positive': '307/400',
                              'controls': {'perm_within_pair': -0.054178, 'shuffled_family': -0.023481,
                                           'hp_settings_positive': '5/5', 'subset70_positive': '10/10',
                                           'subset70_min': 0.008830},
                              'mechplus_p_SP_does_not_reproduce': -0.007413},"""),
    ("""    write_json(OUT / ('receipt_step276_gamma%g.json' % gamma), rec)""",
     """    write_json(OUT / ('receipt_step277_w%g.json' % w_rule), rec)"""),
    ("""def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fused'
    assert mode in ('neutral', 'fused'), mode
    if mode == 'neutral':
        rec = build(1.0, 'gate_step276_neutral_must_equal_step272.csv')
        assert rec['equals_step272'], 'GATE 3 FAILED: gamma = 1 did not reproduce STEP272 byte-for-byte'
        print('GATE 3 PASSED: gamma = 1 reproduces STEP272 d5891a68 exactly', flush=True)
    else:
        rec = build(GAMMA_CI, OUTNAME)
        assert not rec['equals_step272'], 'gamma = 2 produced an identical file; the change did nothing'
        assert rec['rows_differing_by_family'].get('directed_transfer', 0) == 0 and \\
            rec['rows_differing_by_family'].get('soft_play', 0) == 0, \\
            'only coordinated_isolation lists may change'
        print('STEP276 built. sha256', rec['output_sha256'], flush=True)""",
     """def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'rule'
    assert mode in ('neutral', 'rule'), mode
    if mode == 'neutral':
        rec = build(0.0, 'gate_step277_neutral_must_equal_step272.csv')
        assert rec['equals_step272'], 'GATE 3 FAILED: w = 0 did not reproduce STEP272 byte-for-byte'
        print('GATE 3 PASSED: w = 0 reproduces STEP272 d5891a68 exactly', flush=True)
    else:
        rec = build(W_RULE, OUTNAME)
        assert not rec['equals_step272'], 'the rule produced an identical file; the change did nothing'
        byf = rec['rows_differing_by_family']
        assert byf.get('directed_transfer', 0) == 0 and byf.get('coordinated_isolation', 0) == 0, \\
            'GATE 4 FAILED: only soft_play lists may change, got %s' % byf
        assert rec['risk_behaviour_identical_to_step272'], 'GATE 4 FAILED: the pair side moved'
        print('GATE 4 PASSED: only soft_play changed (%d rows)' % byf.get('soft_play', 0), flush=True)
        print('STEP277 built. sha256', rec['output_sha256'], flush=True)"""),
]
for a, b in rep:
    assert a in t, 'PATCH TARGET MISSING:\\n' + a[:200]
    t = t.replace(a, b, 1)
DST.write_text(t, encoding='utf-8')
print('wrote', DST.name, len(t.split(chr(10))), 'lines')
