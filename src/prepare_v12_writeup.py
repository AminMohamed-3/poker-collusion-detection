"""Prepare a private candidate-specific methodology and verify retained cases."""
import json
import polars as pl
from robust_pair_experiment import ROOT,sha
from validation import EVIDENCE_COLUMNS


def main():
    text=(ROOT/'reports/methodology-v11-risk-draft.md').read_text()
    replacements={
        '**PRIVATE DRAFT — conditional-action risk candidate. Unsubmitted and without a public score. Update to the final selected entry before publication.**':'**PRIVATE DRAFT — conditional risk, evidence and behavior candidate v12. Unsubmitted and without a public score. Update to the final selected entry before publication.**',
        'The risk-only candidate preserves every v10 behavior and evidence cell. Behavior uses the earlier 653-feature classifier\'s most likely known target family.':'Behavior is the most likely supported target family from the same 897-feature classifier, excluding the normal class. There are no positive development labels for the undisclosed fourth family, so the model does not assign it.',
        'The retained evidence model trains':'The evidence model trains',
        'The 316-feature retained ranker uses':'The ranker adds the 33 conditional hand features and their within-pair contexts, giving 382 features. It uses',
        'A separate conditional-evidence candidate is being prepared; it is not part of this risk-only draft.':'Risk, evidence and family changes are saved as separate controlled candidates before combination.',
        'The shorter-history tests retain full phase player-style estimates.':'The original shorter-history tests retain full phase player-style estimates. A further stress test samples whole pool hands at 65% and 40%, rebuilding every baseline and action-style rate from retained observations; all six conditional-risk comparisons improve against v10. Pooled evidence MAP gains are .03551 and .02825, respectively, against the prior ranker. These levels have different retained-annotation denominators and are not directly comparable across fractions.',
        '**Five case reviews.** These previously inspected hands belong to the unchanged v10 evidence lists. Reviews do not modify predictions. Family labels are predictions, not revealed truth.':'**Five case reviews.** These previously inspected hands are retained in the v12 evidence lists, and their predicted families are unchanged. Reviews do not modify predictions. Family labels are predictions, not revealed truth.'}
    for old,new in replacements.items():
        assert text.count(old)==1,old
        text=text.replace(old,new)
    marker='These overlapping splits have been reused during adaptive research;'
    addition='Conditional evidence improves pooled full-history MAP by .03920, with pool-bootstrap interval [.01707,.06143]. Using the conditional classifier for family assignments then improves BehaviorMAP on all three full-history splits by .02378–.03383. All six rebuilt-history composite comparisons also improve, although individual evidence or behavior components sometimes decline. The all-label production models are refitted only after these frozen checks.\n\n'
    assert text.count(marker)==1; text=text.replace(marker,addition+marker)
    destination=ROOT/'reports/methodology-v12-draft.md'; assert not destination.exists(); destination.write_text(text,encoding='utf-8')
    assert len(text.split())<=1500
    path=ROOT/'outputs_v12_behavior/submission_conditional_behavior.csv'; expected=json.loads((path.parent/'ready-audit.json').read_text()); assert sha(path)==expected['sha256']
    source=pl.read_csv(path); cases=[]
    old=pl.read_csv(ROOT/'outputs_v10_tabicl/submission_tabicl.csv')
    for case in json.loads((ROOT/'reports/methodology-v11-risk-validation.json').read_text())['cases']:
        row=source.filter(pl.col('pair_id')==case['pair_id']); evidence=row.select(EVIDENCE_COLUMNS).row(0)
        assert case['hand_id'] in evidence
        family=row['predicted_behavior'][0]; assert family==old.filter(pl.col('pair_id')==case['pair_id'])['predicted_behavior'][0]
        cases.append({'pair_id':case['pair_id'],'hand_id':case['hand_id'],'rank':evidence.index(case['hand_id'])+1,'predicted_behavior':family})
    result={'status':'Private, unsubmitted v12 candidate draft','word_count_whitespace':len(text.split()),'methodology_sha256':sha(destination),'submission_sha256':sha(path),'cases':cases,'five_memberships_and_families_verified':True}
    (ROOT/'reports/methodology-v12-validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
