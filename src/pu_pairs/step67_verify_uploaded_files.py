import hashlib, polars as pl
from pathlib import Path
CV = Path(r'C:\Data\Projects\Money Making\work\poker\Claude Version')
F = {
 'S': CV/'outputs_blend_20260911/submission_blend_w0.5_plus_R1.csv',
 'base': CV/'outputs_chronology_20260911/submission_chronology_frac.csv',
 'M3': CV/'outputs_multiview_20260911/submission_multiview_M3_lengthviews.csv',
 'M4': CV/'outputs_multiview_20260911/submission_multiview_M4_evidenceview.csv',
 'M3s2c3': CV/'outputs_stage2_20260911/submission_M3_stage2_chrono3.csv',
 'M4s2c3': CV/'outputs_multiview_20260911/submission_M4_stage2_chrono3.csv',
 'M3r382': CV/'outputs_stage2_r382_20260911/submission_M3_stage2_r382DTSP.csv',
 'Ss2c3': CV/'outputs_stage2_20260911/submission_S_stage2_chrono3.csv',
 'Sr382': CV/'outputs_stage2_r382_20260911/submission_S_stage2_r382DTSP.csv',
}
D = {}
for k, p in F.items():
    D[k] = pl.read_csv(p)
    print(k, hashlib.sha256(open(p,'rb').read()).hexdigest()[:16], D[k].shape, D[k].columns if k=='S' else '')
ev = [c for c in D['S'].columns if c.startswith('evidence')]
bc = [c for c in D['S'].columns if 'behav' in c]
print('evidence cols', ev, 'behaviour cols', bc)
def same(a, b, cols):
    x = D[a].sort('pair_id'); y = D[b].sort('pair_id')
    return all((x[c] == y[c]).all() if x[c].dtype != pl.Float64 else (x[c] - y[c]).abs().max() == 0 for c in cols)
pairs = [('S','base'),('M3','base'),('M3s2c3','M3'),('M4s2c3','M4'),('M3r382','M3'),('M3r382','M3s2c3'),('M4s2c3','M3s2c3'),('Ss2c3','S'),('Sr382','S'),('Sr382','M3r382'),('Ss2c3','M3s2c3')]
for a, b in pairs:
    print(f'{a:8s} vs {b:8s}  risk_same={same(a,b,["risk_score"])}  evidence_same={same(a,b,ev)}  behaviour_same={same(a,b,bc)}')
# how many evidence rows differ between chrono3 and r382DTSP
x = D['M3s2c3'].sort('pair_id'); y = D['M3r382'].sort('pair_id')
diff = sum(((x[c] != y[c]).any() for c in ev))
rows = (pl.concat([ (x[c] != y[c]).to_frame(c) for c in ev], how='horizontal').select(pl.any_horizontal(pl.all())).to_series()).sum()
print('rows with any evidence difference chrono3 vs r382DTSP:', rows, 'of', x.height)
blk = (x['risk_score'] >= .5).sum()
print('rows with risk>=.5 in M3:', blk)
