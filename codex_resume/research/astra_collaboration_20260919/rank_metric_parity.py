import ast,hashlib,json,csv,io
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
P=Path(__file__).resolve().parent
N=P.parent/'official_metric_20260912/slash-poker-competition-metric.ipynb'
j=json.loads(N.read_text(encoding='utf-8')); sources=[''.join(c.get('source',[])) for c in j['cells'] if c.get('cell_type')=='code']
node=next(n for s in sources for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_average_precision')
ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'official_metric_helper','exec'),ns);official=ns['_average_precision']
def local(keys,y,s):
    order=np.lexsort((keys,-s)); hits=y[order]; n=int(hits.sum())
    return float(np.sum((np.cumsum(hits)/np.arange(1,len(hits)+1))*hits)/n) if n else 0.
fixtures=[]
for name,keys,y,s in [('tied_pair',np.array(['p002','p001']),np.array([0,1]),np.array([.5,.5])),('masked_behavior_zeros',np.array(['p001','p002','p003','p004']),np.array([1,0,1,0]),np.array([.9,0.,0.,0.]))]:
    order=np.argsort(keys,kind='stable');a=official(y[order],s[order]);b=float(average_precision_score(y,s));assert local(keys,y,s)==a
    perm=np.random.default_rng(19143).permutation(len(y));assert local(keys[perm],y[perm],s[perm])==a
    fixtures.append({'fixture':name,'official_AP':a,'sklearn_AP':b,'key_aligned_wrapper_exact':True,'row_permutation_invariant':True})
rng=np.random.default_rng(19143);n=1000; rb=np.arange(1,n+1,dtype=float);rp=rng.permutation(rb)
literal=.8*(rb/n)+.2*(rp/n);integer=4*(2*rb).astype(np.int64)+(2*rp).astype(np.int64);canonical=integer/(10*n)
ties=[]
for k in np.unique(integer):
    ix=np.flatnonzero(integer==k)
    if len(ix)>1:ties.append((len(ix),len(np.unique(literal[ix]))))
# Integer half-ranks also cover average ranks for repeated raw probabilities.
b=np.array([.1,.1,.4,.9,.9]);f=np.array([.8,.2,.2,.6,.9]);br=rankdata(b,method='average');fr=rankdata(f,method='average');twob=2*br;twof=2*fr
assert np.array_equal(twob,np.rint(twob)) and np.array_equal(twof,np.rint(twof))
assert np.allclose((4*twob+twof)/(10*len(b)),.8*br/len(b)+.2*fr/len(b),rtol=0,atol=2e-16)
buffer=io.StringIO();writer=csv.writer(buffer);writer.writerow(['risk_score']);writer.writerows([[format(v,'.17g')] for v in canonical]);buffer.seek(0);read=np.array([float(row['risk_score']) for row in csv.DictReader(buffer)]);assert np.array_equal(read,canonical)
r={'utc':datetime.now(timezone.utc).isoformat(),'official_notebook_sha256':hashlib.sha256(N.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fixtures':fixtures,'synthetic_rank_blend':{'n':n,'mathematically_tied_groups':len(ties),'groups_split_by_literal_float_arithmetic':sum(u>1 for _,u in ties),'max_abs_literal_vs_canonical':float(np.max(np.abs(literal-canonical))),'canonical_csv_python_roundtrip_exact':True},'real_labels_or_predictions_read':False,'interpretation':'Official scorer sorts pair IDs before stable AP. sklearn uses threshold-grouped tie AP and is not equivalent. Canonical half-rank integer numerator is algebraically identical .8/.2 and preserves exact rank-sum ties; any change must be preregistered before quality, not chosen from results. No IDs altered, score jitter, tie-order optimization or submission performed.'}
(P/'rank_metric_parity.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r,indent=2))
