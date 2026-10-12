import polars as pl
import numpy as np

VD = 'Claude Version/experiments/viewdis_20260911'
VIEWS = ['FULL','RAW','CA','BASE','OPP','V2','MEAN','TAILS']

def lg(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1-1e-6)
    return np.log(x/(1-x))

def logreg(X, y, l2=10.0, iters=200):
    mu = X.mean(0); sd = X.std(0)+1e-9; Z = (X-mu)/sd
    n, k = Z.shape; w = np.zeros(k); b = 0.0
    for _ in range(iters):
        p = 1/(1+np.exp(-(Z@w+b)))
        g = Z.T@(p-y)+l2*w; gb = (p-y).sum()
        W = p*(1-p); H = (Z*W[:,None]).T@Z + l2*np.eye(k); Hb = W.sum()
        step = np.linalg.solve(H, g); w -= step; b -= gb/max(Hb,1e-9)
        if np.abs(step).max() < 1e-8:
            break
    return w, b, mu, sd

def X_of(df, is_eval):
    cols = []
    for v in VIEWS:
        col = 'risk' if (v=='FULL' and is_eval) else ('risk_prefix' if v=='FULL' else f'r_{v}')
        cols.append(lg(df[col].to_numpy()))
    cols.append(np.log(df['shared_hands'].to_numpy().astype(float)))
    return np.column_stack(cols)

mp = pl.read_parquet(f'{VD}/multiview_prefix.parquet')
posset = ['pos','undisclosed']; negset = ['FP','undisclosed_R1']
c = mp.filter((pl.col('risk_prefix')>=0.5)&(pl.col('shared_hands')>=38)&(pl.col('cls').is_in(posset+negset)))
print('eligible defA n=', c.height, 'pos=', c.filter(pl.col('cls').is_in(posset)).height)
y = c['cls'].is_in(posset).to_numpy().astype(float)
X = X_of(c, False)
w,b,mu,sd = logreg(X, y, l2=10.0)
print('M weights:', dict(zip(VIEWS+['sh'], [round(float(v),3) for v in w])), 'b=', round(float(b),3))

ev = pl.read_parquet(f'{VD}/multiview_eval.parquet')
blk = ev.filter(pl.col('risk')>=0.5)
print('eval block n=', blk.height)
Xe = X_of(blk, True)
Ze = (Xe-mu)/sd
logit = Ze@w+b
p = 1/(1+np.exp(-logit))
F = float((1-p).sum())
print('F=sum(1-p)=%.1f mean_p=%.3f median_p=%.3f' % (F, p.mean(), np.median(p)))
print('p quantiles:', [round(float(np.quantile(p,q)),3) for q in [0,0.1,0.25,0.5,0.75,0.9,1.0]])

s = lg(blk['r_FULL'].to_numpy())+0.5*lg(blk['r_RAW'].to_numpy())
r1 = blk['R1'].to_numpy().astype(bool)
s = np.where(r1, s-100, s)
order = np.argsort(-s, kind='stable')
n = len(blk)
thirds = [order[:n//3], order[n//3:2*n//3], order[2*n//3:]]
for i, idx in enumerate(thirds):
    print('third %d n=%d F_third=%.1f mean_p=%.3f mean_sh=%.1f median_sh=%.1f' % (i+1, len(idx), (1-p[idx]).sum(), p[idx].mean(), blk['shared_hands'].to_numpy()[idx].mean(), np.median(blk['shared_hands'].to_numpy()[idx])))

Zt = (X-mu)/sd
for lab, mask in [('testbed_pos', y==1), ('testbed_neg', y==0)]:
    lsub = Zt[mask]@w+b
    print(lab, 'logit mean %.3f median %.3f sd %.3f' % (lsub.mean(), np.median(lsub), lsub.std()))
print('eval logit mean %.3f median %.3f sd %.3f' % (logit.mean(), np.median(logit), logit.std()))

try:
    from sklearn.mixture import GaussianMixture
    print('sklearn available')
    gmm = GaussianMixture(n_components=2, random_state=2901).fit(Ze)
    print('gmm weights:', gmm.weights_.tolist())
    for k in range(2):
        print('comp%d w=%.3f mean=%s' % (k, gmm.weights_[k], np.round(gmm.means_[k],2).tolist()))
    print('testbed pos mean:', np.round(Zt[y==1].mean(0),2).tolist())
    print('testbed neg mean:', np.round(Zt[y==0].mean(0),2).tolist())
    print('testbed pos n=%d neg n=%d' % (int((y==1).sum()), int((y==0).sum())))
except Exception as e:
    print('GMM fail:', repr(e))

try:
    Zpos = Zt[y==1]; Zneg = Zt[y==0]
    for cname, m in [('pos', Zpos), ('neg', Zneg)]:
        mu_c = m.mean(0); cov = np.cov(m, rowvar=False)+np.eye(m.shape[1])*1e-6
        inv = np.linalg.inv(cov)
        d_test = np.einsum('ij,jk,ik->i', m-mu_c, inv, m-mu_c)
        thr = float(np.quantile(d_test, 0.99))
        d_eval = np.einsum('ij,jk,ik->i', Ze-mu_c, inv, Ze-mu_c)
        print('Mahalanobis vs %s: test99=%.2f eval_max=%.2f eval_med=%.2f n_exceed=%.0f' % (cname, thr, d_eval.max(), np.median(d_eval), (d_eval>thr).sum()))
    mu_p = Zpos.mean(0); mu_n = Zneg.mean(0)
    cov_p = np.cov(Zpos, rowvar=False)+np.eye(Zt.shape[1])*1e-6
    cov_n = np.cov(Zneg, rowvar=False)+np.eye(Zt.shape[1])*1e-6
    inv_p = np.linalg.inv(cov_p); inv_n = np.linalg.inv(cov_n)
    d99_p = float(np.quantile(np.einsum('ij,jk,ik->i', Zpos-mu_p, inv_p, Zpos-mu_p),0.99))
    d99_n = float(np.quantile(np.einsum('ij,jk,ik->i', Zneg-mu_n, inv_n, Zneg-mu_n),0.99))
    d_ep = np.einsum('ij,jk,ik->i', Ze-mu_p, inv_p, Ze-mu_p)
    d_en = np.einsum('ij,jk,ik->i', Ze-mu_n, inv_n, Ze-mu_n)
    both = int(((d_ep>d99_p)&(d_en>d99_n)).sum())
    print('both-exceed-99th: %d/%d' % (both, len(Ze)))
    if both>0:
        idx = np.where((d_ep>d99_p)&(d_en>d99_n))[0]
        sh = blk['shared_hands'].to_numpy()[idx]
        print('outlier sh min/med/max: %d/%.1f/%d' % (int(sh.min()), float(np.median(sh)), int(sh.max())))
        print('outlier p mean %.3f logit mean %.3f' % (float(p[idx].mean()), float(logit[idx].mean())))
        # describe by table size if available? multiview_eval has table_id only; try seats? skip, report view profile
        print('outlier view means (std-logit space):', np.round(Ze[idx].mean(0),2).tolist())
except Exception as e:
    print('Mah fail:', repr(e))
print('DONE')
