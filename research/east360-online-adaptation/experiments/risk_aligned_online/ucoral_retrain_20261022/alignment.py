"""Source recolouring, not target-to-source input correction or unit calibration."""
import numpy as np

KEYS={'shot','device','values','available'}


def signal_rows(rows):
    if not rows:raise ValueError('nonempty signal-only rows required')
    for row in rows:
        if set(row)!=KEYS:raise ValueError('only signal identity, values and availability allowed')
        x=np.asarray(row['values']);m=np.asarray(row['available'])
        if x.ndim!=2 or x.shape[1]!=23 or m.shape!=x.shape or m.dtype!=bool:
            raise ValueError('N by 23 signal and boolean mask required')
    x=np.concatenate([r['values'] for r in rows]).astype(np.float64)
    m=np.concatenate([r['available'] for r in rows]) & np.isfinite(x)
    return x,m


def validate_prefix(rows,order,seals):
    signal_rows(rows)
    if len(rows)!=20 or [r['shot'] for r in rows]!=list(order[:20]):
        raise ValueError('exact chronological first20 required')
    if any(r['device']!='EAST' for r in rows):raise ValueError('target prefix required')
    if set(seals)!={str(s) for s in order[:20]} or any(
            len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in seals.values()):
        raise ValueError('complete prediction hash seals required before fitting')


def fit_statistics(rows):
    x,m=signal_rows(rows);count=m.sum(0);mean=np.zeros(23);scale=np.ones(23);identifiable=np.zeros(23,bool)
    for c in range(23):
        obs=x[m[:,c],c]
        if len(obs):mean[c]=obs.mean()
        if len(obs)>=2:
            sd=np.sqrt(np.mean((obs-mean[c])**2))
            if sd>0 and np.isfinite(sd):scale[c]=sd;identifiable[c]=True
    if not np.isfinite(mean).all():raise FloatingPointError('nonfinite statistics')
    return dict(mean=mean.tolist(),scale=scale.tolist(),count=count.tolist(),
                identifiable=identifiable.tolist(),rows=len(x))


def coordinates(x,available,fit,domain,coral):
    if domain not in ('source','target'):raise ValueError('explicit domain required')
    x=np.asarray(x,np.float64);available=np.asarray(available)
    if x.shape[-1]!=23 or available.shape!=x.shape or available.dtype!=bool:
        raise ValueError('matching flat23 signal/mask required')
    stats=fit[domain];m=available & np.isfinite(x) & np.asarray(fit['active'])
    z=np.where(m,(x-np.asarray(stats['mean']))/np.asarray(stats['scale']),0.)
    if domain=='source' and coral:z=z@np.asarray(fit['matrix'])
    if not np.isfinite(z).all():raise FloatingPointError('nonfinite standardized coordinates')
    return z.astype(np.float32),m


def fit_alignment(source,target):
    if any(r['device']!='J-TEXT' for r in source) or any(r['device']!='EAST' for r in target):
        raise ValueError('explicit source/target signal collections required')
    ss,ts=fit_statistics(source),fit_statistics(target)
    active=np.asarray(ss['identifiable']) & np.asarray(ts['identifiable'])
    if not active.any():raise ValueError('no jointly varying observed channels')
    fit=dict(source=ss,target=ts,active=active.tolist())
    # Covariance uses full-precision standardized values, with missingness
    # mean-imputed. Do not cast to float32 before the eigendecomposition.
    matrices=[]
    for rows,stat in ((source,ss),(target,ts)):
        x,m=signal_rows(rows);m &= active
        z=np.where(m,(x-np.asarray(stat['mean']))/np.asarray(stat['scale']),0.)
        matrices.append(np.cov(z,rowvar=False,ddof=1))
    cs,ct=matrices
    def power(c,p):
        eig,u=np.linalg.eigh(c+np.eye(23))
        if not np.isfinite(eig).all() or (eig<=0).any():raise FloatingPointError('regularized covariance not SPD')
        return (u*eig**p)@u.T
    a=power(cs,-.5)@power(ct,.5)
    # Explicitly isolate inactive columns: no source-only observation can
    # leak through a numerically tiny off-diagonal coefficient.
    a[~active,:]=0.;a[:,~active]=0.;a[~active,~active]=1.
    error=float(np.max(abs(a.T@(cs+np.eye(23))@a-(ct+np.eye(23)))))
    if error>1e-9:raise AssertionError('source-to-target covariance direction failed')
    fit.update(matrix=a.tolist(),source_cov=cs.tolist(),target_cov=ct.tolist(),ridge=1.,
               regularized_identity_max_error=error,condition_number=float(np.linalg.cond(a)))
    return fit
