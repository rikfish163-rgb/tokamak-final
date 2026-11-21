"""Joint-time, equal-shot reverse-CORAL diagnostic for a frozen classifier.

This is NOT a reproduction of source-to-target CORAL plus classifier training,
S-CORAL, Deep CORAL or TTT++. Work in the source model's dimensionless asinh
coordinates. Mixing array summary statistics is an engineering control, NOT a
certified physical diagnostic calibration. Blocks do not change the network.
"""
import numpy as np

from ..io23 import RAW23
from ..warmstart_alignment_20261022.alignment import SIGNAL_KEYS, _arrays

BLOCKS = {'sxr': ('sxr_kurt', 'sxr_mean', 'sxr_skew', 'sxr_var'),
          'xuv': ('xuv_kurt', 'xuv_ratio', 'xuv_skew', 'xuv_var')}
POLICIES = ('mean', 'diagonal', 'coral', 'within_coral')


def coordinates(center, scale):
    c, s = np.asarray(center, float), np.asarray(scale, float)
    if c.shape != (23,) or s.shape != (23,) or not np.isfinite([c, s]).all() or (s <= 0).any():
        raise ValueError('finite locked source center and positive scale required')
    return c, s


def summarize(record, *, center, scale):
    if set(record) != SIGNAL_KEYS or record['device'] not in ('J-TEXT', 'EAST'):
        raise ValueError('authorized signal fields only; no outcome or event fields')
    if type(record['shot']) is not int: raise ValueError('integer shot required')
    c, s = coordinates(center, scale)
    x, m = _arrays(record)
    result = dict(shot=record['shot'], device=record['device'], blocks={})
    for name, features in BLOCKS.items():
        cols = [RAW23.index(f) for f in features]
        valid = np.flatnonzero(m[:, cols].all(1))
        indices = valid[np.linspace(0, len(valid)-1, min(128, len(valid)), dtype=int)] if len(valid) else valid
        block = dict(observed_rows=int(len(valid)), total_rows=len(x), sample_rows=len(indices),
                     mean=None, within_cov=None)
        if len(indices):
            z = np.arcsinh((x[np.ix_(indices, cols)].astype(float)-c[cols])/s[cols])
            mean = z.mean(0); centered = z-mean
            block.update(mean=mean.tolist(), within_cov=(centered.T @ centered/len(z)).tolist())
        result['blocks'][name] = block
    return result


def combine(summaries, *, device, allowed_ids):
    rows = list(summaries)
    ids = [r['shot'] for r in rows]
    if (not rows or len(ids) != len(set(ids)) or not set(ids) <= set(allowed_ids)
            or any(r['device'] != device for r in rows)):
        raise ValueError('distinct authorized device summaries required')
    result = dict(device=device, shot_ids=ids, blocks={})
    for name in BLOCKS:
        obs = [r['blocks'][name] for r in rows if r['blocks'][name]['sample_rows']]
        out = dict(observed_shots=len(obs), sample_rows=sum(r['sample_rows'] for r in obs),
                   mean=None, within_cov=None, between_cov=None, total_cov=None)
        if obs:
            means = np.array([r['mean'] for r in obs]); mean = means.mean(0)
            within = np.mean([r['within_cov'] for r in obs], axis=0)
            between = (means-mean).T @ (means-mean)/len(obs)
            out.update(mean=mean.tolist(), within_cov=within.tolist(),
                       between_cov=between.tolist(), total_cov=(within+between).tolist())
        result['blocks'][name] = out
    return result


def _root(matrix, power):
    vals, vecs = np.linalg.eigh(matrix)
    if vals.min() <= 0 or not np.isfinite(vals).all():
        raise ValueError('strictly positive regularized covariance required')
    return (vecs * (vals ** power)) @ vecs.T


def make_block_map(source, target, policy):
    if policy not in POLICIES: raise ValueError('explicit covariance policy required')
    if not source['observed_shots'] or not target['observed_shots']:
        return dict(status='unobserved_block_identity', policy=policy)
    ms, mt = np.array(source['mean']), np.array(target['mean'])
    dim = len(ms)
    if mt.shape != ms.shape: raise ValueError('source-target dimensions differ')
    key = 'within_cov' if policy == 'within_coral' else 'total_cov'
    cs, ct = np.asarray(source[key]), np.asarray(target[key])
    if not np.isfinite([cs, ct]).all(): raise ValueError('finite covariance required')
    out = dict(status='mapped', policy=policy, source_mean=ms.tolist(), target_mean=mt.tolist(),
               covariance_object=key, correlation_ridge=1. if 'coral' in policy else 0.)
    if policy == 'mean':
        a = np.eye(dim)
    else:
        if (np.diag(cs) <= 0).any() or (np.diag(ct) <= 0).any():
            # A genuinely constant direction has no identifiable scale. The
            # whole block is identity, explicitly logged, with no fabricated
            # epsilon, altered mask, or partial dense mapping.
            return dict(status='unidentifiable_constant_block_identity', policy=policy)
        ss, st = np.sqrt(np.diag(cs)), np.sqrt(np.diag(ct))
        rs, rt = cs/np.outer(ss, ss), ct/np.outer(st, st)
        if np.linalg.eigvalsh(rs).min() < -1e-9 or np.linalg.eigvalsh(rt).min() < -1e-9:
            raise ValueError('joint-time covariance must be positive semidefinite')
        if policy == 'diagonal':
            a = np.diag(ss/st)
        else:
            # CORAL author's ridge=1, after domain standardization. This is
            # an identity ridge, not an estimated Ledoit-Wolf shrinkage rule.
            sr, tr = rs+np.eye(dim), rt+np.eye(dim)
            b = _root(tr, -.5) @ _root(sr, .5)
            a = np.diag(1/st) @ b @ np.diag(ss)
            out.update(source_eigenvalues=np.linalg.eigvalsh(rs).tolist(),
                       target_eigenvalues=np.linalg.eigvalsh(rt).tolist(),
                       target_condition_regularized=float(np.linalg.cond(tr)),
                       regularized_residual_relative=float(np.linalg.norm(b.T@tr@b-sr)/np.linalg.norm(sr)))
    out.update(matrix=a.tolist(), singular_values=np.linalg.svd(a, compute_uv=False).tolist(),
               unregularized_covariance_relative=float(np.linalg.norm(a.T@ct@a-cs)/max(np.linalg.norm(cs), 1e-30)))
    return out


def fit_maps(source, summaries, *, order, count, policy):
    rows = list(summaries)
    if (type(count) is not int or not 0 < count < len(order) or len(order) != len(set(order))
            or [r['shot'] for r in rows] != list(order[:count])):
        raise ValueError('only the exact ended prefix is authorized')
    if source['device'] != 'J-TEXT': raise ValueError('source TRAIN statistics required')
    target = combine(rows, device='EAST', allowed_ids=order[:count])
    return dict(policy=policy, calibration_ids=target['shot_ids'], first_affected_index=count,
                coordinate='unchanged source asinh coordinates', target_statistics=target,
                blocks={k:make_block_map(source['blocks'][k], target['blocks'][k], policy) for k in BLOCKS})


def transform(signal, mapping, *, center, scale):
    if set(signal) != {'values', 'available'}: raise ValueError('signals only')
    c, s = coordinates(center, scale); x, m = _arrays(signal)
    out = x.copy(); audit = {}
    for name, features in BLOCKS.items():
        cols = [RAW23.index(f) for f in features]
        valid = np.flatnonzero(m[:, cols].all(1)); spec = mapping['blocks'][name]
        audit[name] = dict(complete_rows=len(valid), incomplete_rows_identity=len(x)-len(valid),
                           status=spec['status'], mapped_rows=0, negative_variance_rows=0,
                           input_negative_variance_rows=int((x[valid, RAW23.index(name+'_var')] < 0).sum()))
        if spec['status'] != 'mapped' or not len(valid): continue
        z = np.arcsinh((x[np.ix_(valid, cols)].astype(float)-c[cols])/s[cols])
        mapped = (z-np.array(spec['target_mean'])) @ np.array(spec['matrix']) + np.array(spec['source_mean'])
        with np.errstate(over='raise', invalid='raise'):
            raw = np.sinh(mapped)*s[cols]+c[cols]
        if not np.isfinite(raw).all() or (abs(raw) > np.finfo(np.float32).max).any():
            raise FloatingPointError('mapped block not representable; no clipping allowed')
        out[np.ix_(valid, cols)] = raw.astype(np.float32)
        audit[name]['mapped_rows'] = len(valid)
        # A negative inverse-mapped variance proves this is no certified
        # physical calibration. Report it; do NOT silently clip it away.
        audit[name]['negative_variance_rows'] = int((raw[:, features.index(name+'_var')] < 0).sum())
    return dict(values=out, available=m.copy()), audit
