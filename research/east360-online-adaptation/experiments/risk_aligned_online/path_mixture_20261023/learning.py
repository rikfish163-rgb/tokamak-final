"""Past-context retrieval and paired mean/multimodal future explanation."""
import numpy as np
import torch

from ..conditional_forecast_20261023.learning import maturity_error


def signal_context(signal, ends, center, scale, active):
    if set(signal) != {'values', 'available'}:
        raise ValueError('only signal fields are allowed')
    x, a = np.asarray(signal['values']), np.asarray(signal['available'])
    ends = np.asarray(ends, int); active = np.asarray(active, bool)
    if (x.ndim != 2 or x.shape[1] != 23 or x.shape != a.shape or a.dtype != bool
            or active.shape != (23,) or not active.any() or not len(ends)
            or ends.min() < 49 or ends.max() >= len(x)):
        raise ValueError('observed flat23 windows with 50 past points required')
    a = a & np.isfinite(x)
    q = np.arcsinh((np.where(a, x, center).astype(float)-center)/scale)[:, active]
    indices = ends[:, None]+np.arange(-49, 1)
    blocks = q[indices]
    h = np.concatenate((q[ends], blocks[:, 25:].mean(1)-blocks[:, :25].mean(1), q[ends]-q[ends-8]), axis=1)
    valid = a[:, active][indices].all((1, 2))
    h[~valid] = np.nan  # Missing context is not an ordinary zero-valued state.
    return h, valid


def fit_context(x, weights):
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) != len(weights):
        raise ValueError('finite historical contexts with matching weights required')
    w = np.asarray(weights, float)
    if (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('nonnegative historical weights required')
    w = w/w.sum(); center = np.einsum('n,nd->d', w, x)
    scale = np.sqrt(np.einsum('n,nd->d', w, (x-center)**2))
    keep = scale > 64*np.finfo(float).eps
    if not keep.any():
        raise ValueError('no variable context dimension')
    z = (x[:, keep]-center[keep])/scale[keep]
    cov = np.einsum('n,ni,nj->ij', w, z, z)
    cov = .9*cov + .1*np.diag(np.diag(cov))
    return dict(center=center, scale=scale, keep=keep, chol=np.linalg.cholesky(cov))


def whiten(x, mapping):
    keep = mapping['keep']
    z = (x[:, keep]-mapping['center'][keep])/mapping['scale'][keep]
    return np.linalg.solve(mapping['chol'], z.T).T


def neighbors(history, query, ids, labels, *, device):
    if (history.ndim != 2 or query.ndim != 2 or history.shape[1] != query.shape[1]
            or len(history) != len(ids) or len(history) != len(labels)
            or not np.isfinite(history).all() or not np.isfinite(query).all()
            or set(np.unique(labels)) != {0, 1}):
        raise ValueError('finite two-class historical/query contexts required')
    ht = torch.as_tensor(history, dtype=torch.float64, device=device)
    qt = torch.as_tensor(query, dtype=torch.float64, device=device)
    result = {}
    for c in (0, 1):
        shotids = np.unique(ids[labels == c]); indices, distances = [], []
        for sid in shotids:
            bank = np.flatnonzero((ids == sid) & (labels == c))
            # Direct distance avoids cancellation for nearly identical states.
            d = torch.cdist(qt, ht[bank], compute_mode='donot_use_mm_for_euclid_dist').square()/history.shape[1]
            distance, nearest = d.min(1)
            indices.append(bank[nearest.cpu().numpy()]); distances.append(distance.cpu().numpy())
        result[c] = (np.stack(indices, 1), np.stack(distances, 1), shotids)
    return result


def mixture_energy(errors):
    if errors.ndim != 2 or errors.shape[1] < 1:
        raise ValueError('query by historical-path energies required')
    valid = np.isfinite(errors).all(1)
    result = np.full(len(errors), np.nan)
    if valid.any():
        e = errors[valid]; smallest = e.min(1)
        result[valid] = smallest-np.log(np.exp(smallest[:, None]-e).mean(1))
    return result


def score_paths(paths, truth, valid, covariance, active):
    if paths.ndim != 4 or paths.shape[0] != truth.shape[0]:
        raise ValueError('one set of direct historical paths per query required')
    n, count, horizon, channels = paths.shape
    observed = np.repeat(truth[:, None], count, axis=1).reshape(n*count, truth.shape[1], channels)
    mask = np.repeat(valid[:, None], count, axis=1).reshape(observed.shape)
    energy = maturity_error(paths.reshape(n*count, horizon, channels), observed, mask, covariance, active).reshape(n, count)
    mean = maturity_error(paths.mean(1), truth, valid, covariance, active)
    return mean, mixture_energy(energy), energy
