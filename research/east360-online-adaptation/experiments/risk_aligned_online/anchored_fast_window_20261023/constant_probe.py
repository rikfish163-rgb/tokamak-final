"""Signal-only diagnostic: remove variation from the auxiliary learner only.

The real prediction tokens remain unchanged. Constant auxiliary inputs retain
the observed per-channel mean in source-asinh coordinates and the exact mask.
This intervention cannot distinguish position from level or mask dependence.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .run import (READ, sha, io23, make_parent, models_for_calibration, REFERENCE,
    read_signal, transform, state_cpu, assert_unchanged, DEFAULT_PARENT,
    PARENT_SHA256, _durable_json)
from .model import AnchoredAdapter, centered_step
from .reverse_probe import match_write_norm
from ..matured_negative_20261023.learning import tensors


def constant_observed_window(values, available, center, scale):
    if values.ndim != 3 or values.shape != available.shape or available.dtype != torch.bool:
        raise ValueError('matching windows and boolean availability required')
    if not torch.all(torch.isfinite(scale) & (scale > 0)):
        raise ValueError('finite positive source scales required')
    mask = available & torch.isfinite(values)
    z = torch.asinh((torch.where(mask, values, center) - center) / scale)
    average = z.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
    # A wholly missing channel has average zero and maps to the source center;
    # its original false mask is retained by the caller, never imputed as valid.
    return (center + scale * average.sinh()).expand_as(values).clone()


@torch.no_grad()
def main(root):
    target = root / 'constant_probe.json'
    if target.exists() or (root / 'constant_probe.npz').exists():
        raise FileExistsError(target)
    p = READ(root / 'protocol.json')
    verified = READ(root / 'verification.json')
    assert verified['completed_sha256'] == sha(root / 'completed.json')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    base, _ = make_parent()
    parent = models_for_calibration(base)['mapped_fixed']
    net = AnchoredAdapter(copy.deepcopy(parent.base), rule='normalized').cuda().eval()
    net.layer.load_state_dict(torch.load(root / 'layers.pt', map_location='cpu', weights_only=True)['normalized'])
    snapshot = state_cpu(net)
    mapping = READ(REFERENCE / 'input_map.json')
    collected = []

    def deny(*args, **kwargs):
        raise AssertionError('signal-only probe may not read outcomes')

    with patch.object(io23, 'read_outcome', side_effect=deny):
        for index, sid in enumerate(p['shot_ids']):
            s = read_signal(sid)
            signal = transform({k: s[k] for k in ('values', 'available')}, mapping)
            ends = np.arange(127, len(s['times']), 128)
            for lo in range(0, len(ends), 32):
                use = ends[lo:lo+32]
                x, m = tensors(signal, use, 'cuda')
                h, valid = net.tokens(x, m)
                c = constant_observed_window(x, m, net.base.center, net.base.scale)
                constant_h, constant_valid = net.tokens(c, m)
                torch.testing.assert_close(constant_valid, valid, atol=0, rtol=0)
                q, k, v = net.layer.coordinates(h)
                _, ck, cv = net.layer.coordinates(constant_h)
                observed = valid.clone()
                observed[:, -1] = False
                a, _ = centered_step(k, v, observed, net.layer.origin, net.layer.eta,
                    rule='normalized', reverse=False)
                cd, _ = centered_step(ck, cv, observed, net.layer.origin, net.layer.eta,
                    rule='normalized', reverse=False)
                matched = match_write_norm(cd, a)

                def score(delta):
                    hh = h + net.layer.output(q @ delta) * valid[..., None]
                    return net.from_tokens(hh, valid, mode='bypass')[0].sigmoid()

                actual = score(a)
                exact = net.from_tokens(h, valid, mode='on')[0].sigmoid()
                torch.testing.assert_close(actual, exact, atol=0, rtol=0)
                an = a.square().sum((1, 2)).sqrt()
                cn = cd.square().sum((1, 2)).sqrt()
                torch.testing.assert_close(matched.square().sum((1, 2)).sqrt(), an,
                    atol=1e-6, rtol=1e-5)
                collected.append(dict(shot=np.full(len(use), sid), endpoint=use,
                    actual_norm=an.cpu().numpy(), constant_norm=cn.cpu().numpy(),
                    parent=net.from_tokens(h, valid, mode='off')[0].sigmoid().cpu().numpy(),
                    actual=actual.cpu().numpy(), constant=score(cd).cpu().numpy(),
                    matched_constant=score(matched).cpu().numpy(),
                    cosine=(a*cd).sum((1, 2)).div((an*cn).clamp_min(torch.finfo(an.dtype).tiny)).cpu().numpy()))
            if (index+1) % 90 == 0:
                print(dict(stage='constant_probe', shots=index+1), flush=True)
    z = {k: np.concatenate([r[k] for r in collected]) for k in collected[0]}
    assert all(np.isfinite(v).all() for v in z.values())
    use = z['actual_norm'] > 0
    ratios = z['constant_norm'][use] / z['actual_norm'][use]
    by_shot = [float(np.median(z['constant_norm'][(z['shot']==sid)&use] /
        z['actual_norm'][(z['shot']==sid)&use])) for sid in np.unique(z['shot'])
        if np.any((z['shot']==sid)&use)]
    summary = dict(scope='POST_HOC_DIAGNOSTIC_ONLY; every128th full past window; no outcomes, no fitting, not TPR/FPR',
        intervention='constant means only in auxiliary coordinates; real prediction h/q unchanged; mask unchanged',
        windows=len(z['shot']), shots=len(np.unique(z['shot'])), zero_reference_writes=int((~use).sum()),
        constant_write_norm_ratio_quantiles=np.quantile(ratios, [.1,.5,.9]).tolist(),
        median_shot_median_write_ratio=float(np.median(by_shot)),
        cosine_quantiles=np.quantile(z['cosine'][use], [.1,.5,.9]).tolist(),
        score_summary={k: dict(median=float(np.median(z[k])),
            fraction_sampled_points_over_threshold=float(np.mean(z[k] >= p['threshold'])))
            for k in ('parent','actual','constant','matched_constant')},
        absolute_probability_change_quantiles={k: np.quantile(np.abs(z[k]-z['actual']), [.1,.5,.9]).tolist()
            for k in ('constant','matched_constant')},
        completed_sha256=sha(root/'completed.json'), checkpoint_sha256=sha(root/'layers.pt'),
        code_sha256=sha(Path(__file__)), source_unchanged=True)
    assert_unchanged(net, snapshot)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    np.savez_compressed(root/'constant_probe.npz', **z)
    summary['arrays_sha256'] = sha(root/'constant_probe.npz')
    _durable_json(target, summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
