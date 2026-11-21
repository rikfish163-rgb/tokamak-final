"""Post-hoc diagnostics only; never feeds scores or outcomes back to the run."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import models_for_calibration, REFERENCE
from ..simple_online import _durable_json
from ..simple_train import sha
from .audit import read_record
from .learning import meta_loss
from .run import ARMS


def receipt_episode(receipt):
    return {key: receipt[key] for key in ('count', 'end', 'label')}


def main(root):
    torch.set_num_threads(2)
    verified = READ(root / 'verification.json')
    assert verified['completed_sha256'] == sha(root / 'completed.json')
    p = READ(root / 'per_shot.json')
    outcomes = READ(root / 'outcomes.json')
    assert len(outcomes) == 360
    blocks = []
    for start in range(0, 360, 60):
        entries = {}
        for arm in ARMS:
            rows = p[arm + '/calibrated']['predictions'][start:start + 60]
            nd = sum(r['is_disrupt'] for r in rows)
            entries[arm] = dict(disrupt=nd, safe=len(rows)-nd,
                tp=sum(r['timely'] for r in rows),
                fp=sum(r['alarm'] and not r['is_disrupt'] for r in rows))
        blocks.append(dict(first=start+1, last=start+60, counts=entries))
    maxima, means, inner, outer, crossings = [], [], [], [], []
    for row in outcomes:
        sid = row['shot']
        with np.load(root / 'shots' / f'{sid}.npz') as z:
            difference = abs(z['meta_intra']-z['meta_hold'])
            maxima.append(float(difference.max()))
            means.append(float(difference.mean()))
            if row['is_disrupt']:
                lead = np.round((row['down']-z['times'])*1000, 9)
                eligible = (lead > 30) & (lead <= 125)
                if eligible.any():
                    crossings.append([float(z[a][eligible].max()) for a in ARMS])
        for update in READ(root / 'shots' / f'{sid}_intra.json')['meta_intra']:
            if update['gradient_steps']:
                inner.append([update['loss_before'], update['loss_after'], update['parameter_delta_norm']])
        for step in READ(root / 'shots' / f'{sid}_inter.json')['steps']:
            assert 'meta' in step
            m = step['meta']
            outer.append([m['before_bce'], m['after_bce'], m['parameter_delta_norm']])
    inner, outer = np.asarray(inner), np.asarray(outer)
    gradients = []
    parent, _ = make_parent()
    mapping = READ(REFERENCE / 'input_map.json')
    for index in (0, 180, 340):
        state = torch.load(root / 'checkpoints' / f'before{index:03}.pt', map_location='cuda', weights_only=True)
        model = models_for_calibration(parent)['mapped_fixed']
        model.load_state_dict(state['meta'])
        names = set(model.encoder_names())
        for name, param in model.named_parameters():
            param.requires_grad_(name in names)
        parameters = [v for k, v in model.named_parameters() if k in names]
        sid = outcomes[index]['shot']
        # These targets belong to already ended shots. They are available to
        # this retrospective diagnostic, NOT to the original prediction.
        receipt = READ(root / 'shots' / f'{sid}_inter.json')['steps'][0]['meta']
        episodes = []
        for episode in receipt['episodes']:
            outcome = io23.read_outcome('EAST', episode['shot'])
            shot = read_record(episode['shot'], outcome, mapping)
            episodes.append((shot, receipt_episode(episode)))
        vectors = []
        losses = []
        for adapt in (False, True):
            loss, _, _ = meta_loss(model, episodes, device='cuda', adapt=adapt)
            vectors.append(torch.cat([g.detach().reshape(-1).double()
                for g in torch.autograd.grad(loss, parameters)]))
            losses.append(float(loss.detach()))
        ce, meta = vectors
        assert ce.norm() > 0 and meta.norm() > 0
        gradients.append(dict(index=index, shot=sid, ce_loss=losses[0], meta_loss=losses[1],
            ce_norm=float(ce.norm()), meta_norm=float(meta.norm()),
            cosine=float(torch.dot(ce, meta)/(ce.norm()*meta.norm())),
            relative_gradient_difference=float((meta-ce).norm()/ce.norm())))
    result = dict(scope='POST_HOC_DIAGNOSTIC_NOT_AN_ONLINE_POLICY',
        completed_sha256=sha(root/'completed.json'), verification_sha256=sha(root/'verification.json'),
        code_sha256=sha(Path(__file__)), blocks60=blocks, gradient_comparison=gradients,
        real_vs_hold=dict(nonidentical_shots=sum(v > 0 for v in maxima),
            median_max=float(np.median(maxima)), maximum=max(maxima), median_mean=float(np.median(means))),
        real_intra=dict(updates=len(inner), decreased_loss=int((inner[:, 1] < inner[:, 0]).sum()),
            median_loss_drop=float(np.median(inner[:, 0]-inner[:, 1])), median_parameter_delta=float(np.median(inner[:, 2]))),
        meta_training=dict(steps=len(outer), adapted_bce_lower=int((outer[:, 1] < outer[:, 0]).sum()),
            adapted_bce_higher=int((outer[:, 1] > outer[:, 0]).sum()),
            median_bce_change=float(np.median(outer[:, 1]-outer[:, 0])),
            median_parameter_delta=float(np.median(outer[:, 2]))),
        diagnostic_30_125_crossings=dict(zip(ARMS, (np.asarray(crossings) >= READ(root/'protocol.json')['threshold']).sum(0).tolist())))
    _durable_json(root/'diagnostics.json', result)
    print({k:v for k,v in result.items() if k != 'blocks60'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
