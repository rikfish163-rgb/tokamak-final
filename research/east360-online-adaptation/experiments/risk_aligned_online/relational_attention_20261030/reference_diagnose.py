"""Post-hoc conditioning-strength diagnostic; never refits or changes a policy."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import normalized
from .run import schedule, windows


def execute(output):
    target = output/'reference_diagnosis.json'
    if target.exists():
        raise ValueError('new post-hoc sidecar required')
    p = json.loads((output/'protocol.json').read_text())
    reference = Path(p['reference'])
    for name, digest in p['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('reference changed')
    hashes = json.loads((reference/'input_hashes.json').read_text())
    mapping = json.loads((reference/'input_map.json').read_text())
    base, _ = load_parent(DEFAULT_PARENT)
    net = SimpleNamespace(base=base)
    torch.set_num_threads(2)
    u = [io23.RAW23.index(c) for c in ('Bt', 'ip')]
    c = [io23.RAW23.index(c) for c in ('Mir_avg_amp', 'sxr_mean', 'xuv_ratio', 'ne_nG')]
    signals, first = {}, {}
    for sid in p['warmup_ids']:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('cold20 input changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signals[sid] = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        z, mask = normalized(net, torch.from_numpy(signals[sid]['values'][:256]),
                            torch.from_numpy(signals[sid]['available'][:256]))
        joint = mask[:, u+c].all(-1)
        first[sid] = z[joint][:, u].double()
    result = []
    for fold in json.loads((output/'folds.json').read_text()):
        fitted = torch.load(output/f"fold{fold['fold']}.pt", map_location='cpu', weights_only=True)
        ref = fitted['reference']
        mu = torch.stack([first[s].mean(0) for s in fold['fit']]).mean(0)
        cov = torch.stack([(first[s]-mu).T@(first[s]-mu)/len(first[s]) for s in fold['fit']]).mean(0)
        eigenvalues = torch.linalg.eigvalsh(cov)
        held = []
        for sid in fold['held']:
            s = signals[sid]
            fixed = schedule(len(s['values']))
            data = windows(s, [m for _, m, _ in fixed])
            z, masks = normalized(net, data['x'], data['mask'])
            joint = masks[:, :, u+c].all(-1)
            count = joint.reshape(-1, 16, 8).sum(-1)
            selected = torch.where(joint[..., None], z[:, :, u+c], 0.).double()
            means = selected.reshape(-1, 16, 8, 6).sum(2)/count.clamp_min(1)[..., None]
            valid = (count[:, 1:] >= 6) & (count[:, :-1] >= 6)
            delta = (means[:, 1:] - means[:, :-1])[valid]
            correction = delta[:, :2]@ref['beta'][1:]
            raw_energy = (delta[:, 2:].square()/ref['scale']).sum(0)
            correction_energy = (correction.square()/ref['scale']).sum(0)
            held.append(dict(shot=sid, differences=len(delta),
                             raw_energy=raw_energy.tolist(), correction_energy=correction_energy.tolist(),
                             scaled_correction_energy_fraction=float(correction_energy.sum()/raw_energy.sum())
                             if raw_energy.sum() > 0 else None))
        result.append(dict(fold=fold['fold'], covariate_names=['Bt', 'ip'],
            training_covariance=cov.tolist(), training_covariance_eigenvalues=eigenvalues.tolist(),
            ridge1_eigen_shrinkage=(eigenvalues/(eigenvalues+1)).tolist(),
            beta=ref['beta'].tolist(), scale=ref['scale'].tolist(), held=held))
    payload = dict(scope='POST_HOC_NO_REFIT_NO_POLICY_CHANGE', completed=True,
                   outcome_reads=0, diagnostic_channels=[io23.RAW23[i] for i in c],
                   results=result, original_protocol_sha256=sha(output/'protocol.json'),
                   verifier_sha256=sha(Path(__file__)))
    _durable_json(target, payload)
    for f in result:
        print(dict(fold=f['fold'], eigenvalues=f['training_covariance_eigenvalues'],
                   shrinkage=f['ridge1_eigen_shrinkage'],
                   correction_energy_fractions={r['shot']:r['scaled_correction_energy_fraction'] for r in f['held']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.absolute())
