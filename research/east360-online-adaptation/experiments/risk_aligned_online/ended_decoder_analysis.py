"""Post-run forecast fidelity audit; never training or online feedback.

Checks the unchanged zero-fast classifier and actual H25 forecast error on the
same four equally-spaced origins per shot used by maintenance_loss. Fit and
later periods are separate. This numeric accuracy is not physical-law proof.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import simple_data
from .alignment import fit_ended_shot_statistics
from .causal_input import apply_input_contract
from .ended_decoder import DecoderFfnModel, maintenance_loss
from .ended_density_replay import EndedDensityReplay
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    output = args.run/'forecast_fidelity_analysis.json'
    if output.exists() or not (args.run/'independent_verification.json').is_file():
        parser.error('completed independently verified run and new receipt required')
    order = json.loads((args.run/'order.json').read_text())
    torch.set_num_threads(4)
    source, state = load_parent(DEFAULT_PARENT)
    records = [simple_data.attach_outcome(apply_input_contract(simple_data.read_shot(
        'EAST', row['shot'], rate_contract='nominal')), 'EAST', {}) for row in order]
    ids = [s['shot'] for s in records[:30]]
    provider = EndedDensityReplay(state['source_center'], state['source_scale'],
        output_dir=args.run/'meta_after30', order=[s['shot'] for s in records],
        preserve_observed_channels=True)
    # AFTER the whole experiment: reconstruct its ended-first30 transform for
    # offline fidelity evaluation. No pending read/reveal or learner is run.
    provider.statistics = fit_ended_shot_statistics(records[:30], ended_ids=ids)
    records = [provider.reencode(s) for s in records]
    values = {}
    for name in ('original', 'signal_after30', 'meta_after30'):
        base, _ = load_parent(DEFAULT_PARENT)
        model = DecoderFfnModel(base).to('cuda').eval()
        if name != 'original':
            artifact_path = args.run/name/'ended_decoder.pt'
            artifact = torch.load(artifact_path, map_location='cuda', weights_only=True)
            if artifact['parent_sha256'] != PARENT_SHA256 or artifact['fit_ids'] != ids:
                raise AssertionError('auxiliary identity differs from declared training')
            model.decoder.load_state_dict(artifact['decoder'])
        model.requires_grad_(False)
        rows = []
        for shot in records:
            with torch.no_grad():
                loss = maintenance_loss(model, shot)
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'],
                             actual_h25_error=float(loss)))
        values[name] = dict(rows=rows, periods={})
        for period, a, b in (('fit30_reencoded', 0, 30), ('later90', 30, len(records))):
            subset = rows[a:b]
            values[name]['periods'][period] = dict(shot_equal_mean=float(np.mean([
                s['actual_h25_error'] for s in subset])), n=len(subset),
                by_outcome={str(label):float(np.mean([s['actual_h25_error'] for s in subset
                    if s['is_disrupt']==bool(label)])) for label in (0,1)})
    if _hash(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source artifact changed')
    _durable_json(output, dict(values=values, data_used_only_after_completed_replay=True,
        coordinates='same first30 density mapping and fixed source asinh',
        temporary_ffn_zero=True, selection='four equally spaced valid mature origins per shot',
        source_sha256=PARENT_SHA256, no_physical_law_claim=True,
        caveat='fit-period error is training data; later90 still a development cohort'))
    print(json.dumps({k:v['periods'] for k,v in values.items()}), flush=True)


if __name__ == '__main__':
    main()
