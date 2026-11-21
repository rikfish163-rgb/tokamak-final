"""Read-only state-discrimination diagnostic; no outcomes or model selection.

Measure the actual eight learned tanh key coordinates, excluding the appended
constant. Shots have equal weight. A low rank or saturation is descriptive,
not proof of missing physics or a cause of misclassification.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .delayed_recurrent_memory_20261025.model import patch_signal
from .event_teaching_20261026.checkpoint import load_event_memory
from .event_teaching_20261026.learning import KINDS
from .lifecycle_stream_20261025.model import LifecycleAdapter
from .redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json
from .simple_train import sha
from .warmstart_alignment_20261022.alignment import transform


def key_statistics(keys):
    x = keys.detach().cpu().double().numpy()
    if (x.ndim != 2 or x.shape[1] != 8 or not len(x)
            or not np.isfinite(x).all() or np.abs(x).max() > 1):
        raise ValueError('nonempty finite N-by-8 tanh coordinates required')
    mean = x.mean(0)
    centered = x-mean
    cov = centered.T@centered/len(x)
    return dict(patches=len(x), mean=mean.tolist(), covariance=cov.tolist(),
                within_trace=float(np.trace(cov)),
                saturated_fraction=float((np.abs(x) > .95).mean()),
                step_rms=float(np.sqrt(np.mean(np.diff(x, axis=0)**2))) if len(x) > 1 else None)


def summarize(rows):
    if not rows or len({r['shot'] for r in rows}) != len(rows):
        raise ValueError('nonempty distinct shots required')
    means = np.array([r['mean'] for r in rows])
    mean = means.mean(0)
    within = np.mean([r['covariance'] for r in rows], axis=0)
    between = (means-mean).T@(means-mean)/len(rows)
    total = within+between

    def rank(cov):
        squared = float((cov*cov).sum())
        return float(np.trace(cov)**2/squared) if squared else None

    return dict(shots=len(rows), patches=sum(r['patches'] for r in rows),
                mean=mean.tolist(), within_covariance=within.tolist(),
                total_covariance=total.tolist(), within_trace=float(np.trace(within)),
                between_trace=float(np.trace(between)), total_trace=float(np.trace(total)),
                within_effective_rank=rank(within), total_effective_rank=rank(total),
                saturated_fraction=float(np.mean([r['saturated_fraction'] for r in rows])))


@torch.no_grad()
def audit(source_root, stream_root, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    start = time.monotonic()
    base, _ = load_parent(DEFAULT_PARENT)
    models, receipts = {}, {}
    for kind in KINDS:
        model = LifecycleAdapter(base, readout='full', feedback='progressive').eval()
        receipts[kind] = load_event_memory(model, source_root/f'{kind}_epoch2.pt')
        models[kind] = model
    protocol = json.loads((stream_root/'protocol.json').read_text())
    verified = json.loads((stream_root/'independent_audit.json').read_text())
    order = json.loads((stream_root/'order.json').read_text())
    ids = [r['shot'] for r in order]
    mapping = json.loads((stream_root/'input_map.json').read_text())
    if (not verified['complete360'] or protocol['source_sha256'] != PARENT_SHA256
            or sha(stream_root/'order.json') != protocol['order_sha256']
            or mapping['calibration_ids'] != ids[:20] or mapping['policy'] != 'density_gain'
            or len(ids) != 360 or len(set(ids)) != 360):
        raise ValueError('audited complete stream with exact cold20 mapping required')
    val = simple_data.source_roles()['val']
    rows = {k: {d: [] for d in ('source_val160', 'east_raw360', 'east_mapped360')}
            for k in KINDS}
    hashes = {}
    for device, shot_ids in (('J-TEXT', val), ('EAST', ids)):
        for index, sid in enumerate(shot_ids, 1):
            s = apply_input_contract(simple_data.read_shot(
                device, sid, rate_contract='nominal' if device == 'EAST' else 'strict'))
            signal = {k: s[k] for k in ('values', 'available')}
            variants = {'source_val160': signal} if device == 'J-TEXT' else {
                'east_raw360': signal, 'east_mapped360': transform(signal, mapping)}
            for name, signals in variants.items():
                means, complete = patch_signal(signals, base.center, base.scale)
                features = torch.cat((means*complete, complete.to(means)), -1)
                for kind, model in models.items():
                    keys = model.memory.key(features).tanh()
                    rows[kind][name].append(dict(shot=sid, **key_statistics(keys)))
            hashes[f'{device}:{sid}'] = sha(io23.path_for(device, sid))
            if index % 40 == 0:
                print(json.dumps(dict(device=device, shots=index, total=len(shot_ids))), flush=True)
    summary = {k: {d: summarize(v) for d, v in domains.items()} for k, domains in rows.items()}
    for k in KINDS:
        summary[k]['east_mapped_cold20'] = summarize(rows[k]['east_mapped360'][:20])
        summary[k]['east_mapped_suffix340'] = summarize(rows[k]['east_mapped360'][20:])
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source changed during read-only diagnostic')
    result = dict(scope='POST_HOC_SIGNAL_ONLY_REPRESENTATION_DIAGNOSTIC_NOT_PERFORMANCE',
        source_memory=receipts, source_sha256=PARENT_SHA256,
        stream_root=str(stream_root.resolve()), map_sha256=sha(stream_root/'input_map.json'),
        source_val_ids=val, east_ids=ids, data_sha256=hashes,
        equal_shot_weight=True, saturation_definition='abs(tanh_key)>0.95, descriptive only',
        constant_key_excluded=True, hdf5_outcome_reads=0,
        outcomes_used_for_diagnostic=False, parameters_modified=False,
        caveat='Cold20 mapped summaries are retrospective; map first deployed at shot21. '
               'Low rank/saturation alone does not establish risk failure or physical causality.',
        summary=summary, per_shot=rows, seconds=time.monotonic()-start,
        code_sha256=sha(Path(__file__)))
    _durable_json(output, result)
    print(json.dumps(dict(output=str(output), seconds=result['seconds'], summary={
        k: {d: {m: v[m] for m in ('shots', 'saturated_fraction', 'within_trace',
                                 'within_effective_rank', 'total_effective_rank')}
            for d, v in domains.items()} for k, domains in summary.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('stream_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.source_root, args.stream_root, args.output)
