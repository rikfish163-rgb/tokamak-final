"""Post-hoc factorial parameter swaps; not a trained or evaluated online policy."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .delayed_recurrent_memory_20261025.learning import batch
from .event_teaching_20261026.checkpoint import load_event_memory
from .lifecycle_stream_20261025.audit import digest
from .lifecycle_stream_20261025.model import LifecycleAdapter
from .redesign_20261004.prepare_prior import load_parent
from .simple_online import _durable_json
from .warmstart_alignment_20261022.alignment import transform

STAGES = ('source', 'cold_slow_only', 'cold_output_only', 'cold')


def sample_endpoints(valid_indices):
    indices = np.asarray(valid_indices, dtype=int)
    indices = indices[indices >= 127]
    if not len(indices):
        raise ValueError('no complete 128-point causal window')
    return indices[np.linspace(0, len(indices)-1, min(32, len(indices)), dtype=int)]


@torch.no_grad()
def factorial_predictions(model, h, valid, delta, source, cold):
    original = model.adapted_state()
    result = {}
    try:
        for stage, (s, o) in zip(STAGES, ((source, source), (cold, source),
                                        (source, cold), (cold, cold))):
            model.restore({'slow': s['slow'], 'fast_output': o['fast_output']})
            pair = {}
            for name, d in (('off', torch.zeros_like(delta)), ('on', delta)):
                z = model.risk(h, valid, d)
                if not torch.isfinite(z).all():
                    raise FloatingPointError('nonfinite factorial diagnostic')
                pair[name] = z.detach().cpu().double().numpy()
            result[stage] = pair
    finally:
        model.restore(original)
    return result


def stage_statistics(pair):
    zoff, zon = (np.asarray(pair[k], dtype=np.float64) for k in ('off', 'on'))
    poff, pon = (torch.as_tensor(z).sigmoid().numpy() for z in (zoff, zon))
    d = pon-poff
    return dict(mean_probability_change=float(d.mean()), rms_probability_change=float(np.sqrt(np.mean(d*d))),
                mean_logit_change=float((zon-zoff).mean()), positive_point_fraction=float((d > 0).mean()))


def summarize_stage(rows):
    fields = ('mean_probability_change', 'rms_probability_change',
              'mean_logit_change', 'positive_point_fraction')
    if not rows:
        raise ValueError('nonempty shot group required')
    return dict(shots=len(rows), positive_mean_shots=sum(r['mean_probability_change'] > 0 for r in rows),
                negative_mean_shots=sum(r['mean_probability_change'] < 0 for r in rows),
                **{k: dict(mean=float(np.mean([r[k] for r in rows])),
                           median=float(np.median([r[k] for r in rows]))) for k in fields})


@torch.no_grad()
def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    start = time.monotonic()
    protocol = json.loads((root/'protocol.json').read_text())
    audited = json.loads((root/'independent_audit.json').read_text())
    done = json.loads((root/'completed.json').read_text())
    if (done['status'] != 'COMPLETED' or not audited['complete360'] or protocol['warmup'] != 20
            or protocol.get('feedback') != 'progressive' or protocol['readout'] != 'full'):
        raise ValueError('completed audited progressive/full 360 with cold20 required')
    for file, expected in protocol['code_sha256'].items():
        if digest(io23.ROOT/file) != expected:
            raise ValueError(f'stream code changed: {file}')
    order = json.loads((root/'order.json').read_text())
    if len(order) != 360 or len({r['shot'] for r in order}) != 360 or digest(root/'order.json') != protocol['order_sha256']:
        raise ValueError('full360 order mismatch')
    base, _ = load_parent(protocol['source_checkpoint'])
    model = LifecycleAdapter(base, readout='full', feedback='progressive').eval()
    receipt = load_event_memory(model, protocol['source_memory']['path'])
    if receipt['sha256'] != protocol['source_memory']['sha256']:
        raise ValueError('source memory changed')
    source = model.adapted_state()
    cold_path = root/'models'/'cold.pt'
    cold = torch.load(cold_path, map_location='cpu', weights_only=True)
    if set(cold) != set(source) or any(cold[k].shape != source[k].shape or not torch.isfinite(cold[k]).all() for k in source):
        raise ValueError('cold parameter schema mismatch')
    if source['slow'].count_nonzero():
        raise ValueError('source slow expected to be zero')
    initial_hashes = {str(p): digest(p) for p in (cold_path, root/'input_map.json', root/'protocol.json',
        Path(protocol['source_checkpoint']), Path(protocol['source_memory']['path']))}
    mapping = json.loads((root/'input_map.json').read_text())
    hashes = json.loads((root/'input_hashes.json').read_text())
    rows = []
    for index, item in enumerate(order[20:], 21):
        sid = item['shot']
        if digest(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('HDF5 changed')
        path = root/'shots'/f'{index:04d}_{sid}.npz'
        seal = json.loads(path.with_name(path.stem+'_seal.json').read_text())
        if digest(path) != seal['sha256'] or seal['outcome_available']:
            raise ValueError('sealed prediction mismatch')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        # Diagnostic runs AFTER completion, not during the stream. Outcome only
        # reproduces its fixed valid interval and labels descriptive groups.
        disruptive, down = io23.read_outcome('EAST', sid)
        valid = (down-raw['times'])*1000 >= -1e-6 if disruptive else np.ones(len(raw['times']), bool)
        ends = sample_endpoints(np.flatnonzero(valid))
        signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        delta, _ = model.memory_delta(signal)
        x, m = batch(signal, ends, 'cpu')
        h, v = model.tokens(x, m)
        predictions = factorial_predictions(model, h, v, delta[ends], source, cold)
        replay_error = {}
        with np.load(path) as sealed:
            if not np.array_equal(raw['times'], sealed['times']):
                raise ValueError('time base mismatch')
            for mode, arm in (('off', 'cold'), ('on', 'intra')):
                reproduced = torch.as_tensor(predictions['cold'][mode]).sigmoid().numpy()
                error = float(np.max(np.abs(reproduced-sealed[arm][ends])))
                if error > 5e-6:
                    raise AssertionError(f'predeclared CPU replay tolerance exceeded: {sid} {mode} {error}')
                replay_error[mode] = error
        for a, b in (('source', 'cold_output_only'), ('cold_slow_only', 'cold')):
            if not np.array_equal(predictions[a]['off'], predictions[b]['off']):
                raise AssertionError('output must have no effect when fast is off')
        rows.append(dict(shot=sid, index=index, disruptive=disruptive, endpoints=ends.tolist(),
                         replay_error=replay_error,
                         stages={k: stage_statistics(v) for k, v in predictions.items()}))
        if index % 20 == 0:
            print(json.dumps(dict(shots=len(rows), total=340, seconds=round(time.monotonic()-start, 2))), flush=True)
    for path, expected in initial_hashes.items():
        if digest(Path(path)) != expected:
            raise AssertionError('input artifact changed during diagnostic')
    groups = {'all340': rows, 'disruptive': [r for r in rows if r['disruptive']],
              'safe': [r for r in rows if not r['disruptive']]}
    result = dict(scope='POST_HOC_STAGE_FACTORIAL_DIAGNOSTIC_NOT_NEW_STREAM_PERFORMANCE',
        source_root=str(root.resolve()), samples_per_shot=32, shots=340, causal_history=128,
        code_sha256=digest(Path(__file__)), input_hashes=initial_hashes,
        audit_sha256=digest(root/'independent_audit.json'), source_memory=receipt,
        trained_parameters=0, device='cpu', parameter_changes={k: float((cold[k]-source[k]).norm()) for k in source},
        shared_identical_fast_state=True, deployment_threshold_changed=False, seconds=time.monotonic()-start,
        replay_max_abs_error=max(v for r in rows for v in r['replay_error'].values()),
        summary={g: {stage: summarize_stage([r['stages'][stage] for r in group]) for stage in STAGES}
                 for g, group in groups.items()}, per_shot=rows,
        limits=['Uniformly sampled endpoints, not full shot alarm evaluation.',
                'Parameter swapping localizes readout effects; it does not identify the causal training-loss term.',
                'Development data, no independent confirmation or new physics claim.'])
    _durable_json(output, result)
    print(json.dumps({k: result[k] for k in ('summary', 'replay_max_abs_error', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    run(a.root, a.output)
