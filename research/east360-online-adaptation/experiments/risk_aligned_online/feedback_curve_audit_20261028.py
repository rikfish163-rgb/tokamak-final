"""Post-hoc diagnostics of sealed probabilities, never an online update policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import io23
from .lifecycle_stream_20261025.audit import classify, counts, digest
from .simple_online import _durable_json


def contrast(cold, intra, reverse, threshold):
    arrays = [np.asarray(a, dtype=np.float64) for a in (cold, intra, reverse)]
    if (not 0 < threshold < 1 or arrays[0].ndim != 1 or not arrays[0].size
            or any(a.shape != arrays[0].shape or not np.isfinite(a).all()
                   or ((a < 0) | (a > 1)).any() for a in arrays)):
        raise ValueError('finite aligned nonempty probability vectors and interior threshold required')
    cold, intra, reverse = arrays
    fast, mismatch = intra-cold, intra-reverse
    energy, mismatch_energy = np.mean(fast**2), np.mean(mismatch**2)
    return dict(
        endpoints=len(cold), fast_mean=float(fast.mean()), fast_rms=float(np.sqrt(energy)),
        fast_max_abs=float(np.abs(fast).max()),
        up_fraction=float((fast > 0).mean()), down_fraction=float((fast < 0).mean()),
        centered_energy_fraction=float(np.mean((fast-fast.mean())**2)/energy) if energy else None,
        reverse_rms=float(np.sqrt(mismatch_energy)), reverse_max_abs=float(np.abs(mismatch).max()),
        reverse_over_fast_rms=float(np.sqrt(mismatch_energy/energy)) if energy else None,
        new_above_threshold=int(((cold < threshold) & (intra >= threshold)).sum()),
        removed_above_threshold=int(((cold >= threshold) & (intra < threshold)).sum()),
        reverse_threshold_disagreements=int(((intra >= threshold) != (reverse >= threshold)).sum()),
    )


def summarize(rows):
    if not rows:
        return dict(shots=0, statistics={})
    fields = ('fast_mean', 'fast_rms', 'up_fraction', 'down_fraction',
              'centered_energy_fraction', 'reverse_rms', 'reverse_over_fast_rms')
    stats = {}
    for field in fields:
        values = [r[field] for r in rows if r[field] is not None]
        stats[field] = dict(defined_shots=len(values),
                            median=float(np.median(values)) if values else None,
                            mean=float(np.mean(values)) if values else None)
    return dict(shots=len(rows), statistics=stats,
                any_reverse_threshold_difference=sum(r['reverse_threshold_disagreements'] > 0 for r in rows),
                positive_mean_shift_shots=sum(r['fast_mean'] > 0 for r in rows),
                negative_mean_shift_shots=sum(r['fast_mean'] < 0 for r in rows))


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    p = json.loads((root/'protocol.json').read_text())
    done = json.loads((root/'completed.json').read_text())
    audited = json.loads((root/'independent_audit.json').read_text())
    if done['status'] != 'COMPLETED' or not audited['complete360']:
        raise ValueError('independently audited complete360 required')
    order = json.loads((root/'order.json').read_text())
    assert len(order) == 360 and len({r['shot'] for r in order}) == 360
    assert digest(root/'order.json') == p['order_sha256']
    assert digest(Path(p['source_checkpoint'])) == p['source_sha256']
    inputs = json.loads((root/'input_hashes.json').read_text())
    rows, classified = [], {k: [] for k in ('cold', 'intra', 'intra_reverse')}
    for index, item in enumerate(order, 1):
        shot = item['shot']
        path = root/'shots'/f'{index:04d}_{shot}.npz'
        seal = json.loads(path.with_name(path.stem+'_seal.json').read_text())
        assert digest(path) == seal['sha256'] and not seal['outcome_available']
        assert digest(io23.path_for('EAST', shot)) == inputs[str(shot)]
        # Outcomes are read AFTER stream completion, solely to reproduce the
        # already-fixed valid evaluation interval. No predictions are reissued.
        disruptive, down = io23.read_outcome('EAST', shot)
        with np.load(path) as curves:
            times = curves['times']
            valid = (down-times)*1000 >= -1e-6 if disruptive else np.ones(len(times), bool)
            record = contrast(*(curves[k][valid] for k in classified), p['threshold'])
            for arm in classified:
                classified[arm].append(dict(shot=shot, disruptive=disruptive,
                    **classify(times, curves[arm], disruptive, down, p['threshold'])))
        record.update(index=index, shot=shot, disruptive=disruptive)
        rows.append(record)
    for arm, decisions in classified.items():
        assert counts(decisions) == audited['metrics'][arm]
    result = dict(
        scope='POST_HOC_SEALED_CURVE_DIAGNOSTIC_NOT_NEW_PERFORMANCE_EXPERIMENT',
        root=str(root.resolve()), source_sha256=p['source_sha256'],
        protocol_sha256=digest(root/'protocol.json'), audit_sha256=digest(root/'independent_audit.json'),
        diagnostic_code_sha256=digest(Path(__file__)), threshold=p['threshold'],
        complete_stream_shots=360, warmup_in_primary_performance=True,
        probability_space_not_logit_space=True, summaries_weight_shots_equally=True,
        post_disruption_endpoints_excluded=True, model_updates=0,
        summary={
            'full360': summarize(rows), 'warmup20': summarize(rows[:20]),
            'post_warmup340': summarize(rows[20:]),
            'post_warmup_disruptive': summarize([r for r in rows[20:] if r['disruptive']]),
            'post_warmup_safe': summarize([r for r in rows[20:] if not r['disruptive']]),
        },
        metrics={arm: counts(v) for arm, v in classified.items()}, shots=rows,
        limits=['Endpoint threshold disagreement is not a shot-level error flip.',
                'Small curve differences do not prove identical internal states.',
                'Centered energy is a descriptive decomposition, not evidence of physics.'],
    )
    _durable_json(output, result)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
