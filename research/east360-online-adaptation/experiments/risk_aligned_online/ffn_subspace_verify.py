"""Reopen SOURCE seals and raw outcomes, independently recount operational errors.

No training/GPU, no EAST access. Historical code is checked against its OWN
snapshot, not against whatever code is currently in the worktree. Results stay
development diagnostics and failed gates are never rewritten as passed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from . import io23, simple_data
from .ffn_subspace_audit import whole_source_screen
from .redesign_20261004.prepare_prior import DEFAULT_PARENT
from .simple_online import _durable_json, _hash


def validate_audit(a):
    if (a['scope'] != 'OFFLINE_SOURCE_DIAGNOSTIC_NOT_JTEXT_ONLINE'
            or a['fast_parameter_count'] != 8 or not a['fast_discarded']
            or a['model_parameter_change_norm'] != 0. or a['model_buffer_change_norm'] != 0.):
        raise ValueError('source identity, disposable state and unchanged original tensors required')
    attempts = a['attempts']
    if (len(attempts) != a['attempted_updates']
            or sum(r['accepted'] for r in attempts) != a['accepted_updates']):
        raise ValueError('attempt counts disagree')
    if attempts and (attempts[0]['origin_index'] != 0 or any(attempts[0]['fast_at_origin'])):
        raise ValueError('first source proposal must start at zero disposable state')
    for r in attempts:
        origin, maturity = r['origin_index'], r['maturity_index']
        expected = maturity + 1 if maturity + 1 < a['score_count'] else None
        if (maturity != origin + 25 or r['predictions_sealed_through_index'] != maturity
                or (r['accepted'] and r['effective_index'] != expected)
                or (not r['accepted'] and r['effective_index'] is not None)):
            raise ValueError('signal maturity or prediction sealing boundary violated')
    return dict(attempted=len(attempts), accepted=sum(r['accepted'] for r in attempts))


def verify(directory):
    protocol = json.loads((directory / 'protocol.json').read_text())
    reported = json.loads((directory / 'result.json').read_text())
    if (protocol['validation_ids'] != simple_data.source_roles()['val']
            or protocol['source_sha256'] != _hash(DEFAULT_PARENT)
            or protocol['threshold_tuning'] or protocol['source_training']):
        raise ValueError('source parent/role/threshold contract changed')
    with zipfile.ZipFile(directory / 'code_snapshot.zip') as archive:
        for name, digest in protocol['code_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('archived source code hash mismatch')
    expected_ids = set(protocol['validation_ids']); threshold = protocol['threshold']
    output, traces = {}, {}
    for arm, original in reported['metrics'].items():
        receipts = list((directory / arm).glob('*_seal.json'))
        records, scores_by_shot = [], {}; attempted = accepted = 0
        for path in receipts:
            receipt = json.loads(path.read_text()); sid = receipt['shot']
            npz = path.parent / receipt['npz']
            if sid in scores_by_shot or receipt['threshold'] != threshold or _hash(npz) != receipt['sha256']:
                raise ValueError('duplicate source shot, changed threshold or changed prediction seal')
            with np.load(npz, allow_pickle=False) as sealed:
                times, scores = sealed['times'].copy(), sealed['scores'].copy()
                if int(sealed['shot']) != sid: raise ValueError('source NPZ identity changed')
            if (times.shape != scores.shape or not len(times) or not np.isfinite(scores).all()
                    or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
                    or ((scores < 0) | (scores > 1)).any()):
                raise ValueError('invalid sealed source trajectory')
            disruptive, down = io23.read_outcome('J-TEXT', sid)
            if disruptive and times[-1] > down + 1e-9:
                raise ValueError('postterminal SOURCE predictions entered this diagnostic')
            hits = np.flatnonzero(scores >= threshold)
            lead = round((down - times[hits[0]]) * 1000., 9) if disruptive and len(hits) else None
            records.append(dict(shot=sid, is_disrupt=disruptive, alarm=bool(len(hits)),
                lead_ms=lead, timely=bool(lead is not None and lead > 10.)))
            a = json.loads(path.with_name(path.name.replace('_seal.json', '_audit.json')).read_text())
            if a['score_count'] != len(scores): raise ValueError('source audit/curve length mismatch')
            checked = validate_audit(a); attempted += checked['attempted']; accepted += checked['accepted']
            scores_by_shot[sid] = scores
        if set(scores_by_shot) != expected_ids: raise ValueError('source VAL160 cohort is incomplete')
        nd = sum(r['is_disrupt'] for r in records); ns = len(records) - nd
        tp = sum(r['timely'] for r in records); fp = sum(r['alarm'] and not r['is_disrupt'] for r in records)
        counted = dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, tpr=tp / nd, fpr=fp / ns)
        if any(original[k] != value for k, value in counted.items()):
            raise ValueError('source operational counts differ from recorded result')
        if {r['shot']: r for r in records} != {r['shot']: r for r in original['predictions']}:
            raise ValueError('source per-shot alarms differ from recorded result')
        output[arm] = dict(counted, seals=len(receipts), attempted=attempted, accepted=accepted)
        traces[arm] = scores_by_shot
    for arm, rows in traces.items():
        if arm == 'static': continue
        # Different replay batch sizes may introduce tiny CUDA roundoff before
        # the first update. Log it and require the established 2e-6 tolerance;
        # do not call this bitwise equality or rewrite any original prediction.
        error = max(float(np.max(np.abs(scores[:26] - traces['static'][sid][:26])))
                    for sid, scores in rows.items())
        if error > 2e-6: raise ValueError('pre-update source trajectory changed materially')
        output[arm]['preupdate_max_abs_difference'] = error
    return dict(directory=str(directory.resolve()), arms=output,
                whole_source_screen=whole_source_screen(reported['metrics']),
                source_checkpoint_sha256=_hash(DEFAULT_PARENT),
                scope='SOURCE DEVELOPMENT RECOUNT; no EAST or independent future validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directories', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior verification')
    result = dict(runs=[verify(directory) for directory in args.directories],
                  code_sha256=_hash(__file__))
    _durable_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
