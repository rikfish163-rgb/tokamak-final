"""Re-execute every changed shot plus 70323 using the two saved auxiliary targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.risk_aligned_online.anchored_stream_20261031 import run as runner
from experiments.risk_aligned_online.anchored_stream_20261031.prepare import unpack
from experiments.risk_aligned_online.context_vector_20261030.replay_full import (
    _array_error, assert_frozen, no_outcomes, snapshot_frozen,
)

NAMES = runner.NAMES


def select_shots(order, comparisons):
    if len(order) != 360 or len(set(order)) != 360 or order[20] != 70323:
        raise ValueError('locked 360-shot order required')
    if set(comparisons) != set(NAMES):
        raise ValueError('all comparison arms required')
    selected = {70323}
    for item in comparisons.values():
        if set(item) != {'rescued', 'harmed'}:
            raise ValueError('paired shot changes required')
        selected.update(item['rescued'])
        selected.update(item['harmed'])
    if not selected <= set(order[20:]):
        raise ValueError('startup predictions cannot change')
    return [sid for sid in order[20:] if sid in selected]


def compare_exact(times, curves, updates, saved, sealed_updates):
    if set(curves) != set(NAMES) or set(saved) != {'times', *NAMES}:
        raise ValueError('ten complete curves required')
    _array_error(times, saved['times'], 'times')
    for name in NAMES:
        _array_error(curves[name], saved[name], name)
    if json.dumps(updates, sort_keys=True, allow_nan=False) != json.dumps(
            sealed_updates, sort_keys=True, allow_nan=False):
        raise ValueError('update receipts differ')


def replay(directory):
    directory = directory.absolute()
    output = directory.parent.parent/'ANCHORED_FULLSTREAM_GPU_REPLAY_RESULT_20261031.json'
    if output.exists():
        raise FileExistsError(output)
    complete = runner.read(directory/'complete.json')
    if complete['complete'] is not True:
        raise ValueError('completed run required')
    for name, digest in complete['artifact_sha256'].items():
        if runner.sha(directory/name) != digest:
            raise ValueError('completed artifact changed: '+name)
    protocol = runner.read(directory/'protocol.json')
    for name, digest in protocol['code_sha256'].items():
        if runner.sha(runner.io23.ROOT/name) != digest:
            raise ValueError('prediction code changed: '+name)
    reference = Path(protocol['reference'])
    for name, digest in protocol['reference_hashes'].items():
        if runner.sha(reference/name) != digest:
            raise ValueError('reference changed: '+name)
    ids = protocol['order']
    selected = select_shots(ids, complete['comparisons']['full360'])
    barrier = runner.read(directory/'all_predictions_sealed.json')
    receipts = runner.read(directory/'receipts.json')
    if len(receipts) != 360 or barrier['count'] != 360:
        raise ValueError('complete prediction barrier required')
    mapping = runner.read(reference/'input_map.json')
    raw_hashes = runner.read(reference/'input_hashes.json')
    runner.configure()
    base, parent = runner.load_parent(runner.DEFAULT_PARENT)
    if float(parent['threshold']) != protocol['threshold']:
        raise ValueError('threshold changed')
    net = runner.MatrixAttention(base.cuda()).eval()
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    policies = {v: unpack(torch.load(directory/f'{v}.pt', map_location='cpu', weights_only=True))
                for v in runner.VARIANTS}
    before = snapshot_frozen(net, slow, policies)
    donors = {v: {k:p[k].clone() for k in ('donor_origins','donor_feedback')}
              for v,p in policies.items()}
    checked = []
    with no_outcomes():
        for sid in selected:
            index = ids.index(sid)
            stem = f'{index+1:04d}_{sid}'
            path = directory/'shots'/f'{stem}.npz'
            sp = directory/'shots'/f'{stem}_seal.json'
            receipt = receipts[index]
            if (receipt['shot'] != sid or runner.sha(sp) != receipt['seal_sha256']
                    or runner.sha(path) != receipt['sha256']
                    or barrier['seals'][index]['sha256'] != receipt['sha256']):
                raise ValueError('shot seal changed')
            if runner.sha(runner.io23.path_for('EAST',sid)) != raw_hashes[str(sid)]:
                raise ValueError('raw signal changed')
            raw = runner.apply_input_contract(runner.simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal = runner.transform({k:raw[k] for k in ('values','available')},mapping)
            curves, updates = runner.predict(net,signal,slow,policies)
            runner.validate_receipts(updates,len(raw['times']))
            with np.load(path,allow_pickle=False) as saved:
                compare_exact(raw['times'],curves,updates,saved,runner.read(sp)['updates'])
            checked.append(dict(shot=sid,curve_count=len(NAMES),events=len(updates),max_abs_error=0.))
            print(dict(exact_shot=sid,finished=len(checked),total=len(selected)),flush=True)
    assert_frozen(net,slow,policies,before)
    if any(not torch.equal(policies[v][k],x) for v,row in donors.items() for k,x in row.items()):
        raise AssertionError('donor cache changed')
    for name,digest in complete['artifact_sha256'].items():
        if runner.sha(directory/name)!=digest:
            raise ValueError('completed artifact changed during replay')
    report=dict(replay_passed=True,selection='all paired changes in all ten arms plus 70323',
                selected=selected,results=checked,exact_arrays=len(checked)*(len(NAMES)+1),
                parameters_unchanged=True,receipts_exact=True,raw_outcomes_blocked=True,
                complete_sha256=runner.sha(directory/'complete.json'),
                replay_code_sha256=runner.sha(Path(__file__)),
                independent_confirmation=False)
    runner._durable_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    print(replay(parser.parse_args().run))
