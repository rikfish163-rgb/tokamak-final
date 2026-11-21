"""Fixed seven-arm same-shot pairing diagnostic, never an efficacy claim."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..event_routed_20261024.run import (
    READ, PREVIOUS, REFERENCE, SCREEN, load_metric_bank, make_parent,
    models_for_calibration, signal, transform, DEFAULT_PARENT, PARENT_SHA256,
    state_cpu, assert_unchanged, evaluate,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS, prepare_pairs, predict

POINT = io23.ROOT/'experiments/risk_aligned_online/runs/event_routed360_20261024'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/同炮前后配对_事前约定_20261024.md'


def eligible(results):
    if any(r['n_disrupt'] != 180 or r['n_safe'] != 180 for r in results.values()):
        return False
    pair = results['paired']; net = lambda r:r['tp']-r['fp']
    return (pair['fp'] <= 36 and pair['tp'] >= results['off']['tp']
            and all(net(pair) > net(results[a]) for a in ('off', 'point', 'augmented', 'unpaired', 'reverse', 'uniform')))


def load_histories(preparation_ids, mapping):
    histories = {}
    for sid in preparation_ids:
        flag, down = io23.read_outcome('EAST', sid)
        if not flag:
            continue
        raw = signal(sid)
        mapped = transform({k:raw[k] for k in ('values', 'available')}, mapping)
        histories[sid] = dict(mapped, times=raw['times'], down=down)
    return histories


def main(root, limit=None):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True); torch.backends.cuda.matmul.allow_tf32=False
    prior = READ(PREVIOUS/'protocol.json'); order = prior['shot_ids']
    assert len(order) == 360 and set(order) == set(io23.cohort_ids('EAST'))
    if limit is not None:
        order = order[:limit]
    assert not set(order) & set(prior['preparation_ids']+prior['calibration_ids'])
    parent, _ = make_parent(); model = models_for_calibration(parent)['mapped_fixed']
    original = state_cpu(model); mapping = READ(REFERENCE/'input_map.json'); start = time.monotonic()
    root.mkdir(parents=True); (root/'shots').mkdir(); (root/'receipts').mkdir()
    files = set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_within_shot_pair.py')
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    protocol = dict(shot_ids=order, preparation_ids=prior['preparation_ids'], calibration_ids=prior['calibration_ids'],
        threshold=prior['threshold'], source_sha256=PARENT_SHA256,
        parent_sha256=sha(REFERENCE/'fold0/mapped_fixed/prepared.pt'), mapping_sha256=sha(REFERENCE/'input_map.json'),
        previous_completed_sha256=sha(PREVIOUS/'completed.json'), point_completed_sha256=sha(POINT/'completed.json'),
        plan_sha256=sha(PLAN), screen_completed_sha256=sha(SCREEN/'completed.json'),
        screen_recheck_sha256=sha(SCREEN/'same_history_recheck/completed.json'),
        metric_sha256={f:sha(SCREEN/f) for f in ('encoders.pt', 'standardizer.npz')},
        historical_bank_sha256={f:sha(PREVIOUS/f) for f in ('bank.npz', 'bank.json', 'gradients.pt')},
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in order+prior['preparation_ids']},
        arms=('off',)+ARMS, update_every=128, step_learning_rate=.01, step_reset='parent every128points',
        early_lead_min_ms=250., early_stride_points=16, late_lead_ms='(30,125]',
        encoder_names=model.encoder_names(), inter_steps=0, alarm_updates=0,
        event_labels_used='source TRAIN offline only; metric frozen from previous run',
        label_access='20 historical preparation shots only until every arm/shot sealed',
        scope='DEVELOPMENT_ONLY; not chronological commissioning; supervised history routing, not pure SSL',
        inference_backend='eval/no_grad,MHA_fastpath=True,batch32', independent_confirmation=False,
        torch=torch.__version__, gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f.relative_to(io23.ROOT)))
    original_read = io23.read_outcome
    def history_only(device, sid):
        if device != 'EAST' or sid not in prior['preparation_ids']:
            raise AssertionError('stream outcome forbidden before every prediction seal')
        return original_read(device, sid)
    with patch.object(io23, 'read_outcome', side_effect=history_only):
        bank = load_metric_bank()
        assert set(bank['shots']) == set(prior['preparation_ids'])
        tensors, pairs = prepare_pairs(model, bank, load_histories(prior['preparation_ids'], mapping))
        np.savez_compressed(root/'pairs.npz', **tensors)
        _durable_json(root/'pairs.json', pairs)
        torch.save({k:v.cpu() for k,v in bank['pair_gradients'].items()}, root/'pair_gradients.pt')
        print(dict(stage='bank_ready', pairs=len(pairs), seconds=round(time.monotonic()-start,1)), flush=True)
        total = {a:0 for a in ARMS}
        for i, sid in enumerate(order):
            if time.monotonic()-start > 1200:
                raise TimeoutError('declared20minute runtime exceeded')
            raw = signal(sid); mapped = transform({k:raw[k] for k in ('values','available')}, mapping)
            curves, receipts = predict(model, mapped, bank)
            with np.load(POINT/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'], z['times'])
                np.testing.assert_array_equal(curves['off'], z['off'], err_msg=f'parent mismatch{sid}')
                np.testing.assert_array_equal(curves['point'], z['risk'], err_msg=f'point control mismatch{sid}')
            np.savez_compressed(root/'shots'/f'{sid}.npz', times=raw['times'], **curves)
            _durable_json(root/'receipts'/f'{sid}.json', receipts)
            for a in ARMS:
                total[a] += sum(r[a]['step_norm'] > 0 for r in receipts)
            if (i+1)%10 == 0:
                print(dict(stage='predict', shots=i+1, seconds=round(time.monotonic()-start,1)), flush=True)
    assert_unchanged(model, original); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    _durable_json(root/'prediction_seals.json', {str(f.relative_to(root)):sha(f) for f in (root/'shots').glob('*.npz')})
    results = evaluate(root, protocol, arms=ARMS)
    done = dict(status='completed', metrics=results, temporary_updates=total, static_and_point_max_error=0.,
        candidate_promoted=False, eligible_for_inter_study=eligible(results), seconds=time.monotonic()-start,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.iterdir() if f.is_file()}, scientific_goal_achieved=False)
    _durable_json(root/'completed.json', done); print(done, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True); parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    try:
        main(args.out,args.limit)
    except Exception as exc:
        if args.out.exists():
            _durable_json(args.out/'failed.json', dict(type=type(exc).__name__, error=str(exc)))
        raise
