"""Ten development calibration shots; freeze seven predictions before labels."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..curvature_memory_20261024.run import PREVIOUS, load_bank
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import make_parent, READ
from ..heldout_threshold_20261023.run import models_for_calibration, REFERENCE, CALIBRATION_IDS
from ..history_tangent_20261023.learning import permute_basis
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..trajectory_memory_20261024.run import signal
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public, predict_many
from .learning import ARMS, history_spaces, predict

RUNS = io23.ROOT/'experiments/risk_aligned_online/runs'
PAIR = RUNS/'within_shot_pair360_20261024'
CAL = RUNS/'heldout_threshold30_test360_20261023'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/真实反馈驱动历史方向_事前约定_20261024.md'


def diagnostic_losses(rows):
    danger, safe = [], []
    for r in rows:
        p = np.asarray(r['scores'], dtype=np.float64)
        if not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('nonempty finite probability trajectory required')
        if r['is_disrupt']:
            lead = np.round((r['down']-r['times'])*1000., 9)
            take = (lead > 30.) & (lead <= 125.)
            if not take.any():
                raise ValueError('no valid timely-window observations')
            danger.append(float(-np.log(p[take].clip(1e-12, 1)).mean()))
        else:
            safe.append(float(-np.log1p(-p.max().clip(0, 1-1e-12))))
    if not danger or not safe:
        raise ValueError('both shot groups required for development loss')
    d, s = float(np.mean(danger)), float(np.mean(safe))
    return dict(danger_loss=d, safe_loss=s, balanced_loss=(d+s)/2)


def eligible(results):
    if any(r['n_disrupt'] != 2 or r['n_safe'] != 8 for r in results.values()):
        return False
    if not all(np.isfinite(r[k]) for r in results.values()
               for k in ('danger_loss', 'safe_loss', 'balanced_loss')):
        return False
    c, off = results['contrast'], results['off']
    return (c['tp'] >= off['tp'] and c['fp'] <= off['fp']
            and c['danger_loss'] <= off['danger_loss'] and c['safe_loss'] <= off['safe_loss']
            and all(c['balanced_loss'] < results[a]['balanced_loss']-1e-6
                    for a in ('off', 'raw', 'global', 'permuted', 'hold', 'reverse')))


def evaluate(root, protocol):
    rows = {a: [] for a in ('off',)+ARMS}; outcomes = []
    for sid in protocol['shot_ids']:
        flag, down = io23.read_outcome('EAST', sid)
        outcomes.append(dict(shot=sid, is_disrupt=bool(flag), down=None if down is None else float(down)))
        with np.load(root/'shots'/f'{sid}.npz') as z:
            _, valid = io23.labels_at(z['times'], flag, down, 125.)
            for a in rows:
                rows[a].append(dict(shot=sid, times=z['times'][valid], scores=z[a][valid], is_disrupt=flag, down=down))
    per_shot = {a: metrics(r, protocol['threshold'], lead=30.) for a, r in rows.items()}
    results = {a: dict(public(m), **diagnostic_losses(rows[a]),
                      very_early_over3000=sum(r['timely'] and r['lead_ms'] > 3000 for r in m['predictions']))
               for a, m in per_shot.items()}
    _durable_json(root/'outcomes.json', outcomes); _durable_json(root/'per_shot.json', per_shot)
    _durable_json(root/'per_shot_losses.json', {
        a: [dict(shot=r['shot'], is_disrupt=bool(r['is_disrupt']),
                 peak=float(np.max(r['scores'])),
                 timely_mean=float(np.mean(r['scores'][(np.round((r['down']-r['times'])*1000., 9)>30.)
                     & (np.round((r['down']-r['times'])*1000., 9)<=125.)])) if r['is_disrupt'] else None)
            for r in rs] for a, rs in rows.items()})
    return results


def main(root):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True); torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic(); prior = READ(PAIR/'protocol.json')
    assert CALIBRATION_IDS == prior['calibration_ids']
    assert not set(CALIBRATION_IDS) & set(prior['preparation_ids']+prior['shot_ids'])
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    inputs = {PAIR: ('pair_gradients.pt',), PREVIOUS: ('bank.npz', 'bank.json', 'gradients.pt'),
              CAL: ('thresholds.json',)}
    for folder, files in inputs.items():
        manifest = READ(folder/'completed.json')['artifact_sha256']
        for name in files:
            assert sha(folder/name) == manifest[name], (folder, name)
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt') == prior['parent_sha256']
    assert sha(REFERENCE/'input_map.json') == prior['mapping_sha256']
    threshold = READ(CAL/'thresholds.json')['mapped_fixed']['0.2']['threshold']
    assert threshold == prior['threshold'] == 0.12540414929389954
    parent, _ = make_parent(); model = models_for_calibration(parent)['mapped_fixed']
    original = state_cpu(model); mapping = READ(REFERENCE/'input_map.json')
    bank = load_bank(); gradients = torch.load(PAIR/'pair_gradients.pt', map_location='cuda', weights_only=True)['paired']
    assert set(bank['shots']) == set(prior['preparation_ids']) and gradients.shape == bank['gradients'].shape
    spaces, basis_receipt = history_spaces(gradients, bank['shots'], bank['labels'])
    spaces['permuted'] = permute_basis(spaces['contrast'], initial_encoder(model))
    root.mkdir(parents=True); (root/'shots').mkdir(); (root/'receipts').mkdir()
    torch.save({k: v.cpu() for k, v in spaces.items()}, root/'spaces.pt')
    _durable_json(root/'spaces.json', basis_receipt)
    files = set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_signal_driven_span.py')
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            files.add(Path(module.__file__))
    protocol = dict(shot_ids=CALIBRATION_IDS, preparation_ids=prior['preparation_ids'],
        excluded360_ids=prior['shot_ids'], threshold=threshold, source_sha256=PARENT_SHA256,
        parent_sha256=prior['parent_sha256'], mapping_sha256=prior['mapping_sha256'],
        history_sha256={str(folder/name): sha(folder/name) for folder, names in inputs.items() for name in names},
        plan_sha256=sha(PLAN), code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST', sid)): sha(io23.path_for('EAST', sid)) for sid in CALIBRATION_IDS},
        arms=('off',)+ARMS, update_every=128, first_update=256, step_norm=.01,
        parameter_names=model.encoder_names(), step_reset='parent every update; discard at shot end',
        current_supervised_updates=0, inter_steps=0, alarm_updates=0, historical_labeled_shots=20,
        calibration_shots_already_used_for_threshold=10, event_labels_used_in_current_loss=False,
        label_access='all seven arms/all ten shots sealed before evaluation; basis uses authorized past labels',
        scope='DEVELOPMENT_ONLY; not chronological commissioning or independent confirmation',
        independent_confirmation=False, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    protocol['inference_backend'] = 'all arms eval/no_grad MHA fastpath batch32; exact fresh parent comparator'
    protocol['legacy_calibration_backend'] = 'gradient-enabled functional math path; reproduced separately, threshold unchanged'
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f.relative_to(io23.ROOT)))
    print(dict(stage='basis_ready', ranks={k: len(v) for k, v in spaces.items()}, seconds=time.monotonic()-start), flush=True)
    total = {a: 0 for a in ARMS}; backend_differences = {}
    def deny(*args, **kwargs):
        raise AssertionError('current outcomes forbidden until all predictions sealed')
    with patch.object(io23, 'read_outcome', side_effect=deny):
        for n, sid in enumerate(CALIBRATION_IDS, 1):
            if time.monotonic()-start > 900:
                raise TimeoutError('declared15minute screen exceeded')
            raw = signal(sid); mapped = transform({k: raw[k] for k in ('values', 'available')}, mapping)
            curves, receipts = predict(model, mapped, spaces)
            # The existing calibration archive used grad-enabled inference;
            # the latest360 and this experiment use no_grad's fused MHA.
            # Verify BOTH paths exactly against their matching comparator,
            # rather than loosening an assertion across different kernels.
            legacy = predict_many(model, {'parent': mapped}, device='cuda')['parent']
            with torch.no_grad():
                independent = predict_many(model, {'parent': mapped}, device='cuda')['parent']
            np.testing.assert_array_equal(curves['off'], independent)
            with np.load(CAL/'calibration'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'], z['times'])
                np.testing.assert_array_equal(legacy, z['mapped_fixed'], err_msg=f'legacy parent mismatch {sid}')
                backend_differences[str(sid)] = float(np.max(np.abs(curves['off']-legacy)))
            np.savez_compressed(root/'shots'/f'{sid}.npz', times=raw['times'], **curves)
            _durable_json(root/'receipts'/f'{sid}.json', receipts)
            for a in ARMS:
                total[a] += sum(r['gradient_steps'] for r in receipts[a])
            print(dict(stage='predict', shots=n, shot=sid, seconds=round(time.monotonic()-start, 2)), flush=True)
    _durable_json(root/'prediction_seals.json', {str(f.relative_to(root)): sha(f) for f in (root/'shots').glob('*.npz')})
    _durable_json(root/'backend_comparison.json', backend_differences)
    assert_unchanged(model, original); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    results = evaluate(root, protocol)
    done = dict(status='completed', metrics=results, temporary_updates=total, static_max_error=0.,
        eligible_for360=eligible(results), scientific_goal_achieved=False, seconds=time.monotonic()-start,
        artifact_sha256={str(f.relative_to(root)): sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json', done)
    print({k: v for k, v in done.items() if k != 'artifact_sha256'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    try:
        main(args.out)
    except Exception as exc:
        if args.out.exists():
            _durable_json(args.out/'failed.json', dict(type=type(exc).__name__, error=str(exc)))
        raise
