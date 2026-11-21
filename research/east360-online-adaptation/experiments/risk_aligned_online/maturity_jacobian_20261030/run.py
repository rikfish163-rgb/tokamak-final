"""Paired teaching/execution clock study; never change the frozen classifier."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..anchored_feedback_20261030.learning import AnchoredTarget
from ..anchored_feedback_20261030.run import ARMS, candidate_gate, evaluate, fit
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_vector_20261030.scaling import transform_context
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..relational_attention_20261030.run import shot_summary
from ..simple_online import _durable_json
from ..simple_train import sha
from .window import maturity_window

ORIGIN = io23.ROOT/'probe_outputs/anchored_feedback_cold20_v1'
CELLS = ('origin_teacher_mature_step', 'mature_teacher_origin_step', 'mature_teacher_mature_step')


def mature_data(data, records):
    if not records or any(r['maturity'] != r['origin']+25 or r['query'] != r['origin']+50
                          for r in records):
        raise ValueError('fixed origin/maturity/query clock required')
    if len(data['x']) != len(records):
        raise ValueError('one clock per observation required')
    x, mask = maturity_window(data['x'], data['mask'], data['later_x'], data['later_mask'])
    return dict(data, x=x, mask=mask)


def promotion(groups, origin_groups, cosine, distinct):
    result = candidate_gate(groups, cosine, distinct)
    for group in ('safe', 'danger'):
        new = groups['real'][group]['mean_delta_bce']
        old = origin_groups['real'][group]['mean_delta_bce']
        if new is None or old is None or new >= old-1e-6:
            result['reasons'].append(group+': not better than origin-teacher/origin-step')
    result['passed'] = not result['reasons']
    return result


def read(path):
    return json.loads(path.read_text())


def check_files(root, mapping):
    for name, expected in mapping.items():
        if sha(root/name) != expected:
            raise ValueError('changed input: '+str(root/name))


def configure():
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')


def load_inputs(origin):
    done, old = read(origin/'complete.json'), read(origin/'protocol.json')
    if not done['complete']:
        raise ValueError('completed origin experiment required')
    check_files(origin, done['artifacts'])
    check_files(io23.ROOT, old['code_sha256'])
    previous, parent = Path(old['previous']), Path(old['parent'])
    check_files(previous, old['input_hashes'])
    check_files(parent, old['parent_hashes'])
    data = torch.load(previous/'observations.pt', map_location='cpu', weights_only=True)
    records, folds = read(origin/'records.json'), read(origin/'folds.json')
    return old, done, data, mature_data(data, records), records, folds


def choice(cell, controllers, data, mature):
    if cell not in CELLS:
        raise ValueError('undeclared factorial cell')
    teacher = controllers['origin' if cell.startswith('origin_teacher') else 'mature']
    inputs = mature if cell.endswith('mature_step') else data
    return teacher, inputs


def execute(output):
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree output required')
    configure()
    old, origin_done, data, mature, records, folds = load_inputs(ORIGIN)
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    slow = torch.load(Path(old['parent'])/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    output.mkdir()
    files = {io23.ROOT/'MATURITY_JACOBIAN_PLAN_20261030.md',
             io23.ROOT/'tests/test_maturity_jacobian_window.py',
             io23.ROOT/'tests/test_maturity_jacobian_run.py', *Path(__file__).parent.glob('*.py')}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='COLD20_TEACHER_BY_GRADIENT_CLOCK_CONDITIONAL_DIAGNOSTIC',
        origin=str(ORIGIN), origin_complete_sha256=sha(ORIGIN/'complete.json'),
        origin_hashes=origin_done['artifacts'], source_sha256=PARENT_SHA256,
        source_frozen=True, independent_confirmation=False, full_stream=False,
        parent_exposed_to_all20=True, raw_hdf5_reads=0, suffix_label_reads=0,
        steps=256, seed=131, batch_size=24, learning_rate=.001, inner_rate=.1, radius=.1,
        primary='mature_teacher_mature_step', cells=CELLS,
        context_clock='origin128 plus matured25, unchanged from original',
        gradient_clock='origin versus maturity=origin+25; query=origin+50',
        update_use='predict through maturity, then use update only after maturity',
        controls='target-feedback replacement only; actual maturity input retained in mature cells',
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)},
        torch=str(torch.__version__), cuda_device=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json', protocol)
    _durable_json(output/'records.json', records)
    _durable_json(output/'folds.json', folds)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    started = time.monotonic()
    held = {cell: {} for cell in CELLS}
    fit_groups = []
    for spec in folds:
        prior = torch.load(ORIGIN/f"anchored_fold{spec['fold']}.pt", map_location='cpu', weights_only=True)
        if prior['spec'] != spec:
            raise ValueError('fold changed')
        mean, scale = prior['mean'], prior['scale']
        scaled = transform_context(data['context'], scale)
        donors = torch.tensor(read(ORIGIN/f"fold{spec['fold']}_donors.json"), dtype=torch.int64)
        # Reuse the sealed optimizer, initialization and sampling implementation.
        controllers = dict(mature=fit(net, slow, mature, scaled, mean, records, spec, output),
                           origin=AnchoredTarget().cuda())
        controllers['origin'].load_state_dict(prior['state_dict'], strict=True)
        controllers['origin'].eval()
        if read(output/f"anchored_fold{spec['fold']}_training.json")['draws'] != read(
                ORIGIN/f"anchored_fold{spec['fold']}_training.json")['draws']:
            raise AssertionError('teaching samples not paired')
        torch.save(dict(state_dict={k: v.cpu() for k, v in controllers['mature'].state_dict().items()},
                        mean=mean, scale=scale, spec=spec), output/f"mature_fold{spec['fold']}.pt")
        for cell in CELLS:
            teacher, inputs = choice(cell, controllers, data, mature)
            fit_predictions, fit_records = [], []
            for sid in spec['fit']+spec['held']:
                index = np.array([i for i, r in enumerate(records) if r['shot'] == sid])
                pred = evaluate(net, slow, teacher, 'anchored', inputs, scaled, mean, donors, index)
                with np.load(ORIGIN/f"anchored_fold{spec['fold']}_{sid}.npz", allow_pickle=False) as ref:
                    if not np.array_equal(pred['cold'], ref['cold']):
                        raise AssertionError('query baseline changed')
                np.savez_compressed(output/f"{cell}_fold{spec['fold']}_{sid}.npz", **pred)
                if sid in spec['held']:
                    held[cell][sid] = pred
                else:
                    fit_predictions.append(pred)
                    fit_records.extend(records[i] for i in index)
            packed = {k: np.concatenate([p[k] for p in fit_predictions]) for k in fit_predictions[0]}
            _, groups = shot_summary(fit_records, packed, ARMS)
            fit_groups.append(dict(fold=spec['fold'], cell=cell, groups=groups))
    ids = list(dict.fromkeys(r['shot'] for r in records))
    summaries = {}
    for cell in CELLS:
        packed = {k: np.concatenate([held[cell][s][k] for s in ids]) for k in held[cell][ids[0]]}
        rows, groups = shot_summary(records, packed, ARMS)
        valid = packed['direction_nonzero']
        cosine = float(np.median(packed['reverse_cos_real'][valid])) if valid.any() else None
        distinct = float((np.abs(packed['real'][valid]-packed['reverse'][valid]) > 1e-4).mean()) if valid.any() else 0.
        summaries[cell] = dict(groups=groups, gate=promotion(groups,
            origin_done['summaries']['anchored']['groups'], cosine, distinct),
            primary=cell == protocol['primary'])
        _durable_json(output/f'{cell}_held_per_shot.json', rows)
    _durable_json(output/'fit_groups.json', fit_groups)
    check_files(io23.ROOT, protocol['code_sha256'])
    check_files(ORIGIN, protocol['origin_hashes'])
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(
            not torch.equal(frozen[k], v.cpu()) for k, v in net.state_dict().items()):
        raise AssertionError('source changed')
    result = dict(complete=True, full_stream=False, source_unchanged=True,
        elapsed_seconds=time.monotonic()-started, opportunities=len(records),
        primary=protocol['primary'], summaries=summaries,
        artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    _durable_json(output/'complete.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output)
