"""Teach the existing context-vector rule on historical J-TEXT TRAIN only."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ContextTarget
from .observation import observation
from .prepare import adapted_state, evaluate
from .source_episodes import sample_indices, select_records

GROUPS = ('safe', 'danger', 'early')
ACTIONS = ('real', 'reverse', 'mean_feedback', 'constant', 'risk_axis')
PLAN = io23.ROOT / 'CONTEXT_VECTOR_SOURCE_PLAN_20261030.md'


@contextmanager
def source_gate(ids):
    allowed = set(ids)
    readers = (io23.read_signals, io23.read_outcome)

    def checked(reader):
        def call(device, sid, *args, **kwargs):
            if device != 'J-TEXT' or sid not in allowed:
                raise ValueError('signal/outcome outside the declared source role')
            return reader(device, sid, *args, **kwargs)
        return call

    with patch.object(io23, 'read_signals', side_effect=checked(readers[0])), \
            patch.object(io23, 'read_outcome', side_effect=checked(readers[1])):
        yield


def move(data, index, device):
    return {k: v[index].to(device) for k, v in data.items()}


def train_step(net, controller, optimizer, data, labels):
    ids = {id(p) for p in controller.parameters()}
    selected = [id(p) for g in optimizer.param_groups for p in g['params']]
    if set(selected) != ids or len(selected) != len(ids) or any(p.requires_grad for p in net.parameters()):
        raise ValueError('only controller parameters may be trained')
    optimizer.zero_grad(set_to_none=True)
    state = adapted_state(net, controller, data, outer=True)
    z = net(data['later_x'], data['later_mask'], state, None, 'qk', forecast=False)['logit']
    loss = F.binary_cross_entropy_with_logits(z, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite source teaching objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), controller_gradient_norm=float(norm),
                max_state_norm=float(state.detach().norm(dim=1).max()))


def summarize(records, prediction):
    labels = np.array([r['label'] for r in records], np.float64)
    base = np.asarray(prediction['baseline'], np.float64)
    before = np.logaddexp(0., base) - labels * base
    rows, groups = [], {}
    for arm in ACTIONS:
        z = np.asarray(prediction[arm + '_logit'], np.float64)
        after = np.logaddexp(0., z) - labels * z
        groups[arm] = {}
        for group in GROUPS:
            entries = []
            for sid in sorted({r['shot'] for r in records if r['group'] == group}):
                take = np.array([r['shot'] == sid and r['group'] == group for r in records])
                item = dict(arm=arm, group=group, shot=int(sid), windows=int(take.sum()),
                            mean_before_bce=float(before[take].mean()),
                            mean_after_bce=float(after[take].mean()),
                            mean_delta_bce=float((after[take]-before[take]).mean()),
                            mean_logit_change=float((z[take]-base[take]).mean()),
                            mean_action_norm=float(prediction[arm + '_norm'][take].mean()),
                            mean_cos_real=float(prediction[arm + '_cos_real'][take].mean()))
                rows.append(item)
                entries.append(item)
            changes = [r['mean_delta_bce'] for r in entries]
            groups[arm][group] = dict(shots=len(entries), windows=sum(r['windows'] for r in entries),
                mean_delta_bce=float(np.mean(changes)) if entries else None,
                improved_shots=sum(v < -1e-6 for v in changes),
                worsened_shots=sum(v > 1e-6 for v in changes),
                mean_logit_change=float(np.mean([r['mean_logit_change'] for r in entries])) if entries else None)
    reasons = []
    if any(groups['real'][g]['shots'] < 10 for g in ('safe', 'danger')):
        reasons.append('insufficient_validation_shots')
    else:
        for group in ('safe', 'danger'):
            value = groups['real'][group]['mean_delta_bce']
            if value >= 0:
                reasons.append(group + ': real does not improve baseline')
            for arm in ('reverse', 'mean_feedback'):
                if value >= groups[arm][group]['mean_delta_bce']:
                    reasons.append(group + ': real does not outperform ' + arm)
    return dict(groups=groups, per_shot=rows, reasons=reasons, source_screen_passed=not reasons,
                not_shot_tpr_fpr=True, not_east_validation=True)


@torch.no_grad()
def source_scores(net, signal):
    logits = []
    for start in range(0, len(signal['values']), 64):
        ends = range(start, min(start+64, len(signal['values'])))
        x, mask = batch(signal, ends, net.base.center.device)
        state = x.new_zeros(len(x), 8192)
        z = net(x, mask, state, None, 'qk', forecast=False)['logit']
        logits.append(z.cpu().numpy())
    return np.concatenate(logits)


def collect(net, ids, advanced, output, role):
    records, population, chunks, hashes = [], [], [], {}
    with source_gate(ids):
        for index, sid in enumerate(ids):
            path = io23.path_for('J-TEXT', sid)
            digest = sha(path)
            shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
            signal = {k: shot[k] for k in ('values', 'available')}
            selected, counts = select_records(shot, source_scores(net, signal))
            population.append(dict(shot=sid, **counts))
            for record in selected:
                observed = observation(net, signal, record['origin'])
                if not observed['support'].item():
                    raise AssertionError('selected feedback support disagrees with actual context')
                x, mask = batch(signal, [record['query']], net.base.center.device)
                observed.pop('reference')
                observed.update(later_x=x, later_mask=mask)
                chunks.append({k: v.detach().cpu() for k, v in observed.items()})
            records.extend(selected)
            if sha(path) != digest:
                raise ValueError('source input changed during collection')
            hashes[str(sid)] = digest
            if (index+1) % 100 == 0 or index+1 == len(ids):
                print(dict(stage='collect', role=role, shots=index+1, records=len(records)), flush=True)
    if not chunks:
        raise ValueError('no eligible historical source episodes')
    data = {k: torch.cat([row[k] for row in chunks]) for k in chunks[0]}
    groups = {g: dict(shots=len({r['shot'] for r in records if r['group'] == g}),
                      windows=sum(r['group'] == g for r in records)) for g in GROUPS}
    _durable_json(output / f'{role}_population.json', dict(groups=groups, shots=population))
    _durable_json(output / f'{role}_records.json', records)
    _durable_json(output / f'{role}_input_hashes.json', hashes)
    return data, records, groups, hashes


def run(output):
    output = Path(output).absolute()
    if output.exists() or output.parent != io23.ROOT / 'probe_outputs':
        raise ValueError('fresh worktree probe_outputs destination required')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda()).eval().requires_grad_(False)
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    output.mkdir()
    files = {Path(__file__).resolve(), PLAN.resolve()}
    files.update(Path(m.__file__).resolve() for name, m in list(sys.modules.items())
                 if name.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    for name in ('test_context_vector_source_run.py', 'test_context_vector_source_episodes.py'):
        files.add(io23.ROOT / 'tests' / name)
    code = {str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)}
    protocol = dict(scope='JTEXT_TRAIN_CONTEXT_VECTOR_TEACHING_NOT_JTEXT_ONLINE',
        source_sha256=PARENT_SHA256, train_ids=roles['train'], val_ids=roles['val'],
        test_signals_read=0, east_signals_read=0, label='max(25ms, AdvancedTime)',
        query_lead_strict_ms=10., first_origin=127, maturity_offset=25,
        query_offset=[26, 50], max_per_group_per_shot=8, selection='fixed parent peak per interval',
        steps=512, batch_size=12, per_class=6, optimizer='Adam', lr=.001, seed=131,
        gradient_clip=1., inner_learning_rate=.1, qk_radius=.1,
        source_slow=None, source_input='original simple_data, no new input contract',
        threshold_unchanged=parent['threshold'], final_checkpoint_only=True,
        code_sha256=code, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__))
    _durable_json(output / 'protocol.json', protocol)
    with zipfile.ZipFile(output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    started = time.monotonic()
    train, train_records, train_groups, train_hashes = collect(net, roles['train'],
        {sid: advanced[sid] for sid in roles['train'] if sid in advanced}, output, 'train')
    if any(train_groups[g]['shots'] == 0 for g in ('safe', 'danger')):
        raise ValueError('source teaching requires both safe and danger populations')
    sid_array = np.array([r['shot'] for r in train_records])
    mean = torch.stack([train['context'][sid_array == sid].mean(0) for sid in np.unique(sid_array)]).mean(0)
    torch.manual_seed(131)
    controller = ContextTarget().cuda()
    optimizer = torch.optim.Adam(controller.parameters(), lr=.001)
    rng = np.random.default_rng(131)
    history, sampled = [], []
    for step in range(512):
        take = sample_indices(train_records, rng, per_class=6)
        labels = torch.tensor([train_records[i]['label'] for i in take], device='cuda')
        row = train_step(net, controller, optimizer, move(train, take, 'cuda'), labels)
        sampled.append(take)
        history.append(dict(step=step+1, **row))
        if (step+1) % 64 == 0:
            print(dict(stage='fit', step=step+1, **row), flush=True)
            _durable_json(output / 'progress.json', history[-1])
    controller.eval().requires_grad_(False)
    torch.save(dict(state={k: v.detach().cpu() for k, v in controller.state_dict().items()},
                    mean_context=mean, protocol=protocol), output / 'controller.pt')
    _durable_json(output / 'training.json', dict(history=history, sampled_indices=sampled))
    del train
    val, val_records, val_groups, val_hashes = collect(net, roles['val'],
        {sid: advanced[sid] for sid in roles['val'] if sid in advanced}, output, 'val')
    prediction = evaluate(net, None, controller, move(val, slice(None), 'cuda'), mean.cuda())
    np.savez_compressed(output / 'validation_arrays.npz', **prediction)
    result = summarize(val_records, prediction)
    _durable_json(output / 'validation.json', result)
    if (any(not torch.equal(net.state_dict()[k].detach().cpu(), value) for k, value in frozen.items())
            or sha(DEFAULT_PARENT) != PARENT_SHA256):
        raise AssertionError('qualified source changed')
    if any(sha(io23.ROOT / name) != digest for name, digest in code.items()):
        raise AssertionError('code changed during source teaching')
    if any(sha(io23.path_for('J-TEXT', int(sid))) != digest
           for sid, digest in {**train_hashes, **val_hashes}.items()):
        raise AssertionError('source raw data changed')
    artifacts = [p for p in output.iterdir() if p.is_file()]
    complete = dict(complete=True, source_unchanged=True, scientific_goal_achieved=False,
        source_screen_passed=result['source_screen_passed'], reasons=result['reasons'],
        seconds=time.monotonic()-started, train_groups=train_groups, val_groups=val_groups,
        test_signals_read=0, east_signals_read=0, independent_confirmation=False,
        artifact_sha256={p.name: sha(p) for p in artifacts})
    _durable_json(output / 'complete.json', complete)
    print(result['groups'], flush=True)
    print(complete, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
