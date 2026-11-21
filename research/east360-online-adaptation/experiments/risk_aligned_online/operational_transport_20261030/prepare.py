"""Teach a current-parent gradient rule only from the ended archived cold20."""
import argparse
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_forecast_20261023.learning import shot_weights
from ..conditional_likelihood_20261029.run import folds, read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .gradients import observed_gradients, cpu_tree

PREVIOUS = io23.ROOT/'probe_outputs/arrival_reference_east360_v1'
PREVIOUS_SHA = '57dbd7085fba526fb78acee65d14f1033bafc1e3e25d0d54dd7041413e641a35'


def choose_origins(times, flag, down):
    ends = np.arange(127, len(times)-50, 25)
    if flag:
        ends = ends[np.round((down-times[ends+26])*1000, 9) > 30]
    if not len(ends):
        return ends
    chosen = np.linspace(0, len(ends)-1, min(32, len(ends))).round().astype(int)
    return np.unique(np.concatenate((ends[chosen], ends[-4:])))


def task_losses(logits, times, flag, down, threshold):
    if logits.ndim != 1 or len(logits) != len(times) or not 0 < threshold < 1:
        raise ValueError('one logit per future time and probability threshold required')
    lead = np.round((down-times)*1000, 9) if flag else np.full(len(times), np.inf)
    valid = lead > 30 if flag else np.ones(len(times), bool)
    if not valid.any():
        raise ValueError('no timely supervised point')
    eligible = torch.as_tensor(np.flatnonzero(valid), device=logits.device)
    index = int(eligible[logits[eligible].detach().argmax()].item())
    margin = logits[index]-math.log(threshold/(1-threshold))
    point = float(flag and lead[index] <= 125)
    operational = F.binary_cross_entropy_with_logits(margin, margin.new_tensor(float(flag)))
    point_loss = F.binary_cross_entropy_with_logits(margin, margin.new_tensor(point))
    return operational, point_loss, index, point


def execute(output):
    from .learning import fit_rule, predict_rule
    if output.exists() or output.absolute().parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree probe output required')
    if sha(PREVIOUS/'complete.json') != PREVIOUS_SHA:
        raise ValueError('previous full-stream completion changed')
    previous = read(PREVIOUS/'protocol.json')
    reference = Path(previous['reference'])
    for name, digest in previous['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('reference changed: '+name)
    ids = previous['order'][:20]
    for i, sid in enumerate(ids, 1):
        stem = f'{i:04d}_{sid}'
        if sha(PREVIOUS/'shots'/(stem+'.npz')) != read(PREVIOUS/'shots'/(stem+'_seal.json'))['sha256']:
            raise ValueError('cold prediction was not sealed intact')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    slow_copy = slow.clone()
    mapping = read(reference/'input_map.json')
    hashes = read(reference/'input_hashes.json')
    output.mkdir()
    plan = io23.ROOT/'OPERATIONAL_TRANSPORT_PLAN_20261030.md'
    files = {plan, Path(__file__), Path(__file__).with_name('learning.py'),
             io23.ROOT/'tests/test_operational_transport_prepare.py',
             io23.ROOT/'tests/test_operational_transport_learning.py'}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='ENDED_COLD20_MATCHED_QK_GRADIENT_TEACHING',
        source_sha256=PARENT_SHA256, warmup_ids=ids, reference=str(reference),
        reference_hashes=previous['reference_hashes'], previous=str(PREVIOUS), previous_sha256=PREVIOUS_SHA,
        threshold=parent['threshold'], ridge=.1, radius=.05, source_updates=0,
        suffix_labels_accessed=False, conditional_parent_exposed_to_all20=True,
        independent_confirmation=False, plan_sha256=sha(plan),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)})
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(io23.ROOT)))
    started = time.monotonic()
    records, signals, operational, point, raw_risk = [], [], [], [], []
    for sid in ids:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('cold20 data changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        flag, down = io23.read_outcome('EAST', sid)
        sig = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        ends = choose_origins(raw['times'], flag, down)
        if not len(ends):
            raise ValueError('cold20 shot lacks a complete teaching window')
        for origin in ends:
            origin = int(origin)
            g, _, grisk, info = observed_gradients(net, {k: v[:origin+26] for k,v in sig.items()}, origin, slow)
            later = np.arange(origin+26, origin+51)
            x, mask = batch(sig, later, net.base.center.device)
            state = x.new_zeros(1, 8192, requires_grad=True)
            logits = net(x, mask, state.expand(len(x), -1), slow, 'qk', forecast=False)['logit']
            lop, lpoint, which, label = task_losses(logits, raw['times'][later], flag, down, parent['threshold'])
            go, = torch.autograd.grad(lop, state, retain_graph=True)
            gp, = torch.autograd.grad(lpoint, state)
            signals.append(g[0].cpu())
            operational.append(go.detach()[0].cpu())
            point.append(gp.detach()[0].cpu())
            raw_risk.append(grisk[0].cpu())
            records.append(dict(shot=sid, origin=origin, maturity=origin+25,
                query=int(later[which]), disruptive=bool(flag), point_label=label,
                task_loss=float(lop.detach()), point_loss=float(lpoint.detach()), **info))
        print(dict(stage='collect', shot=sid, pairs=len(ends), total=len(records), seconds=time.monotonic()-started), flush=True)
    data = dict(signal=torch.stack(signals), operational=torch.stack(operational), point=torch.stack(point),
                risk=torch.stack(raw_risk), records=records)
    torch.save(data, output/'pairs.pt')
    sid_array = np.array([r['shot'] for r in records])
    flags = np.array([r['disruptive'] for r in records], dtype=int)
    g = data['signal'].cuda()
    targets = {a: data[a].cuda() for a in ('operational', 'point')}
    fold_rows = []
    for spec in folds(ids):
        fit = np.isin(sid_array, spec['fit'])
        held = ~fit
        weights = torch.as_tensor(shot_weights(sid_array[fit], flags[fit]), device=g.device)
        for arm, target in targets.items():
            rule = fit_rule(g[fit], target[fit], weights, ridge=.1)
            mapped = predict_rule(g[held], rule)
            similarity = F.cosine_similarity(mapped, target[held], dim=1).cpu().numpy()
            for sid in spec['held']:
                selected = sid_array[held] == sid
                fold_rows.append(dict(fold=spec['fold'], shot=sid, arm=arm,
                    disruptive=bool(flags[held][selected][0]), examples=int(selected.sum()),
                    mean_cosine=float(similarity[selected].mean()), positive=int((similarity[selected]>0).sum())))
    weights = torch.as_tensor(shot_weights(sid_array, flags), device=g.device)
    order = np.arange(len(records))
    for sid in ids:
        indices = np.flatnonzero(sid_array == sid)
        order[indices] = np.roll(indices, len(indices)//2)
    targets['shuffled'] = targets['operational'][order]
    rules = {a: fit_rule(g, target, weights, ridge=.1) for a,target in targets.items()}
    torch.save(cpu_tree(dict(rules=rules, protocol=protocol, pairs_sha256=sha(output/'pairs.pt'))), output/'policy.pt')
    _durable_json(output/'fold_diagnostic.json', fold_rows)
    if (sha(DEFAULT_PARENT) != PARENT_SHA256 or not torch.equal(slow, slow_copy)
            or any(not torch.equal(v.cpu(), frozen[k]) for k,v in net.state_dict().items())):
        raise AssertionError('frozen source or cold state changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('source changed during experiment')
    result = dict(complete=True, shots=len(ids), examples=len(records),
        disruptive_examples=int(flags.sum()), safe_examples=int((1-flags).sum()),
        differing_labels=sum(r['disruptive'] and not r['point_label'] for r in records),
        zero_signal_gradients=int((g.norm(dim=1)==0).sum()), policy_sha256=sha(output/'policy.pt'),
        source_unchanged=True, seconds=time.monotonic()-started, full_stream_run=False)
    _durable_json(output/'complete.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.absolute())
