"""Offline loss-direction feasibility on20 preparation +10 development shots."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import make_parent, READ
from ..heldout_threshold_20261023.run import models_for_calibration, REFERENCE, CALIBRATION_IDS
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..trajectory_memory_20261024.run import signal
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many
from .learning import collect_geometry, finite_step, solve_weights

PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/通道反馈方向可行性_事前约定_20261024.md'
PAIR = io23.ROOT/'experiments/risk_aligned_online/runs/within_shot_pair360_20261024'


def select_queries(times, flag, down, scores):
    if len(times) != len(scores) or len(times) <= 280:
        raise ValueError('aligned full trajectory with mature feedback required')
    valid = np.arange(len(times)) >= 280
    if flag:
        lead = np.round((down-times)*1000., 9)
        valid &= (lead > 30.) & (lead <= 125.)
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise ValueError('no declared future evaluation endpoints')
    chosen = candidates[np.linspace(0, len(candidates)-1, 8).astype(int)].tolist()
    if not flag:
        chosen.append(int(candidates[np.argmax(scores[candidates])]))
    return sorted(set(chosen))


def evaluate_directions(model, state, record, weights=None):
    a, b = np.array(record['risk_alignment']), np.array(record['signal_alignment'])
    w, lp = solve_weights(a[None], b[None])
    directions = dict(uniform=state['signal_gradient']/23,
                      oracle=torch.as_tensor(w, device=state['parts'].device, dtype=state['parts'].dtype)@state['parts'],
                      direct_risk=state['risk_gradient'])
    if weights is not None:
        directions['historical'] = torch.as_tensor(weights, device=state['parts'].device, dtype=state['parts'].dtype)@state['parts']
    finite = {arm:[finite_step(model, state, d, length) for length in (.001,.01)] for arm,d in directions.items()}
    return dict(oracle_weights=w.tolist(), oracle_lp=lp, finite=finite)


def summarize(rows):
    out = {}
    for group in ('preparation', 'calibration'):
        out[group] = {}
        for flag in (False, True):
            selected = [r for r in rows if r['group'] == group and r['is_disrupt'] == flag]
            shots = sorted({r['shot'] for r in selected})
            arms = {}
            for arm in ('uniform', 'oracle', 'historical', 'direct_risk'):
                arms[arm] = {}
                for length in (.001, .01):
                    pairs = [(r,next(s for s in r['finite'][arm] if s['step_norm']==length)) for r in selected]
                    metrics = {}
                    for name, getter in {
                        'risk_delta':lambda r,s:s['risk_after']-r['risk_before'],
                        'signal_delta':lambda r,s:s['signal_after']-r['signal_before'],
                        'joint_improvement_fraction':lambda r,s:float(s['risk_after']<r['risk_before']-1e-7 and s['signal_after']<r['signal_before']-1e-7),
                        'risk_improvement_fraction':lambda r,s:float(s['risk_after']<r['risk_before']-1e-7),
                        'zero_direction_fraction':lambda r,s:float(s['zero_direction']),
                    }.items():
                        by_shot = [np.mean([getter(r,s) for r,s in pairs if r['shot']==sid]) for sid in shots]
                        metrics[name] = float(np.mean(by_shot))
                    arms[arm][str(length)] = metrics
            out[group]['disruptive' if flag else 'safe'] = dict(shots=len(shots), probes=len(selected),
                oracle_positive_margin_fraction=float(np.mean([np.mean([r['oracle_lp']['margin']>1e-8 for r in selected if r['shot']==sid]) for sid in shots])),
                arms=arms)
    return out


def main(root):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True); torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic(); old = READ(PAIR/'protocol.json')
    history = old['preparation_ids']; assert old['calibration_ids'] == CALIBRATION_IDS
    assert not set(history+CALIBRATION_IDS)&set(old['shot_ids'])
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt') == old['parent_sha256']
    assert sha(REFERENCE/'input_map.json') == old['mapping_sha256']
    parent, _ = make_parent(); model = models_for_calibration(parent)['mapped_fixed']
    original = state_cpu(model); mapping = READ(REFERENCE/'input_map.json')
    root.mkdir(parents=True); (root/'probes').mkdir()
    files = set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_feedback_cone_feasibility.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):
            files.add(Path(module.__file__))
    protocol = dict(preparation_ids=history, calibration_ids=CALIBRATION_IDS, excluded360_ids=old['shot_ids'],
        source_sha256=PARENT_SHA256, parent_sha256=old['parent_sha256'], mapping_sha256=old['mapping_sha256'],
        threshold_unchanged=old['threshold'], labels='OFFLINE ORACLE sampling and gradients, not deployable online inputs',
        shared_weights_fit='20 preparation shots only; sealed before calibration gradients',
        plan_sha256=sha(PLAN), code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in history+CALIBRATION_IDS},
        parameter_names=model.encoder_names(), step_norms=[.001,.01], query_lag_samples=25,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), independent_confirmation=False,
        scope='DIAGNOSTIC_ONLY_NOT_ONLINE_PERFORMANCE; no360,nointer,noalarm',
        backend='gradient-enabled same functional math path for pre/post loss',
        event_labels_used_in_this_diagnostic=False)
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files): z.write(f,str(f.relative_to(io23.ROOT)))
    rows = []; weights = None; shared_info = None; selections = []
    for group, ids in (('preparation',history), ('calibration',CALIBRATION_IDS)):
        for sid in ids:
            if time.monotonic()-start > 900:
                raise TimeoutError('declared15minute limit exceeded')
            raw = signal(sid); mapped = transform({k:raw[k] for k in ('values','available')},mapping)
            scores = predict_many(model,dict(parent=mapped),device='cuda')['parent']
            flag, down = io23.read_outcome('EAST',sid)
            queries = select_queries(raw['times'],flag,down,scores)
            selections.append(dict(shot=sid,group=group,is_disrupt=bool(flag),down=down,queries=queries))
            for query in queries:
                record, state = collect_geometry(model,mapped,query,float(flag),device='cuda')
                record.update(shot=sid,group=group,is_disrupt=bool(flag),lead_ms=float((down-raw['times'][query])*1000) if flag else None)
                record.update(evaluate_directions(model,state,record,weights))
                rows.append(record)
            _durable_json(root/'probes'/f'{sid}.json',[r for r in rows if r['shot']==sid])
            print(dict(stage=group,shot=sid,probes=len(queries),seconds=round(time.monotonic()-start,2)),flush=True)
        if group == 'preparation':
            weights, shared_info = solve_weights(np.array([r['risk_alignment'] for r in rows]),np.array([r['signal_alignment'] for r in rows]))
            _durable_json(root/'shared_weights.json',dict(weights=weights.tolist(),lp=shared_info,history_ids=history,
                calibration_labels_used_for_fit=False,calibration_gradients_computed=0))
            _durable_json(root/'shared_weight_seal.json',dict(sha256=sha(root/'shared_weights.json')))
    # Recompute historical gradients only for finite-step verification of the
    # now fitted weights. Calibration rows never enter the LP fit.
    for sid in history:
        raw = signal(sid); mapped = transform({k:raw[k] for k in ('values','available')},mapping)
        for r in [v for v in rows if v['shot']==sid]:
            fresh,state = collect_geometry(model,mapped,r['query'],r['label'],device='cuda')
            for key in ('risk_alignment','signal_alignment','risk_before','signal_before'):
                np.testing.assert_array_equal(fresh[key],r[key])
            direction = torch.as_tensor(weights,dtype=state['parts'].dtype,device='cuda')@state['parts']
            r['finite']['historical'] = [finite_step(model,state,direction,v) for v in (.001,.01)]
        _durable_json(root/'probes'/f'{sid}.json',[r for r in rows if r['shot']==sid])
        if time.monotonic()-start > 900:
            raise TimeoutError('declared15minute limit exceeded')
    assert_unchanged(model,original); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    _durable_json(root/'selections.json',selections)
    result = summarize(rows); _durable_json(root/'summary.json',result)
    completed = dict(status='completed',seconds=time.monotonic()-start,probes=len(rows),summary=result,
        shared_lp=shared_info,source_and_parent_unchanged=True,independent_confirmation=False,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',completed)
    print(dict(stage='completed',seconds=completed['seconds'],probes=len(rows),shared_lp=shared_info),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
