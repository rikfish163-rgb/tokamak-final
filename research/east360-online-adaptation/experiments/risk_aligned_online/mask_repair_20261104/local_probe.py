"""VAL-only single-step counterfactual: not a J-TEXT online stream."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import sha, write
from ..mask_separated_intra_20261103.run import state_digest
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from . import core, intra
from .evaluation import FIT, load_fit

PLAN = io23.ROOT / 'SOURCE_MASK_REPAIR_INTRA_PLAN_20261104.md'


def queries(length):
    if type(length) is not int or length < 0:
        raise ValueError('nonnegative recorded length required')
    grid = np.arange(129, length - 26, 26)
    if not len(grid):
        return []
    idx = np.linspace(0, len(grid) - 1, min(5, len(grid))).astype(int)
    return sorted(set(map(int, grid[idx])) | set(map(int, grid[-3:])))


def classify_endpoint(disruptive, down, horizon, time_s):
    if not disruptive:
        return 'safe', 0
    lead = round((down - time_s) * 1000, 9)
    if lead < -1e-6:
        return 'postterminal', None
    if lead > horizon + 1e-6:
        return 'earlier_disruption', 0
    return ('timely_positive' if lead > 10 else 'late_positive'), 1


def summarize(rows):
    result = {}
    for mode in intra.MODES:
        result[mode] = {}
        for group in ('safe', 'timely_positive', 'late_positive', 'earlier_disruption'):
            selected = [r for r in rows if r['mode'] == mode and r['group'] == group and r['supported']]
            shots = defaultdict(list)
            for row in selected:
                shots[row['shot']].append(row)
            values = {}
            for key in ('risk_delta', 'held_delta', 'reference_delta'):
                by_shot = [float(np.mean([r[key] for r in rs])) for rs in shots.values()]
                values[key] = float(np.mean(by_shot)) if by_shot else None
            result[mode][group] = dict(shots=len(shots), windows=len(selected), **values)
    checks = {}
    for group in ('safe', 'timely_positive'):
        real = result['real'][group]
        checks[group] = bool(
            real['shots'] >= 10 and real['risk_delta'] is not None
            and real['risk_delta'] < 0 and real['held_delta'] < 0 and real['reference_delta'] <= 0
            and all(result[m][group]['risk_delta'] is not None
                    and real['risk_delta'] < result[m][group]['risk_delta'] - 1e-5
                    for m in intra.MODES if m not in ('real', 'no_update')))
    return dict(groups=result, gate_checks=checks, gate_passed=all(checks.values()),
                gate_is_development_mechanism_not_online_gain=True)


def main(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    ids = io23.source_roles()['val']
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    persistent, _, _ = load_fit(FIT, expected_source_sha=PARENT_SHA256, device='cuda')
    original = dict(source=state_digest(base), repair=state_digest(persistent))
    threshold = float(parent['threshold'])
    logit_cutoff = math.log(threshold / (1 - threshold))
    code = imported_code_hashes()
    bound = {str(p): sha(p) for p in (PLAN, DEFAULT_PARENT, io23.JTEXT_INDEX, FIT / 'completed.json')}
    output.mkdir(parents=True)
    (output / 'shots').mkdir()
    write(output / 'protocol.json', dict(
        scope='OFFLINE_SOURCE_VAL_SINGLE_STEP_NOT_JTEXT_ONLINE_OR_EAST_GAIN',
        source_sha256=PARENT_SHA256, source_state_sha256=original,
        threshold=threshold, source_roles='val160 only', modes=list(intra.MODES),
        independent_episodes=True, temporary_state_dim=16, temporary_step=.025,
        temporary_radius=.25, coordinate_selection_uses_record_length_but_not_outcome=True,
        coordinate_selection_is_not_deployable_policy=True, future_offset_samples=26,
        hard_outcome_after_seal=True, source_original_point_labels=True,
        source_complete_MNM_used_only_for_offline_reference=True,
        no_risk_gradient_in_update=True, data_previously_used_in_development=True,
        input_sha256=bound, code_sha256=code))
    with zipfile.ZipFile(output / 'code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in code:
            archive.write(p, str(Path(p).relative_to(io23.ROOT)))
        archive.write(PLAN, PLAN.name)
    rows, omissions, hashes, seals = [], [], {}, []
    reader = io23.read_signals
    for index, sid in enumerate(ids, 1):
        path = io23.path_for('J-TEXT', sid)
        hashes[str(sid)] = sha(path)
        def checked(device, shot, *args, **kwargs):
            if device != 'J-TEXT' or shot != sid:
                raise ValueError('only the current source VAL signal allowed')
            return reader(device, shot, *args, **kwargs)
        local = []
        with (patch.object(io23, 'read_signals', side_effect=checked),
              patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome forbidden before seal'))):
            raw = simple_data.read_shot('J-TEXT', sid)
            endpoints = queries(len(raw['values']))
            if not endpoints:
                omissions.append(dict(shot=sid, reason='no_full_prefix_plus_future_endpoint'))
            for t in endpoints:
                xv, av = simple_data.window(raw, t, 128)
                x = torch.as_tensor(xv[None], device='cuda')
                a = torch.as_tensor(av[None], device='cuda')
                a[:, :, 2] = False
                q = t + 26
                future_x, full_a = simple_data.window(raw, q, 128)
                fx = torch.as_tensor(future_x[None], device='cuda')
                full = torch.as_tensor(full_a[None], device='cuda')
                masked = full.clone()
                masked[:, :, 2] = False
                with torch.no_grad():
                    baseline = core.forward(base, persistent, fx, masked)
                    reference = core.forward(base, persistent, fx, full, enabled=False)['embedding_after']
                    supported = full.reshape(1, 16, 8, 23).any(-1).any(-1)
                    before_error = float((baseline['embedding_after'][supported] - reference[supported]).square().mean())
                for mode in intra.MODES:
                    state, info = intra.step(base, persistent, intra.initial_state(persistent), x, a,
                                             mode=mode, seed=131 + sid * 100000 + t)
                    with torch.no_grad():
                        after = core.forward(base, intra.LocalRepair(persistent, state), fx, masked)
                        after_error = float((after['embedding_after'][supported] - reference[supported]).square().mean())
                    local.append(dict(shot=sid, update=t, query=q, time_s=float(raw['times'][q]),
                                      before_logit=float(baseline['logit']), after_logit=float(after['logit']),
                                      reference_delta=after_error - before_error,
                                      held_delta=info.get('held_real_after', 0) - info.get('held_real_before', 0),
                                      supported=info['status'] != 'INSUFFICIENT_OBSERVATIONS', **info))
            prediction = output / 'shots' / f'{index:04d}_{sid}_prediction.json'
            write(prediction, local)
            seal = dict(shot=sid, prediction_sha256=sha(prediction), current_outcome_used_in_prediction=False)
            write(prediction.with_name(prediction.stem + '_seal.json'), seal)
            seals.append(seal)
        disruptive, down = io23.read_outcome('J-TEXT', sid)
        horizon = io23.effective_horizon_ms(sid, io23.advanced_times())[0]
        if sha(prediction) != seal['prediction_sha256'] or sha(path) != hashes[str(sid)]:
            raise RuntimeError('prediction or source input changed')
        for row in local:
            group, label = classify_endpoint(disruptive, down, horizon, row['time_s'])
            before, after = row['before_logit'] - logit_cutoff, row['after_logit'] - logit_cutoff
            risk = None if label is None else float(np.logaddexp(0, after) - label * after
                                                   - np.logaddexp(0, before) + label * before)
            rows.append(dict(row, group=group, label=label, risk_delta=risk))
        if index % 20 == 0 or index == len(ids):
            print(dict(stage='single_step_VAL', shots=index, records=len(rows), seconds=time.monotonic() - started), flush=True)
    if (dict(source=state_digest(base), repair=state_digest(persistent)) != original
            or any(p.grad is not None for p in (*base.parameters(), *persistent.parameters()))):
        raise AssertionError('source or persistent repair changed')
    if any(sha(Path(p)) != h for p, h in {**code, **bound}.items()):
        raise AssertionError('producer inputs changed during diagnostic')
    summary = summarize(rows)
    write(output / 'rows.json', rows)
    write(output / 'summary.json', summary)
    write(output / 'raw_hashes.json', hashes)
    write(output / 'omissions.json', omissions)
    result = dict(status='COMPLETED_SOURCE_VAL_LOCAL_NEURAL_MECHANISM_NOT_ONLINE_GAIN',
                  seconds=time.monotonic() - started, rows=len(rows), source_signal_reads=len(ids),
                  test_signal_reads=0, east_signal_reads=0, source_and_persistent_unchanged=True,
                  gate_passed=summary['gate_passed'], seals=seals,
                  artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    write(output / 'completed.json', result)
    print(dict(status=result['status'], seconds=result['seconds'], gate=summary['gate_checks']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
