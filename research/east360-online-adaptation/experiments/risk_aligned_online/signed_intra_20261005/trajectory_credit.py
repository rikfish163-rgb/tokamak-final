"""Bounded historical trajectory feedback; not a replacement online runner.

Report24: retain source/direction/actions, replace single-event teacher CE
with paired independent trajectory REINFORCE on actual first-alarm cost.
All32 historical outcomes were exposed previously. No future32 execution.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import simple_data
from ..simple_online import _durable_json, _hash, seal_predictions
from ..redesign_20261004.prepare_prior import (
    DEFAULT_PARENT, PARENT_SHA256, load_parent, classifier_state, assert_classifier_unchanged,
)
from ..ffn_intra_20261005.model import FfnModel
from .context import VisibleContext, ACTIONS, advance_q
from .finite_probe import reference_order, whole_cost
from .policy import plan, score_plan, history_partition, summarize, guard


def sampled_plan(signals, rule, center, scale, rng):
    """Sample prefix-local actions without outcomes, lengths or future statistics.

The loop observes the saved stream progressively; final length only ends replay.
Independent RNG draws per rollout; previous actions enter context through q.
"""
    if set(signals) != {'device','values','available','times'}:
        raise ValueError('signal-only fields required')
    x, mask = np.asarray(signals['values']), np.asarray(signals['available'])
    if (signals['device'] != 'EAST' or x.ndim != 2 or x.shape[1] != 23
            or mask.shape != x.shape or mask.dtype != bool):
        raise ValueError('EAST flat23 signals and boolean mask required')
    state = VisibleContext('H'); q = np.empty(len(x), np.float64); events = []; start = 0
    for origin in range(0, len(x), 26):
        for offset, segment in ((5,'early'),(25,'late')):
            end = origin + offset
            if end >= len(x): continue
            q[start:end+1] = state.q; start = end+1
            first = origin+1 if segment == 'early' else origin+5
            before = state.q
            context, _ = state.step(torch.as_tensor(x[first:end+1].copy(),dtype=rule.dtype),
                torch.as_tensor(mask[first:end+1].copy()),center,scale,segment,rule)
            if context is None: continue
            probabilities = (rule.detach() @ context).softmax(0).cpu().numpy()
            action = ACTIONS[int(rng.choice(3,p=probabilities))]
            state.q = advance_q(before,action,segment)
            events.append(dict(end=end,effect=end+1 if end+1 < len(x) else None,
                segment=segment,context=context.tolist(),action=action,
                probabilities=probabilities.tolist(),q_before=before,q_after=state.q))
    q[start:] = state.q
    return q,events


def first_alarm_index(shot, scores, threshold):
    """Historical evaluation only; original valid mask and first crossing rule."""
    valid = np.asarray(shot['valid'],bool).copy()
    if shot['is_disrupt']: valid &= np.asarray(shot['times']) <= shot['down']
    hits = np.flatnonzero(valid & (np.asarray(scores) >= threshold))
    return int(hits[0]) if len(hits) else None


def trajectory_log_probability(rule, events, first_alarm):
    """Credit actions effective by absorbing first alarm; don't credit post-alarm.

q in saved context is conditional on past sampled discrete actions. It must be
detached, not differentiated through the hard action path (score-function PG).
No alarm: include all actions with an actual subsequent prediction endpoint.
"""
    used = [e for e in events if e['effect'] is not None
            and (first_alarm is None or e['effect'] <= first_alarm)]
    if not used: return rule.sum()*0
    context = torch.tensor([e['context'] for e in used],dtype=rule.dtype,device=rule.device)
    actions = torch.tensor([ACTIONS.index(e['action']) for e in used],device=rule.device)
    logits = context @ rule.T
    return logits.log_softmax(1).gather(1,actions[:,None]).sum()


def paired_loss(rule, left, right):
    """Mean two-trajectory loss, other independent return as baseline.

Cost=1 iff FP or missing strict timely warning. Rewards=-cost, no score proxy.
Equal outcomes give exactly zero; caller skips Adam if both classes tie so
optimizer momentum cannot create an update unsupported by this batch.
"""
    if any(type(t['cost']) is not int or t['cost'] not in (0,1) for t in (left,right)):
        raise ValueError('binary integer first-alarm cost required')
    advantage = right['cost']-left['cost']
    if advantage == 0: return rule.sum()*0
    return -.5*advantage*(trajectory_log_probability(rule,left['events'],left['first_alarm'])
                         -trajectory_log_probability(rule,right['events'],right['first_alarm']))


def load_audit_after_credit(contrast, items, loader):
    """Do not expose audit waveforms after a known training-credit failure.

Added after independent review26; the archived first run did execute its audit
despite zero disruptive contrast. Preserve that run rather than rewrite it.
"""
    if len(contrast)!=2 or any(type(n) is not int or n<0 for n in contrast):
        raise ValueError('two nonnegative class contrast counts required')
    return [loader(item) for item in items] if all(n>0 for n in contrast) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('direction','history','static','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131); started = time.monotonic()
    # Separate child streams give independently sampled rollouts conditional on
    # the same frozen B. No action is selected from its own outcome or peer.
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(131).spawn(2)]
    order = reference_order(args.static)[:32]
    # Identity contract, not summary metrics containing already-exposed audit
    # outcomes. summary is hashed below without interpreting its content.
    history = json.loads((args.history/'protocol.json').read_text())
    direction_summary = json.loads((args.direction.parent/'summary.json').read_text())
    if (_hash(args.direction) != direction_summary['direction_sha256']
            or history['direction_sha256'] != _hash(args.direction)
            or history['shot_ids'] != [r['shot'] for r in order]
            or history['parent_sha256'] != PARENT_SHA256):
        raise ValueError('same first32 cold-start and direction assets required')
    asset = torch.load(args.direction,map_location='cpu',weights_only=True)
    if asset['protocol']['parent_sha256'] != PARENT_SHA256 or not asset['protocol']['source_only']:
        raise ValueError('fixed source-only direction required')
    base, checkpoint = load_parent(DEFAULT_PARENT); base.cuda(); model = FfnModel(base)
    frozen_source = classifier_state(base); direction = asset['direction'].to(model.center)
    if (direction.shape != (model.delta_count,) or not torch.isfinite(direction).all()
            or abs(float(direction.double().norm())-1) > 1e-6):
        raise ValueError('unit fixed FFN direction required')
    threshold = float(checkpoint['threshold'])
    center, scale = model.center.detach().cpu().double(),model.scale.detach().cpu().double()
    fit_items,audit_items = history_partition(order)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'trajectories').mkdir()
    root = Path(__file__).parents[1]
    files = [p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')]
    files += [root/name for name in ('simple23.py','simple_data.py','io23.py','simple_train.py',
        'simple_online.py','redesign_20261004/intra.py','redesign_20261004/prepare_prior.py',
        'ffn_intra_20261005/model.py')]
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,str(path.relative_to(root)))
    protocol = dict(parent_sha256=PARENT_SHA256,direction_sha256=_hash(args.direction),
        history_summary_sha256=_hash(args.history/'summary.json'),threshold=threshold,
        fit_ids=[r['shot'] for r in fit_items],audit_ids=[r['shot'] for r in audit_items],
        steps=32,training_rollouts=128,seed=131,lr=3e-5,optimizer='Adam, betas=.9/.999, eps=1e-8, wd=0',
        target='paired independent trajectory first-alarm error; no risk-delta reward',
        zero_advantage='skip optimizer step including momentum',
        historical_only=True,future_access=False,admitted_online_model=False,
        actor='same3x117 rule, same ±.02/.08 actions, q±1, original source-only direction',
        evidence='DEVELOPMENT_ONLY; retrospective historical replay, no rewriting original cold-start outcomes',
        hardware=torch.cuda.get_device_name(),code_sha256={str(p.resolve()):_hash(p) for p in files})
    _durable_json(args.output/'protocol.json',protocol)
    zero = torch.zeros(3,117,dtype=torch.float64); parity = []

    def signals_only(shot):
        return {k:shot[k] for k in ('device','values','available','times')}

    def load_historical(item):
        index = next(i for i,r in enumerate(order,1) if r['shot'] == item['shot'])
        folder = args.history/'0'; stem = f'{index:04d}_{item["shot"]}'
        receipt = json.loads((folder/f'{stem}_seal.json').read_text())
        if (receipt['shot'] != item['shot'] or receipt['threshold'] != threshold
                or _hash(folder/receipt['npz']) != receipt['sha256']
                or _hash(Path(item['provenance']['path'])) != item['provenance']['sha256']):
            raise ValueError('historical seal/raw input identity changed')
        signals = simple_data.read_shot('EAST',item['shot'],rate_contract='nominal')
        q,_ = plan(signals_only(signals),zero,center,scale)
        _,score = score_plan(model,signals,direction,q)
        with np.load(folder/receipt['npz'],allow_pickle=False) as saved:
            if not np.array_equal(saved['times'],signals['times']): raise ValueError('time grid changed')
            diff = float(np.max(np.abs(score-saved['scores'])))
            if diff > 3e-6 or np.any((score>=threshold)!=(saved['scores']>=threshold)):
                raise RuntimeError('zero-state parity failed')
        parity.append(dict(shot=item['shot'],max_abs=diff))
        return simple_data.attach_outcome(signals,'EAST',{}),score

    fit = [load_historical(item)[0] for item in fit_items]
    by_class = [[s for s in fit if int(s['is_disrupt'])==k] for k in (0,1)]
    if any(not rows for rows in by_class): raise ValueError('fixed fit must contain both classes')
    rule = zero.clone().requires_grad_(True); optimizer = torch.optim.Adam([rule],lr=3e-5)
    log = []; contrast = [0,0]; applied = 0; trace_index = 0
    for iteration in range(32):
        frozen = rule.detach().clone(); pairs = []; examples = []
        for label in (0,1):
            shot = by_class[label][iteration % len(by_class[label])]; pair=[]
            for rng in rngs:
                q,events = sampled_plan(signals_only(shot),frozen,center,scale,rng)
                logits,scores = score_plan(model,shot,direction,q)
                _,cost,prediction = whole_cost(shot,scores,threshold)
                alarm = first_alarm_index(shot,scores,threshold)
                pair.append(dict(events=events,cost=cost[0],first_alarm=alarm))
                trace_index += 1; dest = args.output/'trajectories'
                receipt = seal_predictions(dest,trace_index,shot['shot'],shot['times'],scores,
                    arm='historical_stochastic_replay',threshold=threshold)
                state_path = dest/f'{trace_index:04d}_{shot["shot"]}_actions.npz'
                np.savez_compressed(state_path,q=q,logits=logits,
                    contexts=np.asarray([e['context'] for e in events],dtype=np.float64).reshape(-1,117),
                    actions=np.asarray([e['action'] for e in events],dtype=np.int8),
                    effects=np.asarray([-1 if e['effect'] is None else e['effect'] for e in events]),
                    probabilities=np.asarray([e['probabilities'] for e in events]).reshape(-1,3))
                examples.append(dict(shot=shot['shot'],label=label,cost=cost[0],risk=cost[1],
                    prediction=prediction,first_alarm_index=alarm,events=len(events),
                    credited_events=sum(e['effect'] is not None and (alarm is None or e['effect']<=alarm) for e in events),
                    q_min=float(q.min()),q_max=float(q.max()),q_final=float(q[-1]),
                    seal_sha256=receipt['sha256'],actions_file=state_path.name,actions_sha256=_hash(state_path)))
            contrast[label] += int(pair[0]['cost'] != pair[1]['cost']); pairs.append(pair)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([paired_loss(rule,*pair) for pair in pairs]).mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite trajectory loss')
        has_contrast = any(pair[0]['cost'] != pair[1]['cost'] for pair in pairs)
        grad_norm = 0.
        if has_contrast:
            loss.backward()
            if not torch.isfinite(rule.grad).all(): raise FloatingPointError('nonfinite policy gradient')
            grad_norm = float(rule.grad.norm()); optimizer.step(); applied += 1
        log.append(dict(iteration=iteration+1,loss=float(loss.detach()),optimizer_step=has_contrast,
                        gradient_norm=grad_norm,examples=examples))
        _durable_json(args.output/'training.json',log)
        if (iteration+1)%4==0:
            print(json.dumps(dict(stage='trajectory_fit',pair=iteration+1,contrast=contrast,
                optimizer_steps=applied,seconds=time.monotonic()-started)),flush=True)
    candidate = rule.detach().clone()
    torch.save(dict(rule=candidate,protocol=protocol,admitted=False),args.output/'candidate.pt')
    # Audit waveform/outcomes are loaded only after the single candidate freezes.
    audit_data = load_audit_after_credit(contrast,audit_items,load_historical)
    if audit_data is None:
        assert_classifier_unchanged(base,frozen_source)
        if _hash(DEFAULT_PARENT)!=PARENT_SHA256: raise RuntimeError('qualified source file changed')
        result=dict(protocol=protocol,checks=dict(both_classes_had_task_contrast=False),
            stopped_at='training_task_credit',audit=None,audit_waveform_access=False,
            passed_development_checks=False,class_contrast_pairs=contrast,optimizer_steps=applied,
            training_rollouts=trace_index,rule_norm=float(candidate.norm()),parity=parity,
            classifier_unchanged=True,admitted=False,future_run=False,
            elapsed_seconds=time.monotonic()-started)
        _durable_json(args.output/'summary.json',result)
        print(json.dumps(dict(stage='trajectory_stopped',reason='missing_class_task_credit',
            contrast=contrast,audit_run=False,seconds=result['elapsed_seconds'])),flush=True)
        return
    rows={a:[] for a in ('S','H','P','L')}; costs={a:[] for a in rows}; records=[]
    for arm in rows: (args.output/arm).mkdir()
    for index,(shot,static_scores) in enumerate(audit_data,1):
        predictions={}
        for arm in rows:
            q,events = plan(signals_only(shot),zero if arm=='S' else candidate,center,scale,
                            arm='H' if arm=='S' else arm)
            _,scores = score_plan(model,shot,direction,q) if arm!='S' else (None,static_scores)
            receipt=seal_predictions(args.output/arm,index,shot['shot'],shot['times'],scores,
                arm='historical_audit_'+arm,threshold=threshold)
            row,cost,pred = whole_cost(shot,scores,threshold)
            rows[arm].append(row); costs[arm].append(cost[1]); predictions[arm]=dict(
                prediction=pred,risk=cost[1],q_min=float(q.min()),q_max=float(q.max()),seal_sha256=receipt['sha256'])
        records.append(dict(shot=shot['shot'],arms=predictions))
    reports={a:summarize(rows[a],costs[a],threshold) for a in rows}; h=reports['H']
    checks=dict(both_classes_had_task_contrast=all(n>0 for n in contrast),
        class_guard=guard(h,reports['S']),fpr_cap=h['fpr'] is not None and h['fpr']<=.20,
        information=all(h['tp']>=reports[a]['tp'] and h['fp']<=reports[a]['fp'] and
            (h['tp']>reports[a]['tp'] or h['fp']<reports[a]['fp']) for a in ('S','P','L')))
    assert_classifier_unchanged(base,frozen_source)
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256: raise RuntimeError('qualified source file changed')
    result=dict(protocol=protocol,checks=checks,passed_development_checks=all(checks.values()),
        audit=reports,records=records,class_contrast_pairs=contrast,optimizer_steps=applied,
        training_rollouts=trace_index,rule_norm=float(candidate.norm()),parity=parity,
        classifier_unchanged=True,admitted=False,future_run=False,
        elapsed_seconds=time.monotonic()-started)
    _durable_json(args.output/'summary.json',result)
    print(json.dumps(dict(stage='trajectory_complete',checks=checks,audit={a:{k:r[k] for k in
        ('tp','fp','n_disrupt','n_safe','risks')} for a,r in reports.items()},
        seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
