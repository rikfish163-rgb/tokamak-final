"""Source TRAIN only: joint vs exact inner-operator meta preparation.

The trained source checkpoint is new and must earn its own TEST220 gate.
The original qualified checkpoint stays immutable. No EAST signal is read.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..event_adapt_20261022.data import attach_training_events
from ..event_adapt_20261022.model import event_loss
from ..event_adapt_20261022.train import load_source, event_pool, sample_batch, public
from ..event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, read_events
from ..fixed_test_protocol import score_fixed_test
from ..masked_state_20261022.model import masked_views, reconstruction_loss
from ..paired_feedback_20261022.model import PairedFeedback, update_observed
from ..paired_feedback_20261022.prepare import diagnose, validation_examples
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import score, sha, pools, select_threshold


WARM = (Path(__file__).parents[1] /
        'runs/paired_feedback_warm_20261022/warm.pt')
INNER_RATE, INNER_STEPS = .01, 2


def sampled_positive_probability(shots, events):
    """Exact class prior of event_adapt.sample_batch, not population prevalence.

    The sampler chooses ordinary positives with probability 1/4 and event
    windows with probability 1/4. Within the event branch it chooses shot
    outcome, an interval entry, and an endpoint uniformly in that order.
    """
    if not shots or any(s['device'] != 'J-TEXT' for s in shots):
        raise ValueError('J-TEXT TRAIN sampling distribution required')
    event_positive = []
    for disruptive in (False, True):
        entries = events[disruptive]
        if not entries or any(not len(ends) for _, ends in entries):
            raise ValueError('nonempty source event sampling pools required')
        event_positive.append(np.mean([
            np.mean(shots[i]['y'][ends] > .5) for i, ends in entries]))
    return float(.25 + .125*sum(event_positive))


def selection_rank(metrics, probe):
    # Source competence is a constraint; within it select the task actually
    # prepared, not merely the best non-adapting static checkpoint.
    return (metrics['tpr'] < .9 or metrics['fpr'] > .15,
            probe['summary']['clean']['after_bce'], -metrics['tpr'], metrics['fpr'])


def enable_joint(model):
    """Train the recipient and the two training-only auxiliary readouts."""
    names = set(model.encoder_names()) | set(model.risk_names())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names or name.startswith(
            ('reconstructor.', 'event_head.')))
        parameter.grad = None
    model.eval()  # No stochastic dropout or batch-statistics adaptation.


def objective(model, x, mask, y, event, *, mode, risk_weight=1.):
    if mode not in ('joint', 'meta'):
        raise ValueError('declared joint or meta source objective required')
    if len(x) != 1 or y.shape != (1,) or event.shape != (1,):
        raise ValueError('one source window per independent temporary update')
    if not math.isfinite(risk_weight) or risk_weight <= 0:
        raise ValueError('positive finite risk weight required')
    before = F.binary_cross_entropy_with_logits(model(x, mask)['logit'], y.float())
    # Event categories are offline weak targets. Unknown remains -1 and is
    # excluded; neither this target nor the risk label enters the inner API.
    observed = model(x, mask, mode='reconstruct')
    event_term = event_loss(observed['event'], event.long())
    if mode == 'meta':
        fast, audit = update_observed(model, x, mask, learning_rate=INNER_RATE,
            steps=INNER_STEPS, create_graph=True, track_initial=True)
        after = F.binary_cross_entropy_with_logits(
            model.predict_fast(x, mask, fast)['logit'], y.float())
        completion = audit['first_loss']
        risk = .5*(before+after)
    else:
        xx, mm, held = masked_views(x, mask)
        completion = reconstruction_loss(
            model, model(xx, mm, mode='reconstruct')['reconstruction'], xx, held)
        after = before
        risk = before
    total = risk_weight*risk+.2*completion+.1*event_term
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite joint source objective')
    return total, dict(risk_before=float(before.detach()),
        risk_after=float(after.detach()), completion=float(completion.detach()),
        event=float(event_term.detach()), total=float(total.detach()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['joint', 'meta'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--balance-risk', action='store_true',
        help='source-only inverse sampler-prior weighting; samples and auxiliaries unchanged')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.epochs, args.steps, args.batch_size) < 1:
        p.error('positive source training budget required')
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    rng = np.random.default_rng(1622)
    start = time.monotonic()
    base, old = load_parent(DEFAULT_PARENT)
    warm_receipt = json.loads((WARM.parent/'completed.json').read_text())
    if sha(WARM) != warm_receipt['checkpoint_sha256']:
        raise ValueError('source-only initialization changed')
    warm = torch.load(WARM, weights_only=True, map_location='cpu')
    if warm['config']['parent_sha256'] != PARENT_SHA256:
        raise ValueError('source parent mismatch')
    for key, value in base.state_dict().items():
        if not torch.equal(value, warm['model']['base.'+key]):
            raise ValueError('warm checkpoint is not the exact qualified source')
    model = PairedFeedback(base, warm['model']['level_scale'])
    model.load_state_dict(warm['model'])
    model.cuda()
    enable_joint(model)
    frozen = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
              if k in ('base.center', 'base.scale', 'level_scale', 'base.fast_origin')
              or k.startswith('base.forecast_head.')}
    roles, adv = io23.source_roles(), io23.advanced_times()
    print(json.dumps(dict(stage='load_JTEXT_train_val', mode=args.mode)), flush=True)
    train, val = load_source(roles['train'], adv), load_source(roles['val'], adv)
    if sha(DEFAULT_LABELS) != LABEL_SHA256:
        raise ValueError('event labels changed')
    grouped, _ = read_events(DEFAULT_LABELS)
    train = attach_training_events(train, grouped, allowed_train_ids=roles['train'])
    groups, events = pools(train), event_pool(train)
    prior = sampled_positive_probability(train, events)
    risk_weights = {0:.5/(1.-prior), 1:.5/prior} if args.balance_risk else {0:1., 1:1.}
    params = list(model.named_parameters())
    opt = torch.optim.AdamW([
        dict(params=[v for k,v in params if v.requires_grad and k.startswith('base.')], lr=1e-5),
        dict(params=[v for k,v in params if v.requires_grad and not k.startswith('base.')], lr=1e-4),
    ], weight_decay=0.)
    args.output.mkdir(parents=True)
    files = [Path(__file__)]
    files += [Path(__file__).parents[1]/n for n in (
        'paired_feedback_20261022/model.py', 'paired_feedback_20261022/prepare.py',
        'masked_state_20261022/model.py', 'masked_state_20261022/probe.py',
        'simple23.py', 'simple_train.py', 'simple_data.py', 'io23.py', 'causal_input.py',
        'fixed_test_protocol.py', 'event_prefix_probe.py',
        'event_adapt_20261022/train.py', 'event_adapt_20261022/data.py',
        'event_adapt_20261022/model.py', 'redesign_20261004/prepare_prior.py')]
    config = dict(mode=args.mode, seed=131, sampling_seed=1622,
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        parent_sha256=PARENT_SHA256, warm_sha256=sha(WARM),
        source_ids=roles, event_sha256=LABEL_SHA256,
        update='same two SGD .01 observed-window steps as deployment; exact outer gradient to source initialization',
        source='original 23-feature source coordinates; JT max(25ms,AdvancedTime); no synthetic gains',
        event='TRAIN known A-F weak labels only; unknown ignored; upstream TEST exposure remains',
        training='joint: BCE+.2 completion+.1 event; meta: .5(BCE_before+BCE_after)+.2 completion+.1 event',
        sampling_positive_probability=prior, risk_weights=risk_weights,
        risk_balance='TRAIN sampler-prior importance weights on risk terms only; event/completion/sample sequence unchanged',
        sequential_followup=bool(args.balance_risk),
        checkpoint_selection='static VAL TPR>=.9/FPR<=.15 constraint; then lowest post-update BCE on64 original VAL windows seed1811; initialization eligible; no TEST selection',
        comparison='matched initial state, source samples and optimizer steps; not matched FLOPs',
        new_source_must_qualify=True, independent_confirmation=False, east_accessed=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(args.output/'config.json', config)
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, str(file.relative_to(io23.ROOT)))
    def save(epoch, metrics):
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            config=config, epoch=epoch, threshold=float(metrics['threshold']),
            validation=public(metrics)), args.output/'source.pt')
    selection_examples = validation_examples(model, val, 64, 1811)
    initial = select_threshold(score(model, val, 'cuda'), .15)
    initial_probe = diagnose(model, selection_examples)
    best = selection_rank(initial, initial_probe)
    save(0, initial)
    history = [dict(epoch=0, validation=public(initial),
                    selection_probe=initial_probe['summary'])]
    digest = hashlib.sha256()
    known, positives, windows = 0, 0, 0
    for epoch in range(1, args.epochs+1):
        enable_joint(model)
        logs = []
        for step in range(args.steps):
            x,m,y,e,_,_ = sample_batch(train, groups, events, rng, model.cfg,
                                      args.batch_size, 'cuda', channel_dropout=0.)
            for v in (x,m,y,e):
                digest.update(v.detach().cpu().numpy().tobytes())
            known += int((e>=0).sum()); positives += int((y>.5).sum()); windows += len(x)
            opt.zero_grad(set_to_none=True)
            for i in range(len(x)):
                loss, row = objective(model, x[i:i+1], m[i:i+1], y[i:i+1], e[i:i+1],
                    mode=args.mode, risk_weight=risk_weights[int(y[i]>.5)])
                (loss/len(x)).backward()
                logs.append(row)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          1., error_if_nonfinite=True)
            opt.step()
            if (step+1)%64 == 0:
                print(json.dumps(dict(stage='source_update', epoch=epoch, step=step+1,
                    last=logs[-1], seconds=time.monotonic()-start)), flush=True)
        measured = select_threshold(score(model, val, 'cuda'), .15)
        checked = diagnose(model, selection_examples)
        row = dict(epoch=epoch, validation=public(measured),
            selection_probe=checked['summary'],
            losses={k:float(np.mean([r[k] for r in logs])) for k in logs[0]},
            seconds=time.monotonic()-start)
        history.append(row)
        _durable_json(args.output/'history.json', history)
        if selection_rank(measured, checked) < best:
            best = selection_rank(measured, checked)
            save(epoch, measured)
        print(json.dumps(dict(stage='VAL_selection', **row)), flush=True)
    # Last checkpoint is diagnostic only; do not silently substitute it if
    # the predeclared validation criterion selected the initialization.
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        config=config, epoch=args.epochs, threshold=measured['threshold'],
        validation=public(measured)), args.output/'last.pt')
    selected = torch.load(args.output/'source.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(selected['model'])
    model.prepare_decoder()
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k,v in frozen.items()):
        raise AssertionError('frozen coordinates/original fast origin/forecast head changed')
    rows = score(model, load_source(roles['test'], adv), 'cuda')
    test = score_fixed_test('J-TEXT', rows, selected['threshold'])
    np.savez_compressed(args.output/'test220_traces.npz',
        **{f'{r["shot"]}_{k}':r[k] for r in rows for k in ('times','scores')})
    _durable_json(args.output/'test220.json', test)
    probe = diagnose(model, validation_examples(model, val, 128, 922))
    _durable_json(args.output/'reporting_probe.json', probe)
    _durable_json(args.output/'training_inputs.json', dict(sequence_sha256=digest.hexdigest(),
        windows=windows, positives=positives, negatives=windows-positives, known_event_windows=known))
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('original qualified checkpoint file changed')
    done = dict(status='COMPLETED', selected_epoch=selected['epoch'], metrics=public(test),
        qualified=test['tpr']>=.9 and test['fpr']<.2, probe_gate=probe['probe_gate'],
        allow_east_candidate=test['tpr']>=.9 and test['fpr']<.2 and probe['probe_gate'],
        checkpoint_sha256=sha(args.output/'source.pt'), reporting_summary=probe['summary'],
        elapsed_seconds=time.monotonic()-start, east_accessed=False, independent_confirmation=False)
    _durable_json(args.output/'completed.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    main()
