"""Matched source-only preparation; no EAST signals, outcomes or thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..event_adapt_20261022.data import attach_training_events
from ..event_adapt_20261022.train import load_source, event_pool, sample_batch
from ..event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, read_events
from ..fixed_test_protocol import score_fixed_test
from ..masked_state_20261022.model import masked_views, reconstruction_loss
from ..masked_state_20261022.probe import distort, summary as risk_summary
from ..masked_state_20261022.train import fit_level_scale
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import pools, batch, score, sha
from .model import PairedFeedback, initial_encoder, update_observed, representation_loss, risk_state_loss
from .sampling import source_groups, balanced_pair

INNER_RATE, INNER_STEPS = .01, 2


def validation_examples(model, shots, n, seed):
    rng = np.random.default_rng(seed)
    eligible = []
    for s in shots:
        valid = s['valid'].copy()
        if s['is_disrupt']:
            valid &= np.round((s['down']-s['times'])*1000, 9) > 10.
        eligible.append(dict(s, valid=valid))
    groups = pools(eligible); rows = []
    for i in range(n):
        group = groups[i % 2]; sid, ends = group[int(rng.integers(len(group)))]
        end = int(rng.choice(ends)); s = shots[sid]
        x, m = simple_data.window(s, end, model.cfg.window)
        rows.append(dict(shot=s['shot'], end=end, y=int(s['y'][end] > .5), x=x, mask=m,
            gain=np.exp(rng.uniform(-.35, .35, 23)).astype('float32'),
            bias=rng.uniform(-1., 1., 23).astype('float32')))
    return rows


def diagnose(model, examples):
    records = []; device = model.base.center.device
    for row in examples:
        x = torch.as_tensor(row['x'], device=device)[None]
        m = torch.as_tensor(row['mask'], device=device)[None]
        gain = torch.as_tensor(row['gain'], device=device)
        bias = torch.as_tensor(row['bias'], device=device)
        origin = initial_encoder(model)
        clean = model.predict_fast(x, m, origin, mode='state')['tokens'].detach()
        clean_logit = float(model.predict_fast(x, m, origin)['logit'].detach())
        for condition in ('clean', 'shift'):
            observed = x if condition == 'clean' else distort(model, x, m, gain, bias)
            first = model.predict_fast(observed, m, origin, mode='state')
            first_rep = float(representation_loss(first['tokens'], clean, first['valid']).detach())
            before = float(model.predict_fast(observed, m, origin)['logit'].detach())
            fast, audit = update_observed(model, observed, m, learning_rate=INNER_RATE, steps=INNER_STEPS)
            after = float(model.predict_fast(observed, m, fast)['logit'].detach())
            out = model.predict_fast(observed, m, fast, mode='state')
            last_rep = float(representation_loss(out['tokens'], clean, out['valid']).detach())
            xx, mm, held = masked_views(observed, m)
            reconstruction = model.predict_fast(xx, mm, fast, mode='reconstruct')['reconstruction']
            last_loss = float(reconstruction_loss(model, reconstruction, xx, held).detach())
            records.append(dict(shot=row['shot'], end=row['end'], y=row['y'], condition=condition,
                before_logit=before, after_logit=after, clean_logit=clean_logit,
                before_bce=float(np.logaddexp(0., before)-row['y']*before),
                after_bce=float(np.logaddexp(0., after)-row['y']*after),
                before_completion=float(audit['first_loss'].detach()), after_completion=last_loss,
                parameter_delta_norm=audit['delta'], before_representation=first_rep, after_representation=last_rep))
    result = risk_summary(records)
    for c in ('clean', 'shift'):
        rr = [r for r in records if r['condition'] == c]
        result[c].update(before_representation=float(np.mean([r['before_representation'] for r in rr])),
            after_representation=float(np.mean([r['after_representation'] for r in rr])),
            representation_improved=sum(r['after_representation'] < r['before_representation'] for r in rr))
    gate = result['clean']['after_bce'] <= result['clean']['before_bce'] and result['shift']['after_bce'] < result['shift']['before_bce']
    return dict(summary=result, probe_gate=gate, records=records)


def train_episode(model, observed, mask, y, reference, mode, risk_reference=None):
    if mode == 'F':
        xx, mm, held = masked_views(observed, mask)
        out = model(xx, mm, mode='reconstruct')
        loss = reconstruction_loss(model, out['reconstruction'], xx, held)
        return loss, dict(completion=float(loss.detach()))
    fast, audit = update_observed(model, observed, mask, learning_rate=INNER_RATE,
                                 steps=INNER_STEPS, create_graph=True)
    risk = F.binary_cross_entropy_with_logits(model.predict_fast(observed, mask, fast)['logit'], y)
    state = model.predict_fast(observed, mask, fast, mode='state')
    paired = representation_loss(state['tokens'], reference, state['valid'])
    bounded = (risk_state_loss(model,model.predict_fast(observed,mask,fast,mode='risk_state')['risk_state'],risk_reference)
               if mode == 'C' else risk*0.)
    loss = risk+.2*audit['first_loss']+(paired if mode == 'P' else bounded if mode == 'C' else 0.)
    return loss, dict(completion=float(audit['first_loss'].detach()), risk=float(risk.detach()),
                     paired=float(paired.detach()), risk_state_bound=float(bounded.detach()), delta=audit['delta'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['warm', 'F', 'R', 'P', 'C'], required=True)
    p.add_argument('--initial', type=Path)
    p.add_argument('--warmup', type=int, default=512)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--sampling', choices=['original','balanced','safe_negative'], default='original')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if min(args.warmup, args.steps) < 1: p.error('positive source budgets required')
    if args.mode != 'warm' and not args.initial: p.error('matched frozen-parent decoder initialization required')
    if not torch.cuda.is_available(): raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131); start = time.monotonic()
    parent, ck = load_parent(DEFAULT_PARENT)
    parent_state = {k:v.detach().cpu().clone() for k,v in parent.state_dict().items()}
    roles, adv = io23.source_roles(), io23.advanced_times()
    print(json.dumps(dict(stage='load_source', mode=args.mode)), flush=True)
    train, val = load_source(roles['train'], adv), load_source(roles['val'], adv)
    if sha(DEFAULT_LABELS) != LABEL_SHA256: raise ValueError('event data changed')
    grouped, _ = read_events(DEFAULT_LABELS)
    train = attach_training_events(train, grouped, allowed_train_ids=roles['train'])
    q = fit_level_scale(parent, train, roles['train'])
    model = PairedFeedback(parent, q).cuda(); model.prepare_decoder()
    if args.initial:
        receipt = json.loads((args.initial.parent/'completed.json').read_text())
        if sha(args.initial) != receipt['checkpoint_sha256']: raise ValueError('initialization artifact changed')
        state = torch.load(args.initial, weights_only=True, map_location='cuda')
        if state['config']['parent_sha256'] != PARENT_SHA256: raise ValueError('parent mismatch')
        model.load_state_dict(state['model'])
    args.output.mkdir(parents=True)
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/name for name in (
        'simple23.py','simple_data.py','io23.py','simple_train.py','causal_input.py','fixed_test_protocol.py',
        'event_prefix_probe.py','event_adapt_20261022/train.py','event_adapt_20261022/data.py',
        'masked_state_20261022/model.py','masked_state_20261022/train.py','masked_state_20261022/probe.py',
        'redesign_20261004/prepare_prior.py')]
    config = dict(format='paired-feedback-v1', mode=args.mode, seed=131,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        parent_sha256=PARENT_SHA256, threshold=float(ck['threshold']), source_ids=roles,
        initial_sha256=sha(args.initial) if args.initial else None,
        event_sha256=LABEL_SHA256, event_usage='TRAIN event TIMES for matched sampling; A-F classes not in loss; upstream TEST exposure',
        source_label='max(25ms,AdvancedTime)', static_source_gate='TEST220 strict lead>10ms, preserve100TP13FP',
        inner=dict(learning_rate=INNER_RATE, steps=INNER_STEPS, reset_each_window=True,
                   update='input projection + first Transformer block only; no labels or clean reference'),
        outer='F completion; R post-update BCE+0.2 completion; P R+paired token recovery; C R+paired fixed-risk-readout-state recovery bound',
        sequential_followup=args.mode == 'C',
        source_sampling=args.sampling,
        sampling='2 TRAIN windows per step, clean+synthetic pair for each; same sequence only when sampling policy matches',
        synthetic='source-asinh gain exp(U[-.35,.35]), bias U[-1,1]; engineering stress, NOT verified physics',
        source_selection='64 VAL windows seed1811; lowest clean/shift post-update meanBCE at steps128,256,384,512',
        reporting_probe='128 VAL windows seed922; cleanBCE must not increase, shiftBCE must decrease',
        east_accessed=False, independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(args.output/'config.json', config)
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f,str(f.relative_to(io23.ROOT)))
    def save(path, step):
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        config=config, step=step), path)
    opt = torch.optim.AdamW(model.reconstructor.parameters(), lr=1e-4, weight_decay=0.)
    rng = np.random.default_rng(131); groups = pools(train); events = event_pool(train)
    history = []
    if args.mode == 'warm':
        for step in range(args.warmup):
            x, m, _, _, _ = batch(train, groups, rng, 32, model.cfg, 'cuda')
            xx, mm, held = masked_views(x, m)
            opt.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model, model(xx,mm,mode='reconstruct')['reconstruction'], xx, held)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.reconstructor.parameters(), 1., error_if_nonfinite=True); opt.step()
            if (step+1)%128 == 0:
                print(json.dumps(dict(stage='warm',step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start)),flush=True)
        save(args.output/'warm.pt',args.warmup); artifact = args.output/'warm.pt'
        test_rows = score(model, load_source(roles['test'], adv), 'cuda')
        metrics = score_fixed_test('J-TEXT',test_rows,ck['threshold'])
        if (metrics['tp'],metrics['fp']) != (100,13): raise AssertionError('parent static reproduction failed')
        np.savez_compressed(args.output/'test220_traces.npz',**{f'{r["shot"]}_{k}':r[k] for r in test_rows for k in ('times','scores')})
        _durable_json(args.output/'static_test220.json',metrics)
        diagnostic = None; selected = args.warmup
    else:
        selected_examples = validation_examples(model,val,64,1811)
        best = float('inf'); selected = None; rng = np.random.default_rng(811)
        digest = hashlib.sha256()
        sampled_labels=[];negative_disruptive=0;sampling_receipts=[]
        balanced_groups=source_groups(train,policy=args.sampling) if args.sampling!='original' else None
        for step in range(args.steps):
            if balanced_groups is None:
                x,m,y,_,_,_ = sample_batch(train,groups,events,rng,model.cfg,2,'cuda',channel_dropout=0.)
            else:
                x,m,y,receipts = balanced_pair(train,balanced_groups,rng,model.cfg,'cuda')
                negative_disruptive+=sum(r['is_disrupt'] and r['y']==0 for r in receipts)
                sampling_receipts.extend(receipts)
            sampled_labels.extend(y.detach().cpu().tolist())
            opt.zero_grad(set_to_none=True); stats=[]
            for i in range(len(x)):
                clean, mask = x[i:i+1],m[i:i+1]
                origin = initial_encoder(model)
                reference = model.predict_fast(clean,mask,origin,mode='state')['tokens'].detach()
                risk_reference = (model.predict_fast(clean,mask,origin,mode='risk_state')['risk_state'].detach()
                                  if args.mode == 'C' else None)
                gain = np.exp(rng.uniform(-.35,.35,23)).astype('float32')
                bias = rng.uniform(-1.,1.,23).astype('float32')
                for tensor in (clean,mask,y[i:i+1]): digest.update(tensor.detach().cpu().numpy().tobytes())
                digest.update(gain.tobytes()); digest.update(bias.tobytes())
                changed = distort(model,clean,mask,torch.as_tensor(gain,device='cuda'),torch.as_tensor(bias,device='cuda'))
                for observed in (clean,changed):
                    loss, info = train_episode(model,observed,mask,y[i:i+1].float(),reference,args.mode,risk_reference)
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite source objective')
                    (loss/(2*len(x))).backward(); stats.append(info)
            torch.nn.utils.clip_grad_norm_(model.reconstructor.parameters(),1.,error_if_nonfinite=True); opt.step()
            if (step+1)%32 == 0:
                r=dict(stage='source_pair_training',step=step+1,train={k:float(np.mean([s[k] for s in stats])) for k in stats[0]},seconds=time.monotonic()-start)
                print(json.dumps(r),flush=True)
            if (step+1)%128 == 0 or step+1 == args.steps:
                diagnostic = diagnose(model, selected_examples)
                criterion = np.mean([diagnostic['summary'][c]['after_bce'] for c in ('clean','shift')])
                r=dict(step=step+1,validation=diagnostic['summary'],criterion=float(criterion))
                history.append(r); _durable_json(args.output/'history.json',history)
                if criterion < best: best=criterion; selected=step+1; save(args.output/'selected.pt',selected)
                print(json.dumps(dict(stage='selection_VAL',**r)),flush=True)
        _durable_json(args.output/'training_inputs.json',dict(sequence_sha256=digest.hexdigest(),steps=args.steps,source_windows_per_step=2,
            policy=args.sampling,positive_windows=sum(v>.5 for v in sampled_labels),negative_windows=sum(v<=.5 for v in sampled_labels),
            negative_disruptive_windows=negative_disruptive if balanced_groups is not None else None))
        if sampling_receipts: _durable_json(args.output/'sampling_receipts.json',sampling_receipts)
        artifact=args.output/'selected.pt'; chosen=torch.load(artifact,weights_only=True,map_location='cuda')
        model.load_state_dict(chosen['model'])
        diagnostic=diagnose(model,validation_examples(model,val,128,922))
        _durable_json(args.output/'reporting_probe.json',diagnostic)
    if any(not torch.equal(v,model.base.state_dict()[k].cpu()) for k,v in parent_state.items()):
        raise AssertionError('persistent classifier changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('qualified parent file changed')
    result=dict(status='COMPLETED',mode=args.mode,selected_step=selected,
        checkpoint_sha256=sha(artifact),parent_unchanged=True,source_static_qualified=True,
        reporting_summary=diagnostic['summary'] if diagnostic else None,
        allow_east_candidate=diagnostic['probe_gate'] if diagnostic else False,
        seconds=time.monotonic()-start,east_accessed=False)
    _durable_json(args.output/'completed.json',result);print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
