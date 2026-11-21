"""Source-only token-interface alignment; fixed qualified-source risk readout."""
import argparse
import copy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..event_adapt_20261022.train import load_source
from ..fixed_test_protocol import score_fixed_test
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import pools, batch, score, sha, metrics
from .model import AlignedState, first_tokens, fixed_readout, observed_loss


def public(m):
    return {k:v for k,v in m.items() if k != 'predictions'}


def validation_score(rows, threshold):
    ids = [r['shot'] for r in rows]
    expected = io23.source_roles()['val']
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('full J-TEXT VAL identities required for source selection')
    return metrics(rows, threshold, lead=10.)


def load_candidate(ssl_path):
    base, parent_ck = load_parent(DEFAULT_PARENT)
    ssl = torch.load(ssl_path, map_location='cpu', weights_only=True)
    if ssl['model_config'] != base.cfg.to_dict():
        raise ValueError('SSL and supervised source architecture mismatch')
    model = AlignedState(copy.deepcopy(base), ssl['model']['level_scale'])
    state = model.state_dict()
    names = set(model.encoder_names())
    # Old SSL checkpoint's RETRAINED readout is deliberately not imported.
    selected = {k:v for k,v in ssl['model'].items()
                if k in names or k.startswith('reconstructor.')}
    if not names <= selected.keys():
        raise ValueError('incomplete observation encoder')
    state.update(selected); model.load_state_dict(state, strict=True)
    return model, base, parent_ck


def alignment_step(model, teacher, optimizer, x, m, *, reconstruct):
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        ht, vt = first_tokens(teacher, x, m)
        zt = fixed_readout(teacher, ht, vt)
    out = model(x, m, mode='aligned')
    token = ((out['tokens']-ht).square().mean(-1)*vt).sum()/vt.sum()
    risk = F.smooth_l1_loss(out['logit'], zt)
    rec = observed_loss(model, x[:16], m[:16]) if reconstruct else risk*0.
    loss = risk+.25*token+.20*rec
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite source alignment')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                   1., error_if_nonfinite=True)
    optimizer.step()
    return dict(total=float(loss.detach()), risk=float(risk.detach()),
                token=float(token.detach()), reconstruction=float(rec.detach()))


def reconstruction_check(model, shots):
    rng = np.random.default_rng(711); groups = pools(shots); values = []
    with torch.no_grad():
        for _ in range(16):
            x,m,*_ = batch(shots, groups, rng, 16, model.cfg, 'cuda')
            values.append(float(observed_loss(model, x, m)))
    return float(np.mean(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ssl', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--seed', type=int, default=131)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if min(args.epochs, args.steps) < 1: p.error('positive source budget required')
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    # Disable inference-only transformer fusion for the split-block API; this
    # keeps offline teacher, student and temporary-gradient paths comparable.
    torch.backends.mha.set_fastpath_enabled(False)
    start = time.monotonic(); rng = np.random.default_rng(args.seed)
    model, teacher, ck = load_candidate(args.ssl)
    model.cuda().eval(); teacher.cuda().eval(); teacher.requires_grad_(False)
    threshold = float(ck['threshold'])
    protected = {k:v.detach().clone() for k,v in model.state_dict().items()
                 if k not in set(model.encoder_names()) and not k.startswith('bridge.')}
    roles, advanced = io23.source_roles(), io23.advanced_times()
    print(json.dumps(dict(stage='loading_JTEXT_TRAIN_VAL', ssl=str(args.ssl))), flush=True)
    train, val = load_source(roles['train'], advanced), load_source(roles['val'], advanced)
    groups = pools(train)
    args.output.mkdir(parents=True)
    config = dict(format='aligned-state-v1', arguments={**vars(args),
        'ssl':str(args.ssl.resolve()), 'output':str(args.output.resolve())},
        parent_sha256=PARENT_SHA256, ssl_sha256=sha(args.ssl), threshold=threshold,
        fixed_readout='qualified source second block + LayerNorm + fast_origin + risk_head',
        ssl_scope='first block/embedding/position/CLS and signal reconstructor only',
        alignment='source SmoothL1(logit,teacher)+.25 tokenMSE+.20 masked-signal reconstruction',
        source_ids=roles, selection='VAL160 fixed source threshold; FPR<.15 priority then TPR,FPR,alignment loss',
        test_gate='TEST220 TPR>=.9 and FPR<.2, strict lead>10ms',
        event_use='inherited from selected SSL pretraining; no target weak labels fabricated',
        source_label='max(25ms,AdvancedTime), missing=25ms',
        input='flat23 unchanged; MNM missing in EAST, CIII sentinel contract; JTEXT ip*1000',
        discrete_auxiliary_exclusion='mode_number_n; risk input retained',
        source_lr=dict(bridge=.001, encoder=.00001), gradient_clip=1.,
        evidence='DEVELOPMENT_ONLY; no independent confirmation; no EAST access in source preparation')
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/s for s in
        ('simple23.py','simple_data.py','simple_train.py','simple_online.py','io23.py',
         'causal_input.py','fixed_test_protocol.py','masked_state_20261022/model.py',
         'event_adapt_20261022/train.py','redesign_20261004/prepare_prior.py')]
    config['code_sha256'] = {str(f.relative_to(io23.ROOT)):sha(f) for f in files}
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f,str(f.relative_to(io23.ROOT)))
    _durable_json(args.output/'config.json',config)
    initial = validation_score(score(model,val,'cuda'),threshold)
    rec_before = reconstruction_check(model,val)
    _durable_json(args.output/'before_alignment.json',dict(validation=initial,reconstruction=rec_before))
    print(json.dumps(dict(stage='before_alignment', validation=public(initial), reconstruction=rec_before)),flush=True)
    model.set_alignment(encoder=False)
    opt = torch.optim.AdamW(model.bridge.parameters(),lr=.001,weight_decay=0.)
    for step in range(256):
        x,m,*_ = batch(train,groups,rng,64,model.cfg,'cuda')
        loss = alignment_step(model,teacher,opt,x,m,reconstruct=False)
        if (step+1)%64 == 0:
            print(json.dumps(dict(stage='bridge_only',step=step+1,loss=loss,seconds=time.monotonic()-start)),flush=True)
    model.set_alignment(encoder=True)
    names = set(model.encoder_names())
    opt = torch.optim.AdamW([dict(params=[v for k,v in model.named_parameters() if k in names],lr=1e-5),
                            dict(params=list(model.bridge.parameters()),lr=.001)],weight_decay=0.)
    best = None; history = []
    for epoch in range(1,args.epochs+1):
        losses = []
        for _ in range(args.steps):
            x,m,*_ = batch(train,groups,rng,64,model.cfg,'cuda')
            losses.append(alignment_step(model,teacher,opt,x,m,reconstruct=True))
        measured = validation_score(score(model,val,'cuda'),threshold)
        row = dict(epoch=epoch,loss={k:float(np.mean([s[k] for s in losses])) for k in losses[0]},
                   validation=public(measured),seconds=time.monotonic()-start)
        history.append(row); _durable_json(args.output/'history.json',history); print(json.dumps(row),flush=True)
        rank = (measured['fpr'] >= .15, -measured['tpr'], measured['fpr'],row['loss']['risk'])
        if best is None or rank < best:
            best = rank
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            model_config=model.cfg.to_dict(),threshold=threshold,config=config,
                            validation=measured,epoch=epoch),args.output/'source.pt')
    for k,v in protected.items():
        if not torch.equal(v,model.state_dict()[k]): raise AssertionError(f'protected tensor changed: {k}')
    selected = torch.load(args.output/'source.pt',map_location='cuda',weights_only=True)
    model.load_state_dict(selected['model']); model.requires_grad_(False)
    rec_after = reconstruction_check(model,val)
    test = load_source(roles['test'],advanced)
    rows = score(model,test,'cuda'); measured = score_fixed_test('J-TEXT',rows,threshold)
    np.savez_compressed(args.output/'test220_traces.npz',**{f'{r["shot"]}_{k}':r[k] for r in rows for k in ('times','scores')})
    done = dict(status='COMPLETED',metrics=measured,epoch=selected['epoch'],
                source_qualified=measured['tpr'] >= .9 and measured['fpr'] < .2,
                reconstruction_before=rec_before,reconstruction_after=rec_after,
                seconds=time.monotonic()-start,source_sha256=sha(args.output/'source.pt'),
                source_original_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,protected_tensors=len(protected),
                east_accessed=False)
    _durable_json(args.output/'completed.json',done)
    print(json.dumps({**done,'metrics':public(measured)}),flush=True)


if __name__ == '__main__': main()
