"""EAST-only supervised representation control, not an online candidate."""

import argparse
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from . import commissioning_embedding_fast_20261114 as embed
from . import io23, simple_data, warning_lead_metrics as lead
from .causal_input import apply_input_contract
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, write
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot, timely_auc
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


ROOT = io23.ROOT
REFERENCE = ROOT/'outputs/east360_opposed_feedback_20261114_v2'
EXTENDED = ROOT/'probe_outputs/commissioning_frozen_extension_east2593_20261113_v1'
WIDTH, HORIZON, DEADLINE = 128, 125., 30.
STRIDE, STEPS, EPOCHS, BATCH, SEED = 8, 180, 6, 48, 131
CAPTURE_BATCH = 512
CAPS = (.10, .15, .20)
ARMS = ('frozen_encoder', 'trained_encoder')
ORDER_ARMS = (*ARMS, 'old_patches_as_set')
LOCATIONS = {
    'embedding_and_head': 'embedding.',
    'first_block_and_head': 'encoder.layers.0.',
    'last_block_and_head': 'encoder.layers.1.',
}
NUMERIC = {
    'eight_values_fixed_head': embed.VALUES,
    'all_values_fixed_head': embed.ALL_AVAILABLE_VALUES,
}


class EarlierPositions(nn.Module):
    def forward(self, positions):
        early = positions[:, :12].mean(1, keepdim=True).expand(-1, 12, -1)
        return torch.cat((early, positions[:, 12:]), dim=1)


class ValueColumns(nn.Module):
    """Unselected weights are constants, including under decoupled decay."""
    def __init__(self, weight, enabled, patch):
        super().__init__()
        self.register_buffer('fixed', weight.detach().clone())
        cols = torch.tensor(enabled, device=weight.device)
        self.register_buffer('columns', (46*torch.arange(patch,device=weight.device)[:,None]+cols).flatten())

    def right_inverse(self, weight):
        return weight.index_select(1,self.columns)

    def forward(self, selected):
        return self.fixed.index_copy(1,self.columns,selected)


def tie_earlier_positions(net):
    if (net.cfg.window, net.cfg.patch)!=(128, 8):
        raise ValueError('the old-patch set keeps 12×8 history and 4×8 ordered recent samples')
    parametrize.register_parametrization(net, 'position', EarlierPositions())


def train_scope(net, arm):
    if arm in NUMERIC:
        if net.embedding.in_features!=net.cfg.patch*46:
            raise ValueError('separate value and mask columns are required')
        net.requires_grad_(False)
        parametrize.register_parametrization(net.embedding,'weight',
            ValueColumns(net.embedding.weight,NUMERIC[arm],net.cfg.patch))
        net.embedding.parametrizations.weight.original.requires_grad_(True)
        return
    if arm=='old_patches_as_set':
        tie_earlier_positions(net)
    elif arm not in (*ARMS, *LOCATIONS):
        raise ValueError(f'unknown offline control: {arm}')
    for name, p in net.named_parameters():
        extra = (name.startswith(LOCATIONS[arm]) if arm in LOCATIONS else
                 arm!='frozen_encoder' and not name.startswith(('forecast_head', 'fast_origin')))
        p.requires_grad_(name.startswith('risk_head.') or extra)


def archival_roles(ids, stream):
    if set(stream)-set(ids) or len(set(ids))!=len(ids):
        raise ValueError('archival inventory and stream membership must agree')
    # Alternate contiguous ID blocks, without interpreting ID as acquisition
    # time. The split is made before opening either archival label set.
    extra = sorted(set(ids)-set(stream))
    return [s for s in extra if (s//50)%4!=0], [s for s in extra if (s//50)%4==0]


def windows(x, m, endpoints):
    ticks = endpoints[:, None]+np.arange(1-WIDTH, 1)
    valid = ticks>=0
    at = np.maximum(ticks, 0)
    return np.where(valid[..., None], x[at], 0.), valid[..., None] & m[at]


def interval(times, disruptive, down):
    y, valid = io23.labels_at(times, disruptive, down, HORIZON)
    target = (y>0.) & valid
    eligible = np.round((down-times)*1000., 9)>DEADLINE if disruptive else np.zeros(len(y), bool)
    return (np.flatnonzero(valid & ~target), np.flatnonzero(target & eligible),
            np.flatnonzero(target & ~eligible))


def load_signal(sid):
    return apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))


def archival_shots(ids):
    result = []
    for sid in ids:
        s = load_signal(sid)
        y, d = io23.read_outcome('EAST', sid)
        result.append(dict(s, is_disrupt=y, down=d))
    return result


def example_pools(shots):
    pools = [[] for _ in range(4)]
    for i, s in enumerate(shots):
        normal, early, late = interval(s['times'], s['is_disrupt'], s['down'])
        for j, endpoints in ((int(not s['is_disrupt']), normal), (2, early), (3, late)):
            if len(endpoints):
                pools[j].append((i, endpoints))
    if not all(pools):
        raise ValueError('requires normal/safe earlier and later positive training windows')
    return pools


def draw(shots, pools, rng, device):
    # Balanced early/late positives; two thirds of negatives are safe shots.
    # Draw the shot first so a long safe shot cannot dominate an update.
    states, masks, labels = [], [], []
    group_id = rng.choice(4, size=BATCH, p=(1/6, 1/3, 1/4, 1/4))
    for j in group_id:
        group = pools[j]
        i, candidates = group[rng.integers(len(group))]
        s = shots[i]
        x, m = windows(s['values'], s['available'], rng.choice(candidates, size=1))
        states.append(x[0])
        masks.append(m[0])
        labels.append(j>=2)
    return (torch.as_tensor(np.stack(states), device=device),
            torch.as_tensor(np.stack(masks), device=device),
            torch.tensor(labels, dtype=torch.float32, device=device))


@torch.inference_mode()
def curves(models, shot, *, stride):
    # Every grid point is a score for the latest observed input; future values
    # are neither masked into the model nor used to build its normalization.
    take = np.arange(0, len(shot['times']), stride)
    device = models[0].center.device
    x, m = shot['values'], shot['available']
    chunks = []
    for start in range(0, len(take), CAPTURE_BATCH):
        values, mask = windows(x, m, take[start:start+CAPTURE_BATCH])
        values, mask = (torch.as_tensor(a, device=device) for a in (values, mask))
        batch = [net(values, mask)['logit'].sigmoid().cpu().numpy() for net in models]
        chunks.append(np.stack(batch))
    return take, np.concatenate(chunks, axis=1)


def calibrated(scores):
    """Lock calibration using validation outcomes, not fixed360 outcomes."""
    return {str(c): best_at_count(scores, int(c*sum(not s['disruptive'] for s in scores)))
            for c in CAPS}


def validate(models, shots):
    rows = [[] for _ in models]
    for s in shots:
        idx, value = curves(models, s, stride=STRIDE)
        for z, target in zip(value, rows, strict=True):
            peak = peak_for_shot(s['times'][idx], z, s['is_disrupt'], s['down'])
            target.append(dict(shot=s['shot'], disruptive=s['is_disrupt'], peak=peak))
    return rows


def train_models(out, models, fitting, heldout, arms):
    device = models[0].center.device
    parameters = [list(p for p in m.parameters() if p.requires_grad) for m in models]
    opt = [torch.optim.AdamW(q, lr=3e-4, weight_decay=.01) for q in parameters]
    pools = example_pools(fitting)
    rng = np.random.default_rng(SEED)
    best = [(-1., -1.) for _ in models]
    history = []
    for ep in range(1, EPOCHS+1):
        losses = np.zeros(len(models))
        for m in models:
            m.train()
        for _ in range(STEPS):
            x, mask, y = draw(fitting, pools, rng, device)
            for i, m in enumerate(models):
                opt[i].zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(m(x, mask)['logit'], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters[i], 1., error_if_nonfinite=True)
                opt[i].step()
                losses[i] += loss.item()
        for m in models:
            m.eval()
        val = validate(models, heldout)
        fits = [calibrated(r) for r in val]
        row = dict(epoch=ep, loss=(losses/STEPS).tolist(), validation=dict(zip(arms, fits, strict=True)))
        history.append(row)
        for i, (k, fit, m) in enumerate(zip(arms, fits, models, strict=True)):
            rank = (fit['0.2']['tpr'], -fit['0.2']['fpr'])
            if rank>best[i]:
                best[i] = rank
                torch.save(dict(epoch=ep, state=m.state_dict(), val=fit), out/f'{k}.pt')
        print(row, flush=True)
    write(out/'fitting.json', history)
    receipts = {}
    for k, m in zip(arms, models, strict=True):
        state = torch.load(out/f'{k}.pt', weights_only=True, map_location=device)
        m.load_state_dict(state['state'])
        receipts[k] = dict(epoch=state['epoch'], calibration=state['val'])
    return receipts


def after_seal(out, p, order, receipts, files, arms):
    original = read(REFERENCE/'per_shot.json')['0.2']
    names = ['bias_only', 'current_5', *arms]
    rows = {k: [] for k in names}
    ranking = {k: [] for k in names}
    fixed = {str(c):{k:[] for k in names} for c in CAPS}
    for i, (sid, f) in enumerate(zip(order, files, strict=True)):
        raw = load_signal(sid)
        y, down = io23.read_outcome('EAST', sid)
        with np.load(out/'shots'/f, allow_pickle=False) as s, np.load(
                REFERENCE/'shots'/f, allow_pickle=False) as o:
            prediction = {k:s[k] for k in arms}
            prediction.update({k:o[k] for k in ('bias_only', 'current_5')})
        for k, z in prediction.items():
            if k in ('bias_only', 'current_5'):
                threshold = p['original'] if i<p['startup'] else p['calibration']['0.2']
            else:
                threshold = receipts[k]['calibration']['0.2']['threshold']
            identity = dict(shot=sid, disruptive=y, down_s=down)
            one = dict(**identity, **score_curve(raw['times'], z, y, down, threshold))
            rows[k].append(one)
            if k in original and one!=original[k][i]:
                raise ValueError(f'frozen reference changed: {k} {sid}')
            ranking[k].append(dict(shot=sid, disruptive=y, peak=peak_for_shot(raw['times'], z, y, down)))
            for cap in fixed:
                t = (p['original'] if i<p['startup'] else p['calibration'][cap]) if k in original else (
                    receipts[k]['calibration'][cap]['threshold'])
                fixed[cap][k].append(dict(**identity, **score_curve(raw['times'], z, y, down, t)))
    write(out/'per_shot.json', fixed)
    best = {c: {} for c in fixed}
    # Keep the original prep alarms for all members of this post-seal
    # comparison. Offline frozen thresholds above are run from shot one.
    cold = lead.summarize(rows['current_5'][:p['startup']])
    for c in fixed:
        for k in names:
            v = best_at_count(ranking[k][p['startup']:], int(float(c)*180)-cold['fp'])
            best[c][k] = dict(tp=v['tp']+cold['by_deadline']['30']['tp'], fp=v['fp']+cold['fp'])
    report = dict(fixed={c:{k:lead.summarize(v) for k,v in d.items()} for c,d in fixed.items()},
        headline_flips={k:flips(rows['current_5'], rows[k]) for k in arms},
        matched_fp_ranking=best, timing_auc_suffix={k:timely_auc(ranking[k][p['startup']:]) for k in names},
        ranking_note='post-seal peak ranking in suffix; same locked cold20 added to all; not deployed thresholds')
    write(out/'result.json', report)
    print({c:{k:(r['by_deadline']['30']['tp'],r['fp']) for k,r in d.items()}
           for c,d in report['fixed'].items()}, 'archival_only', flush=True)


def verify_numeric_scope(base, models, arms):
    before = base.state_dict()
    receipt = {}
    for k,model in zip(arms,models,strict=True):
        effective = model.embedding.weight.detach().cpu()
        fixed = before['embedding.weight']
        cols = torch.arange(fixed.shape[1])%46
        allowed = torch.isin(cols,torch.tensor(NUMERIC[k]))
        delta = effective-fixed
        if torch.count_nonzero(delta[:,~allowed]):
            raise ValueError(f'fixed embedding coordinates changed in {k}')
        after = model.state_dict()
        if any(not torch.equal(after[n].cpu(),v) for n,v in before.items() if n!='embedding.weight'):
            raise ValueError(f'non-embedding parameter changed in {k}')
        receipt[k] = dict(allowed_names=[io23.RAW23[j] for j in NUMERIC[k]],
            modified_parameters=int(torch.count_nonzero(delta)),embedding_delta_L2=float(delta.norm()),
            by_channel_L2=dict(zip(io23.RAW23,delta.unflatten(1,(-1,46))[:,:,:23]
                                   .square().sum((0,1)).sqrt().tolist(),strict=True)))
    return receipt


def run(out, mode='core'):
    start = monotonic()
    ref, ext = read(REFERENCE/'protocol.json'), read(EXTENDED/'protocol.json')
    order = ext['prior_360']
    files = [f'{i:04d}_{s}.npz' for i,s in enumerate(order,1)]
    if order!=ref['order'] or len(order)!=360 or ref['source_sha256']!=PARENT_SHA256:
        raise ValueError('requires the locked online reference')
    archival = [x['shot'] for x in ext['order']]
    if set(archival)&set(order) or len(archival)!=2593:
        raise ValueError('no fixed360 signal or outcome is allowed in the archival fit')
    train, val = archival_roles(archival+order, order)
    arms = {'core': ARMS, 'order': ORDER_ARMS, 'location': (*ARMS, *LOCATIONS),
            'numeric_scope':tuple(NUMERIC)}[mode]
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__, out/'executed.py')
    write(out/'protocol.json', dict(scope='offline high-supervision same-input EAST comparison',
        train=train, val=val, order=order, source=PARENT_SHA256, seeds=[SEED],
        base_inputs='same HDF5 flat23, same MNM/CIII masking, source scales; raw file grid',
        label_horizon_ms=HORIZON, timely_lead_ms=DEADLINE, validation_stride=STRIDE,
        steps_per_epoch=STEPS, epochs=EPOCHS, batch=BATCH,
        arms=arms, trainable_groups={k: LOCATIONS[k] for k in arms if k in LOCATIONS},
        value_columns={k:[io23.RAW23[j] for j in NUMERIC[k]] for k in arms if k in NUMERIC},
        ordered_control=('oldest 12 position vectors tied; latest 4 and CLS unchanged'
                         if mode=='order' else None),
        online_parent_changed=False, current_outcome_access='after all 360 curves are saved'))
    fitting, heldout = archival_shots(train), archival_shots(val)
    base, _ = load_parent(DEFAULT_PARENT)
    torch.manual_seed(SEED)
    models = [deepcopy(base).cuda() for _ in arms]
    counts = {}
    for k, m in zip(arms, models, strict=True):
        train_scope(m, k)
        counts[k] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    fits = train_models(out, models, fitting, heldout, arms)
    if mode=='numeric_scope':
        write(out/'frozen_scope.json',verify_numeric_scope(base,models,arms))
    write(out/'selected.json', dict(fits=fits, trainable_parameters=counts, train_classes=[
        int(sum(s['is_disrupt']==c for s in fitting)) for c in (True,False)], val_classes=[
        int(sum(s['is_disrupt']==c for s in heldout)) for c in (True,False)]))
    del fitting, heldout
    witnesses = {}
    for i,(sid,f) in enumerate(zip(order, files, strict=True),1):
        s = load_signal(sid)
        _, z = curves(models, s, stride=STRIDE)
        # Use the same causally available scoring schedule as the archival
        # calibration. Hold a previously issued risk until the next issue.
        z = np.repeat(z, STRIDE, axis=1)[:,:len(s['times'])]
        if i in (21,120,360) or sid in (90874,74970,85422):
            n = min(550, len(z[0]))
            cut = {k:s[k][:n] for k in ('times','values','available')}
            _, before = curves(models, cut, stride=STRIDE)
            before = np.repeat(before, STRIDE, axis=1)[:,:n]
            np.testing.assert_allclose(before, z[:,:n], atol=3e-6, rtol=0.)
            for m,a,b in zip(arms, before, z, strict=True):
                for v in fits[m]['calibration'].values():
                    np.testing.assert_array_equal(a>=v['threshold'], b[:n]>=v['threshold'])
            witnesses[str(sid)] = dict(points=n, max_abs=float(abs(before-z[:,:n]).max()))
        with (out/'shots'/f).open('xb') as h:
            np.savez_compressed(h, **dict(zip(arms, z, strict=True)))
        if i in (20,120,240,360):
            print(dict(stage='archival_static_seal', rank=i, seconds=round(monotonic()-start,2)), flush=True)
    write(out/'seal.json', dict(order=order, files=files, prefix=witnesses,
                               score_stride=STRIDE, seconds=monotonic()-start))
    after_seal(out, ref, order, fits, files, arms)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', choices=('core','order','location','numeric_scope'), default='core',
                        help='compare whole representation, time order or the layer allowed to learn')
    args = parser.parse_args()
    run(args.output.resolve(), args.mode)
