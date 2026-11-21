"""Source-only semantic supervision of a direct signal CHANGE decoder.

Actual future risk is a frozen source TRAIN teacher, never an EAST online
input. Only an external forecast-head copy is optimized. The qualified risk
network, normalization, parent artifact and alarm threshold remain fixed.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data
from .anticipatory_decoder import load_anchored_decoder
from .anticipatory_readout import future_window, complete_context
from .decoder_intra_20261005.model import last_observation
from .decoder_intra_20261005.source import (normalized_level_loss, activity_thresholds,
    sampling_pools, sample_indices)
from .ffn_intra_20261005.source import split_shots
from .redesign_20261004.prepare_prior import (DEFAULT_PARENT, load_parent, load_source_train,
    classifier_state, assert_classifier_unchanged)
from .simple_online import _durable_json, _hash

DEFAULT_DECODER = Path(__file__).parent/'runs/anchored_head_only_checked_20261005/decoder.pt'


def semantic_loss(model, values, available, prediction, teacher_logits):
    if (teacher_logits.shape != (len(values),) or not torch.isfinite(teacher_logits).all()
            or teacher_logits.device != values.device):
        raise ValueError('finite matching detached source teacher logits required')
    virtual, mask = future_window(values, available, prediction, model.center, model.scale)
    logits = model(virtual, mask)['logit']
    teacher = teacher_logits.detach().sigmoid()
    negative_entropy = torch.special.xlogy(teacher, teacher) + torch.special.xlogy(1-teacher, 1-teacher)
    loss = F.binary_cross_entropy_with_logits(logits, teacher, reduction='none') + negative_entropy
    if not torch.isfinite(loss).all(): raise FloatingPointError('nonfinite semantic KL')
    return loss


def cache_events(model, shots, *, require_full_context=False):
    pieces = {key: [] for key in ('features', 'anchor', 'target', 'mask', 'values', 'available', 'teacher_logits')}
    rows = []
    for index, shot in enumerate(shots, 1):
        if shot['device'] != 'J-TEXT': raise ValueError('offline source cache only')
        origins = np.arange(0, max(0, len(shot['times'])-25), 26)
        origins = [int(o) for o in origins if shot['valid'][o:o+26].all()]
        if require_full_context:
            observed = (shot['available'] & np.isfinite(shot['values'])).any(-1)
            origins = [o for o in origins if o >= 127 and observed[o-127:o+1].all()]
        for start in range(0, len(origins), 256):
            ends = origins[start:start+256]
            def windows(endpoints):
                pairs = [simple_data.window(shot, o, 128) for o in endpoints]
                return [torch.as_tensor(np.stack(v), device=model.center.device) for v in zip(*pairs)]
            x, a = windows(ends)
            if require_full_context and not complete_context(x, a).all():
                raise ValueError('source training and inference context eligibility differ')
            actual_future, actual_mask = windows([o+25 for o in ends])
            captured = []
            handle = model.base.forecast_head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
            try:
                with torch.no_grad(): model(x, a)
            finally:
                handle.remove()
            if len(captured) != 1: raise RuntimeError('one original head feature capture required')
            with torch.no_grad(): teacher = model(actual_future, actual_mask)['logit']
            anchor, seen = last_observation(x, a, model.center, model.scale)
            raw, mask = actual_future[:, -25:], actual_mask[:, -25:]
            target = torch.asinh((torch.where(mask, raw, model.center)-model.center)/model.scale)
            common = mask & seen[:, None]
            activity = normalized_level_loss(anchor[:, None].expand_as(target), target, common)
            for j, origin in enumerate(ends):
                if not common[j].any(): continue
                rows.append(dict(shot=int(shot['shot']), origin=origin, label=int(shot['is_disrupt']), activity=float(activity[j])))
                for key, value in zip(pieces, (captured[0], anchor, target, mask, x, a, teacher)):
                    pieces[key].append(value[j].detach().cpu())
        if index % 100 == 0 or index == len(shots):
            print(json.dumps(dict(source_cached=index, shots=len(shots), events=len(rows))), flush=True)
    if not rows: raise ValueError('no fully observed source horizons')
    return {key: torch.stack(value) for key, value in pieces.items()}, rows


def batch_losses(model, head, cache, indices):
    predicted = head(cache['features'][indices]).reshape(-1, 25, 23) + cache['anchor'][indices, None]
    signal = normalized_level_loss(predicted, cache['target'][indices], cache['mask'][indices])
    semantic = semantic_loss(model, cache['values'][indices], cache['available'][indices], predicted,
                              cache['teacher_logits'][indices])
    return signal, semantic


def report(model, head, cache, rows, pools):
    signal, semantic = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            s, r = batch_losses(model, head, cache, slice(start, start+256))
            signal.extend(s.cpu().tolist()); semantic.extend(r.cpu().tolist())
    def balanced(values):
        return float(np.mean([np.mean([np.mean([values[i] for i in indices]) for indices in group.values()])
                              for group in pools.values()]))
    return dict(signal=balanced(signal), semantic=balanced(semantic),
                combined=balanced(signal)+.20*balanced(semantic),
                weighting='class/activity/shot/event uniform; source development only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-full-context', action='store_true',
                        help='fit only origins with 128 measured past rows; old experiment default unchanged')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('preserve source experiment')
    torch.set_num_threads(4); torch.manual_seed(131)
    started = time.monotonic()
    parent_sha, decoder_sha = _hash(DEFAULT_PARENT), _hash(DEFAULT_DECODER)
    base, _ = load_parent(DEFAULT_PARENT)
    initial_classifier = classifier_state(base)
    source, train_ids = load_source_train()
    fit, selection = split_shots(source, 131)
    fit_ids, selection_ids = [s['shot'] for s in fit], [s['shot'] for s in selection]
    model = load_anchored_decoder(base, DEFAULT_DECODER, parent_sha256=parent_sha, train_ids=train_ids).to('cuda').eval()
    model.requires_grad_(False)
    initial = {key: value.detach().cpu().clone() for key, value in model.base.forecast_head.state_dict().items()}
    args.output.mkdir(parents=True)
    fit_cache, fit_rows = cache_events(model, fit, require_full_context=args.require_full_context)
    sel_cache, sel_rows = cache_events(model, selection, require_full_context=args.require_full_context)
    del source, fit, selection
    thresholds = activity_thresholds(fit_rows)
    fit_pools, sel_pools = sampling_pools(fit_rows, thresholds), sampling_pools(sel_rows, thresholds)
    fit_cache = {key: value.to('cuda') for key, value in fit_cache.items()}
    sel_cache = {key: value.to('cuda') for key, value in sel_cache.items()}
    protocol = dict(source_only=True, parent_sha256=parent_sha, initial_decoder_sha256=decoder_sha,
        decoder_kind='last_observation_plus_direct_increment', source_train_ids=train_ids,
        fit_ids=fit_ids, selection_ids=selection_ids,
        eligible_fit_ids=sorted({r['shot'] for r in fit_rows}), eligible_selection_ids=sorted({r['shot'] for r in sel_rows}),
        source_classifier_frozen=True, source_scaler_frozen=True, original_source_labels_unchanged=True,
        no_east_reads=True, source_holdout_used_for_training=False,
        source_label='max(25ms,AdvancedTime)', seed=131, steps=200, batch_size=64, lr=.0003,
        require_full_context=args.require_full_context,
        context_eligibility=('all128 past rows have a finite observed diagnostic' if args.require_full_context
                             else 'past padding allowed; historical default'),
        sampling='class/activity/shot/event uniform; TRAIN-only full mature H25 valid at o..o+25; stride26',
        activity_thresholds=thresholds, fit_events=len(fit_rows), selection_events=len(sel_rows),
        teacher='same frozen source risk on actual future W128, source TRAIN only',
        head_only=True, semantic_weight=.20, current_alarm_time=True,
        qualification='full VAL160 at original threshold; no EAST before source gate',
        scope='SOURCE AUXILIARY DEVELOPMENT; no new classifier or independent validation')
    root = Path(__file__).parent
    files = [Path(__file__), root/'anticipatory_readout.py', root/'anticipatory_decoder.py',
             root/'simple23.py', root/'persistent_ffn.py', root/'ffn_intra_20261005/model.py',
             root/'ffn_intra_20261005/source.py', root/'redesign_20261004/prepare_prior.py',
             root/'decoder_intra_20261005/model.py', root/'decoder_intra_20261005/source.py',
             root/'simple_data.py', root/'io23.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, str(path.relative_to(root)))
    protocol['code_snapshot_sha256'] = _hash(args.output/'code_snapshot.zip')
    _durable_json(args.output/'protocol.json', protocol)
    summaries = {}
    for arm in ('signal', 'semantic'):
        directory = args.output/arm; directory.mkdir()
        head = copy.deepcopy(model.base.forecast_head).requires_grad_(True)
        head.load_state_dict(initial)
        optimizer = torch.optim.Adam(head.parameters(), lr=.0003)
        rng = np.random.default_rng(131)
        first = report(model, head, sel_cache, sel_rows, sel_pools)
        key = 'signal' if arm == 'signal' else 'combined'
        best_loss, best_step = first[key], 0
        best = copy.deepcopy(initial)
        history = [dict(step=0, selection=first)]
        for step in range(1, 201):
            indices = sample_indices(fit_pools, rng, 64)
            signal, semantic = batch_losses(model, head, fit_cache, indices)
            loss = (signal if arm == 'signal' else signal+.20*semantic).mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite source objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 50 == 0:
                measured = report(model, head, sel_cache, sel_rows, sel_pools)
                history.append(dict(step=step, selection=measured))
                if measured[key] < best_loss:
                    best_loss, best_step = measured[key], step
                    best = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                print(json.dumps(dict(arm=arm, step=step, selection=measured)), flush=True)
        child_protocol = dict(protocol, objective=('signal only' if arm == 'signal' else 'signal+.20 Bernoulli KL'),
                              selected_by='minimum source selection '+key+' loss including epoch0')
        head.load_state_dict(best)
        summaries[arm] = dict(selected_step=best_step, selection=report(model, head, sel_cache, sel_rows, sel_pools))
        _durable_json(directory/'history.json', history)
        with (directory/'decoder.pt').open('xb') as handle:
            torch.save(dict(forecast_head=best, protocol=child_protocol, selected_step=best_step), handle)
        _durable_json(directory/'protocol.json', child_protocol)
    assert_classifier_unchanged(model.base, initial_classifier)
    if (model.memory.any() or _hash(DEFAULT_PARENT) != parent_sha or _hash(DEFAULT_DECODER) != decoder_sha
            or any(not torch.equal(v.cpu(), initial[k]) for k, v in model.base.forecast_head.state_dict().items())):
        raise ValueError('source parent, starting head or internal state changed')
    result = dict(arms=summaries, source_classifier_unchanged=True,
                  elapsed_seconds=time.monotonic()-started, east_accessed=False)
    _durable_json(args.output/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
