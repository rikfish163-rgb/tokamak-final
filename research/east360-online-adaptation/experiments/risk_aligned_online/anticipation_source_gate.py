"""Source-only whole-VAL gate for a changed, anticipatory inference graph.

No source classifier training or J-TEXT online branch. Zero EAST records are
accessed. All controls use the original threshold and CURRENT alarm time.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from . import simple_data
from .anticipatory_readout import AnticipatoryReadout
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.run import DEFAULT_SOURCE, load_parent
from .simple_online import _durable_json, _hash, seal_predictions
from .simple_train import metrics


def evaluate(model, *, output, threshold, device='cuda', data_api=simple_data, require_full_context=False):
    """All declared VAL shots, all valid endpoints; no threshold search."""
    rows = {key: [] for key in ('observed', 'hold', 'forecast')}
    forecasters = {key: AnticipatoryReadout(model, mode=key, require_full_context=require_full_context)
                   for key in ('hold', 'forecast')}
    ids = data_api.source_roles()['val']
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('unique full source VAL required')
    advanced = data_api.advanced_times()
    errors = []
    started = time.monotonic()
    for key in rows:
        (output/key).mkdir()
    for index, sid in enumerate(ids, 1):
        shot = data_api.attach_outcome(data_api.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        if shot['device'] != 'J-TEXT' or shot['shot'] != sid:
            raise ValueError('source identity changed')
        ends = np.flatnonzero(shot['valid'])
        if not len(ends): raise ValueError('empty source VAL shot')
        scores = {key: [] for key in rows}
        signal_errors = []
        for start in range(0, len(ends), 256):
            current = ends[start:start+256]
            windows = [data_api.window(shot, int(t), 128) for t in current]
            values, available = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
            with torch.no_grad():
                actual = model(values, available)
                scores['observed'].extend(actual['logit'].sigmoid().cpu().tolist())
                for key in forecasters:
                    scores[key].extend(forecasters[key](values, available)['logit'].sigmoid().cpu().tolist())
                # Signal skill is a development diagnostic, not a risk label.
                # Equal mean across sampled complete horizons per source shot.
                for offset, end in enumerate(current):
                    end = int(end)
                    if end % 100 or end+25 >= len(shot['values']): continue
                    future_valid = shot['valid'][end+1:end+26]
                    truth = torch.as_tensor(shot['values'][end+1:end+26], device=device)[None]
                    mask = torch.as_tensor(shot['available'][end+1:end+26] & future_valid[:, None], device=device)[None]
                    if not mask.any(): continue
                    held = torch.asinh((torch.where(available[offset:offset+1, -1:],
                        values[offset:offset+1, -1:], model.center)-model.center)/model.scale).expand(-1, 25, -1)
                    signal_errors.append((float(model.forecast_loss(actual['forecast'][offset:offset+1], truth, mask)),
                                          float(model.forecast_loss(held, truth, mask))))
        for key in rows:
            row = dict(shot=sid, is_disrupt=shot['is_disrupt'], down=shot['down'],
                       times=shot['times'][ends], scores=np.asarray(scores[key], np.float32))
            if not np.isfinite(row['scores']).all(): raise FloatingPointError('nonfinite risk')
            seal_predictions(output/key, index, sid, row['times'], row['scores'], arm=key, threshold=threshold)
            rows[key].append(row)
        if signal_errors:
            mean = np.mean(signal_errors, axis=0)
            errors.append(dict(shot=sid, sampled_horizons=len(signal_errors),
                               direct_smooth_l1=float(mean[0]), hold_smooth_l1=float(mean[1])))
        if index % 10 == 0 or index == len(ids):
            print(json.dumps(dict(source_val_completed=index, total=len(ids), elapsed_seconds=time.monotonic()-started)), flush=True)
    counts = {key: metrics(arm, threshold, lead=10.) for key, arm in rows.items()}
    # This is only a source gate. Passing does not qualify EAST performance.
    return dict(metrics=counts, source_gate={key: bool(m['tpr'] >= .9 and m['fpr'] < .2)
                for key, m in counts.items()}, signal_error_per_shot=errors,
                signal_error_mean=({key: float(np.mean([r[key] for r in errors]))
                    for key in ('direct_smooth_l1', 'hold_smooth_l1')} if errors else None),
                elapsed_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, help='explicit compatible archived source TRAIN-only change decoder')
    parser.add_argument('--require-full-context', action='store_true',
                        help='use original observed risk until all128 past rows are measured; does not suppress alarms')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior source gate')
    torch.set_num_threads(4); torch.manual_seed(131)
    loaded, state, gate, _ = load_parent(DEFAULT_SOURCE, 'cuda', persistent_site='embedding')
    if args.decoder is None:
        model = PersistentFfnModel(loaded.base).to('cuda').eval()
    else:
        from .anticipatory_decoder import load_anchored_decoder
        model = load_anchored_decoder(loaded.base, args.decoder,
            parent_sha256=gate['checkpoint_sha256'], train_ids=simple_data.source_roles()['train']).to('cuda').eval()
    head_at_start = {k: v.detach().cpu().clone() for k, v in model.base.forecast_head.state_dict().items()}
    args.output.mkdir(parents=True)
    root = Path(__file__).parent
    files = [Path(__file__), root/'anticipatory_readout.py', root/'anticipatory_decoder.py',
             root/'decoder_intra_20261005/model.py', root/'persistent_ffn.py',
             root/'ffn_intra_20261005/model.py', root/'simple23.py', root/'simple_data.py',
             root/'simple_train.py', root/'simple_online.py', root/'io23.py', root/'redesign_20261004/run.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, str(path.relative_to(root)))
    protocol = dict(scope='SOURCE VAL DEVELOPMENT GATE; NOT JTEXT ONLINE OR EAST EVALUATION',
        source_sha256=gate['checkpoint_sha256'], threshold=float(state['threshold']),
        source_val_ids=simple_data.source_roles()['val'], source_label='max(25ms,AdvancedTime)',
        source_training=False, source_classifier_parameters_changed=False, east_records_accessed=0,
        require_full_context=args.require_full_context,
        context_eligibility=('all128 past rows have a finite observed diagnostic; otherwise original observed risk'
                            if args.require_full_context else 'historical ungated virtual window'),
        archived_decoder=(None if args.decoder is None else dict(path=str(args.decoder.resolve()),
                           sha256=_hash(args.decoder), source_only=True, kind='last_observation_plus_direct_increment')),
        virtual_window='103 real past +25 direct predicted; current availability repeated for future mask',
        forecast_coordinates='source-asinh inverse exactly once; not physical unit calibration',
        autoregressive=False, new_risk_head=False, alarm_time='actual current endpoint; no +25ms shift',
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json', protocol)
    result = evaluate(model, output=args.output, threshold=state['threshold'],
                      require_full_context=args.require_full_context)
    if (model.memory.any() or _hash(DEFAULT_SOURCE) != protocol['source_sha256']
            or not all(torch.equal(v.detach().cpu(), state['model'][k]) for k, v in model.base.state_dict().items()
                       if not k.startswith('forecast_head.'))
            or not all(torch.equal(v.detach().cpu(), head_at_start[k]) for k, v in model.base.forecast_head.state_dict().items())):
        raise ValueError('qualified source or internal state changed')
    _durable_json(args.output/'result.json', result)
    print(json.dumps(dict(metrics={k: {p: m[p] for p in ('tp', 'fp', 'n_disrupt', 'n_safe', 'tpr', 'fpr')}
        for k, m in result['metrics'].items()}, source_gate=result['source_gate'],
        signal_error_mean=result['signal_error_mean'])), flush=True)


if __name__ == '__main__': main()
