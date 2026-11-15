"""Compare same-recipe J-TEXT seeds and their fixed EAST transfers, without adaptation."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import io23, simple_data, simple_train
from .causal_input import apply_input_contract
from .commissioning_failure_modes_20261111 import consensus, transitions
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_repair_20261104.east_evaluation import BASELINE, configure, load_inputs
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple23 import Config, Simple23
from .simple_online import predict_shot
from .source_anticipation_20261104.static_stream import timing_decomposition
from .static_threshold_diagnostic_20261104 import CAPS, shot_maximum
from .warmstart_alignment_20261022.alignment import transform


SEEDS = (131, 137, 149)
VIEWS = ('direct', 'aligned_after20')
STARTUP = 20


def capacity(rows, startup, cap):
    """Post-seal upper envelope for one suffix threshold; first 20 alarms are fixed."""
    cold = rows[:startup]
    future = rows[startup:]
    tp = [r['shot'] for r in cold if r['timely']]
    fp = [r['shot'] for r in cold if not r['disruptive'] and r['alarm']]
    safe = sorted((r['maximum'] for r in future if not r['disruptive']), reverse=True)
    budget = int(np.floor(cap * sum(not r['disruptive'] for r in rows))) - len(fp)
    if budget < 0:
        return None
    cutoff = float(np.nextafter(np.float32(safe[budget]), np.float32(np.inf))) if budget < len(safe) else 0.
    for r in future:
        if r['maximum'] is not None and r['maximum'] >= cutoff:
            (tp if r['disruptive'] else fp).append(r['shot'])
    return dict(cutoff=cutoff, tp=len(tp), fp=len(fp), timely_shots=tp, false_alarm_shots=fp,
                deployable=False)


def without_rows(result):
    return {k: v for k, v in result.items() if k != 'predictions'}


def run(root):
    configure()
    locked = load_inputs()
    roles = io23.source_roles()
    old_source = read(DEFAULT_PARENT.with_name('test220.json'))
    if old_source['provenance']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('qualified source report differs')
    configs, parents, specs = {}, {}, {}
    for seed in SEEDS:
        name = str(seed)
        folder = root / f'base_{seed}'
        cfg = read(folder / 'config.json')
        if cfg['seed'] != seed:
            raise ValueError('base seed manifest differs')
        configs[name] = {k: v for k, v in cfg.items() if k not in ('seed', 'output')}
        receipt = read(folder / 'hashes.json')
        if any(sha(folder / path) != receipt[field] for path, field in (
                ('source.pt', 'source_pt_sha256'), ('test220.json', 'test220_sha256'),
                ('config.json', 'config_sha256'))):
            raise ValueError('base receipt differs')
        trained = root / f'aligned_{seed}'
        spec, validation = (read(trained / k) for k in ('protocol.json', 'validation.json'))
        if (spec['train_ids'] != roles['train'] or spec['val_ids'] != roles['val']
                or spec['parent_sha256'] != receipt['source_pt_sha256']
                or sha(trained / 'source.pt') != validation['checkpoint_sha256']):
            raise ValueError('meta-stage lineage differs')
        specs[name] = {k: v for k, v in spec.items() if k != 'parent_sha256'}
        parents[name] = trained / 'source.pt'
    if any(c != configs['131'] for c in configs.values()) or any(s != specs['131'] for s in specs.values()):
        raise ValueError('only J-TEXT base random seed may vary across new fits')

    models, cutoffs = {}, {}
    for name, path in parents.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        cfg = Config(**saved['model_config'])
        net = Simple23(cfg, saved['source_center'], saved['source_scale'])
        net.load_state_dict(saved['model'])
        models[name] = net.cuda().eval().requires_grad_(False)
        cutoffs[name] = float(saved['threshold'])
    cutoffs['qualified'] = locked['baseline_protocol']['threshold']
    frozen = {k: state_digest(v) for k, v in models.items()}
    out = root / 'replay'
    out.mkdir()
    (out / 'shots').mkdir()
    write(out / 'protocol.json', dict(
        scope='CONTROLLED_SOURCE_SEED_EAST_STATIC_DEVELOPMENT',
        seeds=SEEDS, fixed_meta_seed=specs['131']['seed'], order=locked['ids'],
        source_val_cutoffs=cutoffs, startup=STARTUP, base_recipe=configs['131'],
        parent_sha256={k: sha(v) for k, v in parents.items()},
        qualified_source_sha256=PARENT_SHA256, same_fixed_east_map=True,
        label='J-TEXT HDF5 max(25ms,AdvancedTime); scoring strict >10ms / >30ms',
        inference_code_sha256=sha(Path(__file__)), east_training_labels=0, state_updates=0))
    test = simple_train.load_shots(roles['test'], io23.advanced_times())
    source = {k: simple_train.metrics(simple_train.score(net, test, 'cuda'), cutoffs[k])
              for k, net in models.items()}
    source['qualified'] = old_source['jtext']
    jtext = {k: [dict(r, disruptive=r['is_disrupt']) for r in v['predictions']] for k, v in source.items()}
    if any([r['shot'] for r in v] != roles['test'] for v in jtext.values()):
        raise ValueError('source TEST220 order changed')
    del test
    write(out / 'jtext_test220.json', source)
    print({k: (v['tp'], v['fp']) for k, v in source.items()}, flush=True)

    hashes = {}
    archive_seals = read(BASELINE / 'completed.json')['seals']
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('EAST outcomes still sealed')):
        for i, sid in enumerate(locked['ids'], 1):
            original = BASELINE / 'shots' / f'{i:04d}_{sid}.npz'
            if (sha(original) != archive_seals[i - 1]['curve_sha256']
                    or sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]):
                raise ValueError(f'raw or reference signal changed: {sid}')
            raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            with np.load(original, allow_pickle=False) as f:
                np.testing.assert_array_equal(f['times'], raw['times'])
                curves = {'qualified_direct': f['unmapped'], 'qualified_aligned_after20': f['map_after20']}
            direct = {k: raw[k] for k in ('values', 'available')}
            aligned = transform(direct, locked['mapping']) if i > STARTUP else direct
            if not np.array_equal(aligned['available'], direct['available']):
                raise ValueError('mapping changed actual sensor availability')
            for k, net in models.items():
                for mode, signal in zip(VIEWS, (direct, aligned), strict=True):
                    curve, updates = predict_shot(net, dict(signal, shot=sid), intra=False, device='cuda')
                    if updates:
                        raise ValueError('static replay mutated an online state')
                    curves[f'{k}_{mode}'] = curve
                    if i <= STARTUP:
                        curves[f'{k}_{VIEWS[1]}'] = curve
                        break
            path = out / 'shots' / original.name
            with path.open('xb') as f:
                np.savez_compressed(f, times=raw['times'], **curves)
            hashes[path.name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed_east', shots=i), flush=True)
    if any(state_digest(v) != frozen[k] for k, v in models.items()):
        raise ValueError('frozen source state changed')
    write(out / 'seal.json', dict(curve_sha256=hashes, persistent_updates=0))
    east = {f'{k}_{v}': [] for k in cutoffs for v in VIEWS}
    for name, digest in hashes.items():
        path = out / 'shots' / name
        if sha(path) != digest:
            raise ValueError('post-seal curve changed')
        sid = int(path.stem.split('_')[-1])
        d, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as f:
            for k, rows in east.items():
                t, curve = f['times'], f[k]
                rows.append(dict(shot=sid, disruptive=d, **score_curve(t, curve, d, down, cutoffs[k.split('_')[0]]),
                                 maximum=shot_maximum(t, curve, d, down)))
    original = read(BASELINE / 'per_shot.json')
    for k, old in zip(VIEWS, ('unmapped', 'map_after20'), strict=True):
        if [{a: b for a, b in v.items() if a != 'maximum'} for v in east[f'qualified_{k}']] != original[old]:
            raise ValueError('archived qualified replay scores differ')
    envelopes = {k: {str(c): capacity(v, STARTUP, c) for c in CAPS} for k, v in east.items()}
    cohort_disruptive = {r['shot'] for r in east['131_direct'] if r['disruptive']}
    at20 = {}
    for mode in VIEWS:
        cells = [envelopes[f'{s}_{mode}']['0.2'] for s in SEEDS]
        if all(v is not None for v in cells):
            tp, fp = ([set(v[k]) for v in cells] for k in ('timely_shots', 'false_alarm_shots'))
            at20[mode] = dict(always_untimely=sorted(cohort_disruptive - set.union(*tp)),
                              always_false_alarm=sorted(set.intersection(*fp)))
    result = dict(
        status='COMPLETED_CONTROLLED_SOURCE_SEED_STATIC_REPLAY_DEVELOPMENT',
        jtext={k: without_rows(v) for k, v in source.items()},
        jtext_three_new_seeds=consensus({str(s): jtext[str(s)] for s in SEEDS}),
        east={k: timing_decomposition(v) for k, v in east.items()},
        east_three_new_seeds={v: consensus({str(s): east[f'{s}_{v}'] for s in SEEDS}) for v in VIEWS},
        transitions={k: transitions(east[f'{k}_direct'], east[f'{k}_aligned_after20']) for k in cutoffs},
        full_flow_retrospective_threshold_envelope=envelopes, east_shared_failures_at_20_capacity=at20)
    write(out / 'east_per_shot.json', east)
    write(out / 'result.json', result)
    print({k: (v['tp'], v['fp']) for k, v in result['east'].items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training', type=Path, required=True)
    run(cli.parse_args().training.resolve())
