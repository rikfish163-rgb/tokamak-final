"""Information-block ablation of the existing conditional event-family probe.

Only new linear probe heads are fitted. No source backbone or EAST data changes.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..event_prefix_probe import fit_probe, DEFAULT_PARENT, PARENT_SHA256, DEFAULT_LABELS, LABEL_SHA256, sha256
from ..simple_online import _durable_json
from ..io23 import ROOT

BLOCKS = dict(mask=(4,), mean=(1, 4), moments=(1, 2, 4), change=(3, 4),
              variation=(2, 3, 4), full=(0, 1, 2, 3, 4))
CACHE = ROOT / 'outputs/event_prefix_probe_20261021'
PLAN = ROOT / 'docs/research/source_event_rebuild_20261023/事件标签_水平与变化信息消融_事前约定.md'


def select_features(raw, mode):
    if (mode not in BLOCKS or raw.ndim != 2 or raw.shape[1] != 115
            or not np.isfinite(raw).all()):
        raise ValueError('finite 115-column raw cache and declared information blocks required')
    selected = np.zeros_like(raw)
    for block in BLOCKS[mode]:
        sl = slice(23 * block, 23 * (block + 1))
        selected[:, sl] = raw[:, sl]
    return selected


def independent_f1(y, prediction, event_ids):
    """Direct weighted confusion matrix, independent of sklearn metric calls."""
    _, inverse, counts = np.unique(event_ids, return_inverse=True, return_counts=True)
    matrix = np.zeros((6, 6), dtype=np.float64)
    np.add.at(matrix, (y, prediction), 1. / counts[inverse])
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(6), where=denominator > 0)
    return matrix, float(f1.mean())


def main(output):
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    read = lambda p: json.loads(p.read_text())
    old = read(CACHE / 'completed.json')
    cache_sha = sha256(CACHE / 'features.npz')
    assert cache_sha == old['feature_cache_sha256']
    assert sha256(DEFAULT_PARENT) == PARENT_SHA256
    assert sha256(DEFAULT_LABELS) == LABEL_SHA256
    with np.load(CACHE / 'features.npz', allow_pickle=False) as archive:
        cache = {k: archive[k] for k in archive.files}
    for i, a in enumerate(('train', 'val', 'test')):
        for b in ('train', 'val', 'test')[i + 1:]:
            for key in ('shot', 'event_id'):
                assert not set(cache[key][cache['role'] == a]) & set(cache[key][cache['role'] == b])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True)
    protocol = dict(scope='DEVELOPMENT_ONLY_CONDITIONAL_EVENT_INFORMATION_ABLATION',
        cache_sha256=cache_sha, source_sha256=PARENT_SHA256, labels_sha256=LABEL_SHA256,
        source_online_branch=False, east_signals_read=False, backbone_trained=False,
        teacher='offline whole-record K6, historical source-test exposure',
        unknown_intervals='excluded, not normal', feature_groups=BLOCKS,
        head='115->6 linear; absent groups zero, active information dimensions differ',
        fit='TRAIN only; event and class balanced; fixed steps, no selection',
        seeds=[131, 132, 133], steps=1024, batch_size=512, lr=.003,
        plan_sha256=sha256(PLAN), code_sha256=sha256(Path(__file__)),
        reused_fit_code_sha256=sha256(Path(__file__).parent.parent / 'event_prefix_probe.py'))
    _durable_json(output / 'protocol.json', protocol)
    old_heads = torch.load(CACHE / 'probe_heads.pt', weights_only=True, map_location='cpu')
    runs, checkpoints = [], {}
    start = time.monotonic()
    for mode, shuffled in [(m, False) for m in BLOCKS] + [('full', True)]:
        cache[mode] = select_features(cache['raw'], mode)
        for seed in protocol['seeds']:
            result, checkpoint, prediction = fit_probe(cache, feature=mode, seed=seed,
                shuffled=shuffled, steps=1024, device='cuda')
            name = f'{mode}_{"shuffled" if shuffled else "true"}_{seed}'
            for role in ('val', 'test'):
                use = cache['role'] == role
                matrix, f1 = independent_f1(cache['target'][use], prediction[use], cache['event_id'][use])
                np.testing.assert_allclose(matrix, result['results'][role]['confusion_event_weighted'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(f1, result['results'][role]['event_weighted_macro_f1'], atol=1e-12, rtol=0)
            if mode == 'full' and not shuffled:
                reference = f'raw_true_{seed}'
                np.testing.assert_array_equal(prediction, np.load(CACHE / f'{reference}_predictions.npy'))
                for key, value in checkpoint.items():
                    torch.testing.assert_close(value, old_heads[reference][key], atol=0, rtol=0)
            checkpoints[name] = checkpoint
            np.save(output / f'{name}_predictions.npy', prediction)
            runs.append(result)
            _durable_json(output / 'results.json', runs)
            print(dict(name=name, test_f1=result['results']['test']['event_weighted_macro_f1'],
                       seconds=round(time.monotonic()-start, 2)), flush=True)
    torch.save(checkpoints, output / 'probe_heads.pt')
    assert sha256(DEFAULT_PARENT) == PARENT_SHA256
    assert sha256(CACHE / 'features.npz') == cache_sha
    summary = []
    for mode, shuffled in [(m, False) for m in BLOCKS] + [('full', True)]:
        rows = [r for r in runs if r['feature'] == mode and r['shuffled'] == shuffled]
        means = {role: float(np.mean([r['results'][role]['event_weighted_macro_f1'] for r in rows])) for role in ('val', 'test')}
        summary.append(dict(feature=mode, shuffled=shuffled, mean_f1=means))
    _durable_json(output / 'summary.json', summary)
    _durable_json(output / 'completed.json', dict(status='COMPLETED_DIAGNOSTIC_ONLY',
        seconds=time.monotonic()-start, fits=len(runs), independent_metric_checks=2*len(runs),
        exact_original_full_head_reproductions=3, source_and_cache_unchanged=True,
        artifact_sha256={str(f.relative_to(output)): sha256(f) for f in output.rglob('*') if f.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
