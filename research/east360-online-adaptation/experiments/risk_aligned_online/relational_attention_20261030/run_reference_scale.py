"""Paired scale-aware reference ablation of the frozen relational experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import normalized
from .run import ARMS, chunks, evaluate, fixed_controls, shot_summary, windows


def acceptance(groups, pred):
    reasons = []
    for group, minimum, better in [('safe', 3, 2), ('danger', 10, None)]:
        g = groups['real'][group]
        if g['supported_shots'] < minimum:
            reasons.append(group+': insufficient supported shots')
        if g['mean_delta_bce'] is None or g['mean_delta_bce'] >= 0:
            reasons.append(group+': real did not improve cold')
        if g['improved'] < (better if better else (g['shots']+1)//2):
            reasons.append(group+': insufficient improved shots')
        for arm in ARMS[1:]:
            if (g['mean_delta_bce'] is None or groups[arm][group]['mean_delta_bce'] is None
                    or g['mean_delta_bce'] >= groups[arm][group]['mean_delta_bce']):
                reasons.append(group+': real did not beat '+arm)
    valid = pred['direction_nonzero']
    cosine = float(np.median(pred['real_reverse_cosine'][valid])) if valid.any() else None
    different = float((np.abs(pred['real'][valid]-pred['reverse'][valid]) > 1e-4).mean()) if valid.any() else None
    if cosine is None or cosine >= .99 or different < .5:
        reasons.append('directional/function specificity gate failed')
    return dict(full_stream_gate=not reasons, reasons=reasons,
                real_reverse_median_cosine=cosine, distinct_future_logit_fraction=different)


def execute(previous, output):
    from .standardized_reference import fit_reference_standardized
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree probe output required')
    original = json.loads((previous/'protocol.json').read_text())
    completed = json.loads((previous/'complete.json').read_text())
    for name, digest in completed['artifact_sha256'].items():
        if sha(previous/name) != digest:
            raise ValueError('original artifact changed: '+name)
    for name, digest in original['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise ValueError('original implementation changed: '+name)
    reference = Path(original['reference'])
    for name, digest in original['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('cold dependency changed: '+name)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping = json.loads((reference/'input_map.json').read_text())
    hashes = json.loads((reference/'input_hashes.json').read_text())
    ids = original['warmup_ids']
    records = json.loads((previous/'records.json').read_text())
    folds = json.loads((previous/'folds.json').read_text())
    output.mkdir()
    files = {io23.ROOT/'RELATIONAL_REFERENCE_SCALE_PLAN_20261030.md',
             io23.ROOT/'tests/test_relational_standardized_reference.py',
             io23.ROOT/'tests/test_relational_reference_run.py', *Path(__file__).parent.glob('*.py')}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(original, previous=str(previous), previous_complete_sha256=sha(previous/'complete.json'),
        reference_spec='shot-weighted covariate standardization before ridge1, recompute residual D',
        posthoc_design_after_v1=True, outcome_groups_from_previous_records=True,
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)})
    _durable_json(output/'protocol.json', protocol)
    _durable_json(output/'folds.json', folds)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(io23.ROOT)))
    started = time.monotonic()
    data, later, first = {}, {}, {}
    for sid in ids:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise ValueError('raw cold20 changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        sig = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        rows = [r for r in records if r['shot'] == sid]
        data[sid] = windows(sig, [r['maturity'] for r in rows])
        later[sid] = windows(sig, [r['query'] for r in rows])
        z, m = normalized(net, torch.as_tensor(sig['values'][:256], device='cuda'),
                          torch.as_tensor(sig['available'][:256], device='cuda'))
        first[sid] = (z.detach().cpu(), m.cpu())
    all_pred = {}
    for spec in folds:
        ref_cpu = fit_reference_standardized([first[s] for s in spec['fit']])
        ref = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in ref_cpu.items()}
        mean, mean_g = fixed_controls(net, data, spec['fit'], ref, slow)
        torch.save(dict(reference=ref_cpu, mean_target=mean.cpu(), mean_gradient=mean_g.cpu(),
                        fit_ids=spec['fit'], held_ids=spec['held']), output/f"fold{spec['fold']}.pt")
        for sid in spec['held']:
            parts = {}
            for sl, d in chunks(data[sid], 'cuda'):
                q = {k: v[sl].cuda() for k, v in later[sid].items()}
                pred = evaluate(net, d, q, ref, mean, mean_g, slow)
                for k, v in pred.items():
                    parts.setdefault(k, []).append(v)
            merged = {k: np.concatenate(v) for k, v in parts.items()}
            old = np.load(previous/f'{sid}_predictions.npz')
            if not np.array_equal(merged['cold'], old['cold']):
                raise AssertionError('paired baseline not exact')
            if not np.array_equal(merged['edge_mask'], old['edge_mask']):
                raise AssertionError('paired support changed')
            np.savez_compressed(output/f'{sid}_predictions.npz', **merged)
            _durable_json(output/f'{sid}_seal.json', dict(sha256=sha(output/f'{sid}_predictions.npz')))
            all_pred[sid] = merged
            print(dict(stage='held', fold=spec['fold'], shot=sid, total=len(merged['cold'])), flush=True)
    pred = {k: np.concatenate([all_pred[s][k] for s in ids]) for k in all_pred[ids[0]]}
    rows, groups = shot_summary(records, pred)
    _durable_json(output/'records.json', records)
    _durable_json(output/'per_shot.json', rows)
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(not torch.equal(v.cpu(), frozen[k]) for k, v in net.state_dict().items()):
        raise AssertionError('source parameters changed')
    if any(p.grad is not None for p in net.parameters()):
        raise AssertionError('frozen model accumulated gradients')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('code changed during run: '+name)
    for name, digest in protocol['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise AssertionError('parent changed during run: '+name)
    if any(sha(io23.path_for('EAST', sid)) != hashes[str(sid)] for sid in ids):
        raise AssertionError('raw input changed during run')
    result = dict(complete=True, source_unchanged=True, full_stream_run=False,
        suffix_labels_read=0, hdf5_outcome_reads=0, opportunities=len(records),
        paired_baseline_exact=True, paired_support_exact=True, groups=groups,
        supported=int(pred['support'].sum()), matched_nonzero=int(pred['direction_nonzero'].sum()),
        **acceptance(groups, pred), seconds=time.monotonic()-started,
        artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    _durable_json(output/'complete.json', result)
    print({k: v for k, v in result.items() if k != 'artifact_sha256'}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    execute(args.previous.absolute(), args.output.absolute())
