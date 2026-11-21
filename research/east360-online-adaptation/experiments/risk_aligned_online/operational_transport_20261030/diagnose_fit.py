"""CPU-only geometry of the saved cold20_v2 fit; never fits or loads a source model.

Run from the worktree with PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='':
python -m experiments.risk_aligned_online.operational_transport_20261030.diagnose_fit
The only output is cold20_v2/fit_geometry.json. No full-stream artifacts are read.
"""

import hashlib
import json
from pathlib import Path
import platform

import torch

from .learning import predict_rule


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / 'probe_outputs/operational_transport_cold20_v2'
ARMS = ('operational', 'point', 'shuffled')


def unit_rows(value):
    value = value.to(dtype=torch.float64)
    scale = value.abs().amax(dim=-1, keepdim=True)
    scaled = value / torch.where(scale > 0, scale, torch.ones_like(scale))
    norm = scaled.norm(dim=-1, keepdim=True)
    return scaled / torch.where(norm > 0, norm, torch.ones_like(norm))


def row_cosine(a, b):
    """Float64 directional cosine; zero vectors are undefined, not orthogonal."""
    a, b = torch.broadcast_tensors(a.double(), b.double())
    valid = (a != 0).any(dim=-1) & (b != 0).any(dim=-1)
    cosine = (unit_rows(a) * unit_rows(b)).sum(dim=-1).clamp(-1, 1)
    return torch.where(valid, cosine, torch.full_like(cosine, float('nan')))


def summarize(values, weights):
    values, weights = values.double(), weights.double()
    valid = torch.isfinite(values)
    v, w = values[valid], weights[valid]
    result = dict(count=len(values), defined=int(valid.sum()), undefined=int((~valid).sum()),
                  positive=int((v > 0).sum()), negative=int((v < 0).sum()), zero=int((v == 0).sum()),
                  defined_weight_fraction=float(w.sum()/weights.sum()))
    result.update({key: None for key in ('window_mean', 'weighted_mean', 'min', 'p05', 'median', 'p95', 'max')})
    if len(v):
        result.update(window_mean=float(v.mean()), weighted_mean=float((v*w).sum()/w.sum()))
        result.update(zip(('min', 'p05', 'median', 'p95', 'max'),
                          torch.quantile(v, v.new_tensor([0., .05, .5, .95, 1.])).tolist()))
    return result


def direction_summary(a, b, weights):
    cosine = row_cosine(a, b)
    result = summarize(cosine, weights)
    for name, threshold in (('same_within_1e6', 1-1e-6), ('cos_ge_0_99', .99), ('cos_ge_0_9', .9)):
        result[name] = int((cosine >= threshold).sum())
    return result


def balanced_weights(records):
    """Exactly the class/shot/window weighting in prepare.py, without importing it."""
    classes, counts = {}, {}
    for record in records:
        sid, flag = record['shot'], record['disruptive']
        if flag not in (False, True) or (sid in classes and classes[sid] != flag):
            raise ValueError('each shot must have one consistent binary outcome')
        classes[sid] = bool(flag)
        counts[sid] = counts.get(sid, 0) + 1
    class_counts = {flag: sum(v == flag for v in classes.values()) for flag in (False, True)}
    if not all(class_counts.values()):
        raise ValueError('both shot outcome classes required')
    return torch.tensor([.5/class_counts[bool(r['disruptive'])]/counts[r['shot']] for r in records],
                        dtype=torch.float64)


def half_roll_indices(records):
    order = torch.arange(len(records))
    for sid in dict.fromkeys(r['shot'] for r in records):
        indices = torch.tensor([i for i, r in enumerate(records) if r['shot'] == sid])
        order[indices] = torch.roll(indices, len(indices)//2)
    return order


def ratio(numerator, denominator):
    return torch.where(denominator > 0, numerator/denominator,
                       torch.full_like(numerator, float('nan')))


def scalar_ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else None


def mapping_metrics(mapped, target, mean, weights):
    mapped, target, mean = mapped.double(), target.double(), mean.double()
    weights = weights.double()/weights.sum()
    delta = mapped-mean
    norm, delta_norm, constant_norm = mapped.norm(dim=1), delta.norm(dim=1), mean.norm()
    own, constant = row_cosine(mapped, target), row_cosine(mean, target)
    return dict(
        target_cosine=direction_summary(mapped, target, weights),
        constant_target_cosine=direction_summary(mean, target, weights),
        cosine_gain_over_constant=summarize(own-constant, weights),
        mapped_constant_cosine=direction_summary(mapped, mean, weights),
        mapped_norm=summarize(norm, weights), constant_norm=float(constant_norm),
        mapped_norm_above_one=int((norm > 1).sum()),
        deviation_over_mapped_norm=summarize(ratio(delta_norm, norm), weights),
        deviation_over_constant_norm=summarize(ratio(delta_norm, constant_norm), weights),
        deviation_energy_over_mapped_energy=scalar_ratio((weights*delta_norm.square()).sum(),
                                                        (weights*norm.square()).sum()),
        weighted_deviation_mean_norm=float((weights[:, None]*delta).sum(0).norm()),
    )


def pair_metrics(first, second, weights):
    first, second = first.double(), second.double()
    weights = weights.double()/weights.sum()
    difference = (first-second).norm(dim=1)
    norm = first.norm(dim=1)
    return dict(cosine=direction_summary(first, second, weights),
                difference_over_first_norm=summarize(ratio(difference, norm), weights),
                difference_energy_over_first_energy=scalar_ratio((weights*difference.square()).sum(),
                                                                 (weights*norm.square()).sum()))


def spectrum(gram, *, feature_dim):
    """Eigenvalues are squared singular values of the weighted data matrix."""
    raw = torch.linalg.eigvalsh(gram.double()).flip(0)
    if float(raw[-1]) < -1e-10*max(1., float(raw[0])):
        raise ValueError('Gram matrix has materially negative eigenvalues')
    eigenvalues = raw.clamp_min(0)
    energy, largest = float(eigenvalues.sum()), float(eigenvalues[0])
    tolerance = largest*max(len(gram), feature_dim)*torch.finfo(torch.float64).eps
    result = dict(energy=energy, eigenvalues=eigenvalues.tolist(),
                  minimum_raw_eigenvalue=float(raw[-1]), numerical_rank_tolerance=tolerance,
                  numerical_gram_rank=int((eigenvalues > tolerance).sum()))
    if energy == 0:
        result.update(entropy_rank=0., participation_rank=0., stable_rank=0.)
        result.update({f'components_{level}pct': 0 for level in (90, 95, 99)})
        result.update({f'top_{k}_energy_fraction': 0. for k in (1, 5, 10)})
        return result
    probabilities = eigenvalues/energy
    positive = probabilities[probabilities > 0]
    result.update(entropy_rank=float(torch.exp(-(positive*positive.log()).sum())),
                  participation_rank=float(1/probabilities.square().sum()), stable_rank=energy/largest)
    cumulative = probabilities.cumsum(0)
    for level in (90, 95, 99):
        result[f'components_{level}pct'] = int(torch.searchsorted(cumulative, level/100))+1
    for k in (1, 5, 10):
        result[f'top_{k}_energy_fraction'] = float(probabilities[:k].sum())
    return result


def matrix_geometry(matrix, weights, shot_ids):
    matrix, weights = matrix.double(), weights.double()/weights.sum()
    rootw = weights.sqrt()[:, None]
    mean = (weights[:, None]*matrix).sum(0)
    per_row_mean = torch.empty_like(matrix)
    shot_means, shot_weights = [], []
    for sid in dict.fromkeys(shot_ids):
        selected = torch.tensor([s == sid for s in shot_ids])
        mass = weights[selected].sum()
        shot_mean = (weights[selected, None]*matrix[selected]).sum(0)/mass
        shot_means.append(shot_mean)
        shot_weights.append(mass)
        per_row_mean[selected] = shot_mean
    between = torch.stack(shot_weights).sqrt()[:, None]*(torch.stack(shot_means)-mean)
    matrices = dict(uncentered=rootw*matrix, centered=rootw*(matrix-mean),
                    between_shots=between, within_shots=rootw*(matrix-per_row_mean))
    result = {name: spectrum(value@value.T, feature_dim=matrix.shape[1]) for name, value in matrices.items()}
    total = result['centered']['energy']
    between_energy, within_energy = result['between_shots']['energy'], result['within_shots']['energy']
    result.update(between_fraction_of_centered_energy=scalar_ratio(between_energy, total),
                  within_fraction_of_centered_energy=scalar_ratio(within_energy, total),
                  centered_fraction_of_uncentered_energy=scalar_ratio(total, result['uncentered']['energy']),
                  decomposition_error=total-between_energy-within_energy)
    return result


def make_groups(records, weights):
    ids = list(dict.fromkeys(r['shot'] for r in records))
    masks = dict(all=torch.ones(len(records), dtype=torch.bool))
    masks.update({name: torch.tensor([bool(r['disruptive']) == flag for r in records])
                  for name, flag in (('disruptive', True), ('safe', False))})
    masks.update({f'shot_{sid}': torch.tensor([r['shot'] == sid for r in records]) for sid in ids})
    metadata = {}
    for name, mask in masks.items():
        selected = [r for r, keep in zip(records, mask.tolist()) if keep]
        shots = list(dict.fromkeys(r['shot'] for r in selected))
        metadata[name] = dict(windows=len(selected), shots=len(shots), shot_ids=shots,
                              training_weight_mass=float(weights[mask].sum()))
    return masks, metadata


def verify_saved_fit(signal, targets, weights, rules):
    """Check saved normal equations; never solve for or replace coefficients."""
    x = unit_rows(signal)
    mx = (weights[:, None]*x).sum(0)
    keys = weights.sqrt()[:, None]*(x-mx)
    gram = keys@keys.T
    checks = {}
    for arm in ARMS:
        rule = rules[arm]
        target = unit_rows(targets[arm])
        mean = (weights[:, None]*target).sum(0)
        rhs = weights.sqrt()[:, None]*(target-mean)
        error = (gram+rule['ridge']*torch.eye(len(x), dtype=torch.float64))@rule['coefficients']-rhs
        check = dict(keys_max_error=float((keys-rule['keys']).abs().max()),
                     center_max_error=float((mx-rule['center']).abs().max()),
                     mean_max_error=float((mean-rule['mean']).abs().max()),
                     relative_normal_equation_residual=float(error.norm()/rhs.norm()))
        if not all(value < 1e-10 for value in check.values()):
            raise ValueError(f'saved fit verification failed: {arm} {check}')
        checks[arm] = check
    return checks


DEFINITIONS = dict(
    scope='In-fit descriptions on the 688 saved cold20 windows; no refit, no new validation or arm selection.',
    weights='w_i=0.5/(shots in outcome class)/(windows in shot); renormalized within each reported group.',
    statistics='weighted_mean uses w; window_mean and quantiles count windows equally. Undefined zero-vector cosines/zero denominators are omitted and counted, not set to zero.',
    mapped='Saved predict_rule(signal, rule) on CPU with original float32 query/output; geometry is then computed in float64. No bounded_step is applied.',
    constant='mu_y=rule.mean, the weighted mean of unit target gradients. This is the plan constant direction, not the affine intercept mu_y-mu_x B.',
    deviation='d_i=mapped_i-mu_y. Ratios are ||d_i||/||mapped_i|| and ||d_i||/||mu_y||; these need not be <=1.',
    deviation_energy='sum(w_i ||d_i||^2)/sum(w_i ||mapped_i||^2); a second-moment ratio, not explained variance. Group cross-terms need not vanish.',
    cosine_gain='cos(mapped_i,target_i)-cos(mu_y,target_i); in-fit gain over the constant baseline.',
    half_roll='Exactly prepare.py: for each shot indices, order[indices]=roll(indices,len(indices)//2); no cross-shot or class reassignment.',
    pair_difference='||first_i-second_i||/||first_i||; pooled energy uses sum(w ||first-second||^2)/sum(w ||first||^2).',
    direction_counts='same_within_1e6 means cosine>=1-1e-6; cos_ge_0_99 and cos_ge_0_9 are descriptive thresholds, not hypothesis tests.',
    gram='G=(sqrt(w) X)(sqrt(w) X)^T. Input/target X are row-unit gradients; mapped X retains output norms. Full sorted eigenvalues are included.',
    centered='Subtract the weighted global mean. Between-shot matrix rows are sqrt(w_s)(mu_s-mu); within-shot rows are sqrt(w_i)(X_i-mu_s). Their energies sum to centered energy.',
    effective_rank='With p_j=lambda_j/sum(lambda): entropy=exp(-sum p log p), participation=1/sum p^2, stable=sum(lambda)/max(lambda).',
    numerical_rank='Count Gram eigenvalues > max(lambda)*max(number of rows,feature_dim)*float64_eps; this is a stated numerical cutoff, not independent sample count.',
    ridge_df='Sum lambda/(lambda+ridge) for the weighted centered input Gram; +1 adds the unpenalized intercept, per output coordinate.',
    shot_weight_kish='1/sum(w_s^2) describes weight concentration only; it does not estimate the number of independent experimental shots.',
)


@torch.no_grad()
def analyze(pairs, policy):
    records, rules = pairs['records'], policy['rules']
    weights = balanced_weights(records)
    order = half_roll_indices(records)
    targets = {arm: pairs[arm] for arm in ('operational', 'point')}
    targets['shuffled'] = pairs['operational'][order]
    masks, groups = make_groups(records, weights)
    report = dict(definitions=DEFINITIONS, groups=groups,
                  fit_checks=verify_saved_fit(pairs['signal'], targets, weights, rules))
    mapped = {arm: predict_rule(pairs['signal'], rules[arm]) for arm in ARMS}
    report['fit'] = {}
    for arm in ARMS:
        report['fit'][arm] = {}
        for name, mask in masks.items():
            row = mapping_metrics(mapped[arm][mask], targets[arm][mask], rules[arm]['mean'], weights[mask])
            row['operational_target_cosine'] = direction_summary(mapped[arm][mask], pairs['operational'][mask], weights[mask])
            report['fit'][arm][name] = row
    unit_targets = {arm: unit_rows(value) for arm, value in targets.items()}
    comparisons = dict(half_roll_target=(unit_targets['operational'], unit_targets['shuffled']),
                       operational_point_target=(unit_targets['operational'], unit_targets['point']),
                       operational_shuffled_mapped=(mapped['operational'], mapped['shuffled']),
                       operational_point_mapped=(mapped['operational'], mapped['point']))
    report['comparisons'] = {comparison: {name: pair_metrics(a[mask], b[mask], weights[mask])
                                         for name, mask in masks.items()}
                             for comparison, (a, b) in comparisons.items()}
    report['shuffle_invariants'] = dict(
        cross_shot_pairs=sum(records[i]['shot'] != records[j]['shot'] for i, j in enumerate(order.tolist())),
        weights_preserved_exact=bool(torch.equal(weights, weights[order])),
        constant_mean_max_difference=float((rules['operational']['mean']-rules['shuffled']['mean']).abs().max()),
        target_spectrum='Within-shot uniform-weight permutation preserves all four operational target spectra below.',
        permutation=order.tolist(),
    )
    geometry_inputs = dict(unit_signal=unit_rows(pairs['signal']),
                           unit_operational=unit_targets['operational'], unit_point=unit_targets['point'])
    geometry_inputs.update({f'mapped_{arm}': value for arm, value in mapped.items()})
    report['geometry'] = {}
    for name, matrix in geometry_inputs.items():
        print(f'CPU spectrum: {name}', flush=True)
        report['geometry'][name] = matrix_geometry(matrix, weights, [r['shot'] for r in records])
    eigenvalues = torch.tensor(report['geometry']['unit_signal']['centered']['eigenvalues'], dtype=torch.float64)
    ridge = rules['operational']['ridge']
    report['ridge_complexity'] = dict(ridge=ridge, slope_df_per_output=float((eigenvalues/(eigenvalues+ridge)).sum()),
                                      including_intercept_per_output=1+float((eigenvalues/(eigenvalues+ridge)).sum()),
                                      eigenvalues_above_ridge=int((eigenvalues > ridge).sum()))
    shot_masses = [v['training_weight_mass'] for k, v in groups.items() if k.startswith('shot_')]
    report['sample_structure'] = dict(shots=groups['all']['shots'], windows=len(records), features=pairs['signal'].shape[1],
        disruptive_shots=groups['disruptive']['shots'], safe_shots=groups['safe']['shots'],
        shot_weight_kish=1/sum(w*w for w in shot_masses),
        max_centered_window_rank=min(len(records)-1, pairs['signal'].shape[1]),
        max_between_shot_rank=groups['all']['shots']-1,
        zero_rows={name: int((value == 0).all(dim=1).sum()) for name, value in {**targets, 'signal': pairs['signal']}.items()})
    report['limitations'] = [
        'All mapping cosines are in-fit: the same 20 shots taught these rules; 688 temporally correlated windows are not 688 independent shots.',
        'The parent and input alignment already used all cold20; neither these results nor the existing component folds are independent system validation.',
        'Only 3 safe shots carry half the fit weight (1/6 each); 17 disruptive shots carry the other half (1/34 each). No reliability or uncertainty claim follows from this weighting.',
        'Half-roll preserves shot identity, outcome, the shot target distribution and the constant mean. Its strength is measured here, not assumed.',
        'Ranks describe observed linear geometry and ridge shrinkage, not physical information, mutual information, alarm improvement or population generalization.',
        'No post-cold20 outcomes or full360 results were accessed; no candidate is selected using these diagnostics.',
    ]
    return report


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def execute():
    torch.set_num_threads(2)
    inputs = [DIRECTORY/name for name in ('pairs.pt', 'policy.pt', 'protocol.json', 'complete.json')]
    inputs += [Path(__file__), Path(__file__).with_name('learning.py'), Path(__file__).with_name('prepare.py'),
               ROOT/'tests/test_operational_transport_fit_diagnostic.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protocol = json.loads((DIRECTORY/'protocol.json').read_text())
    complete = json.loads((DIRECTORY/'complete.json').read_text())
    pairs = torch.load(DIRECTORY/'pairs.pt', map_location='cpu', weights_only=True)
    policy = torch.load(DIRECTORY/'policy.pt', map_location='cpu', weights_only=True)
    if not complete['complete'] or complete['full_stream_run']:
        raise ValueError('completed preparation artifact required')
    if (policy['pairs_sha256'] != before[str((DIRECTORY/'pairs.pt').relative_to(ROOT))]
            or complete['policy_sha256'] != before[str((DIRECTORY/'policy.pt').relative_to(ROOT))]
            or policy['protocol'] != protocol):
        raise ValueError('artifact provenance mismatch')
    for name in ('learning.py', 'prepare.py'):
        relative = str(Path(__file__).with_name(name).relative_to(ROOT))
        if before[relative] != protocol['code_sha256'][relative]:
            raise ValueError('fitting implementation differs from v2 snapshot: '+name)
    records = pairs['records']
    if (len(records) != 688 or len(records) != complete['examples']
            or list(dict.fromkeys(r['shot'] for r in records)) != protocol['warmup_ids']
            or len(protocol['warmup_ids']) != 20):
        raise ValueError('expected exact cold20 fit records')
    for name in ('signal', 'operational', 'point'):
        value = pairs[name]
        if value.device.type != 'cpu' or value.shape != (688, 8192) or not torch.isfinite(value).all():
            raise ValueError('finite CPU 688x8192 pairs required')
    report = analyze(pairs, policy)
    if report['sample_structure']['disruptive_shots'] != 17 or report['sample_structure']['safe_shots'] != 3:
        raise ValueError('unexpected cold20 class composition')
    if before != {str(p.relative_to(ROOT)): sha(p) for p in inputs}:
        raise ValueError('input or diagnostic source changed during CPU analysis')
    report['provenance'] = dict(input_sha256=before, input_hashes_unchanged=True, device='cpu', threads=2,
                                torch_version=str(torch.__version__), python_version=platform.python_version(),
                                independent_confirmation=False, refit=False)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    output = DIRECTORY/'fit_geometry.json'
    if output.is_symlink():
        raise ValueError('diagnostic output must not be a symlink')
    output.write_text(text)
    print(json.dumps(dict(output=str(output), sample_structure=report['sample_structure'],
                          ridge_complexity=report['ridge_complexity'], input_hashes_unchanged=True), indent=2))


if __name__ == '__main__':
    execute()
