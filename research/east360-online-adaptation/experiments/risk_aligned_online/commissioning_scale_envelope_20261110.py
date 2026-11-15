"""Post-seal reachability of a fixed event readout under scale changes only."""

import numpy as np


def cue_bounds(ratio, weights, adaptable, first_effect, factor=4.0):
    """Extrema of the weighted cue, allowing independent current and lag scales."""
    z = np.asarray(ratio, dtype=np.float64)
    active = np.asarray(adaptable, dtype=bool)
    c = z.shape[1]
    magnitude = np.abs(z)
    change = np.sinh(magnitude[:, active])
    lo, hi = magnitude.copy(), magnitude.copy()
    lo[first_effect:, active] = np.arcsinh(change[first_effect:] / factor)
    hi[first_effect:, active] = np.arcsinh(change[first_effect:] * factor)
    sign = np.sign(z)
    a = sign * weights[:c]
    b = np.zeros_like(z)
    b[32:] = sign[32:] * (sign[32:] == sign[:-32]) * weights[c:]
    old_lo, old_hi = np.zeros_like(z), np.zeros_like(z)
    old_lo[32:], old_hi[32:] = lo[:-32], hi[:-32]

    def extremum(direction):
        old = np.where(direction * b >= 0, old_hi, old_lo)

        def objective(m):
            return direction * (a * m + b * np.minimum(m, old))

        # For a fixed current magnitude the lag objective is monotone. Once
        # its optimal end is chosen, the only possible kink is at that end.
        best = np.maximum.reduce([objective(m) for m in (lo, hi, np.clip(old, lo, hi))])
        return direction * best.sum(axis=1)

    return extremum(-1), extremum(1)


def run():
    from pathlib import Path

    from . import io23, simple_data
    from .commissioning_event_roughness_20261110 import COLUMNS, CUE, FIRST_EFFECT, roughness
    from .commissioning_intra_20261110 import STARTUP
    from .commissioning_shared_variation_20261110 import PAIR, REFERENCE
    from .history_gradient_metric_20261102.run import read, sha, write
    from .mask_repair_20261104.east_evaluation import load_inputs, prepare_actual
    from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent

    path = io23.ROOT / 'probe_outputs/commissioning_shared_variation360_20261110_v1'
    proto, seal = (read(path / k) for k in ('protocol.json', 'seal.json'))
    inputs = load_inputs()
    rows = read(path / 'per_shot.json')['0.2']['static']
    if [r['shot'] for r in rows] != proto['order'] or proto['order'] != inputs['ids']:
        raise ValueError('sealed ordered cohort required')
    fitted = REFERENCE / 'parameters.npz'
    if sha(fitted) != proto['model_and_threshold_sha256']['parameters.npz']:
        raise ValueError('reference parameters changed')
    with np.load(fitted, allow_pickle=False) as f:
        log_ref = f['prior']
        supported = f['supported']
        weights = f['event_axis'].astype(np.float64) * float(f['event_risk_head'][-2])
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (a.detach().numpy()[COLUMNS] for a in (base.center, base.scale))
    mask = np.isin(np.arange(len(CUE)), PAIR)
    changed = dict(profile_pair=mask & supported, all_supported=supported)
    deficits = {k: [] for k in changed}
    feasible = {k: {c: [] for c in ('late', 'miss', 'false_alarm')} for k in changed}
    totals = dict(late=0, miss=0, false_alarm=0)
    threshold = proto['suffix_thresholds']['0.2']
    boundary = np.log(threshold / (1. - threshold))
    # Float32 readout and rounded inverse sigmoid; expand the interval rather
    # than exclude a shot on a marginal numerical discrepancy.
    tolerance = 4e-5
    for rank, row in enumerate(rows, 1):
        if rank <= STARTUP or row['timely'] or (not row['disruptive'] and not row['alarm']):
            continue
        sid = row['shot']
        d, down = io23.read_outcome('EAST', sid)
        if (d, down) != (row['disruptive'], row['down_s']):
            raise ValueError(f'sealed outcome changed: {sid}')
        kind = ('late' if row['alarm'] else 'miss') if d else 'false_alarm'
        totals[kind] += 1
        curve = path / 'shots' / f'{rank:04d}_{sid}.npz'
        if (sha(curve) != seal['curve_sha256'][curve.name]
                or sha(io23.path_for('EAST', sid)) != inputs['hashes'][str(sid)]):
            raise ValueError(f'static curve or raw signal changed: {sid}')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        signal = prepare_actual(raw, rank, inputs['mapping'])
        z, m = roughness.coordinates(signal['values'][:, COLUMNS], signal['available'][:, COLUMNS], center, scale)
        delta = roughness._contrast(z, m)
        delta[:, ~supported] = 0.
        ratio = np.arcsinh(delta / np.exp(np.where(supported, log_ref, 0.)))
        fixed, _ = cue_bounds(ratio, weights, np.zeros_like(supported), FIRST_EFFECT)
        with np.load(curve, allow_pickle=False) as f:
            times, p = f['times'], f['static'].astype(np.float64)
        np.testing.assert_array_equal(times, raw['times'])
        # Use the full float32 rounding interval, including saturated 0/1.
        # These probability bounds are conservative before inversion.
        lower_p = np.maximum(0., p - 1e-7)
        upper_p = np.minimum(1., p + 1e-7)
        with np.errstate(divide='ignore'):
            lower_logit = np.log(lower_p) - np.log1p(-lower_p)
            upper_logit = np.log(upper_p) - np.log1p(-upper_p)
        eligible = np.round((down-times)*1000., 9) > 30. if d else np.ones(len(times), bool)
        for name, flags in changed.items():
            lo, hi = cue_bounds(ratio, weights, flags, FIRST_EFFECT)
            if d:
                deficit = boundary - (upper_logit + hi - fixed)[eligible].max(initial=-np.inf)
                if np.isfinite(deficit):
                    deficits[name].append(float(deficit))
                possible = deficit <= tolerance
            else:
                possible = (lower_logit + lo - fixed).max() <= boundary + tolerance
            if possible:
                feasible[name][kind].append(sid)
    out = dict(scope='POSTHOC_RELAXATION_NOT_AN_ONLINE_POLICY', first_effect=FIRST_EFFECT,
               scale_factor=4, threshold=threshold, float_logit_tolerance=tolerance,
               errors=totals, feasible=feasible,
               adaptable={k: [n for n, use in zip(CUE, flags, strict=True) if use]
                          for k, flags in changed.items()},
               remaining_disruptive_gap_median_logit={
                   k: float(np.median([v for v in g if v > tolerance])) for k, g in deficits.items()},
               source_sha256=proto['source_sha256'], parameters_sha256=sha(fitted),
               static_seal_sha256=sha(path / 'seal.json'), producer_sha256=sha(Path(__file__)))
    write(path / 'scale_envelope.json', out)
    print(out, flush=True)


if __name__ == '__main__':
    run()
