"""EFD-inspired per-device mean/std control in a frozen source coordinate system.

Shen's thesis section 4.5.1 uses feature-wise device means/stds. This candidate
estimates target moments from ENDED signals, not the complete future stream.
It does not reproduce the paper's zero-training-shot/ten-target-disruption
statistics protocol. Numeric distribution matching is not physical calibration.
The qualified source model's median/IQR/asinh buffers are left unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import io23, simple_data
from .ended_density_replay import EndedDensityReplay
from .simple_online import _durable_json, _hash


class ChannelMoments:
    """Masked Chan/Welford merge; population std (ddof=0), all valid points."""

    def __init__(self):
        self.counts = np.zeros(23, np.int64)
        self.mean = np.zeros(23, np.float64)
        self.m2 = np.zeros(23, np.float64)

    def update(self, values, available):
        x, a = np.asarray(values, np.float64), np.asarray(available, bool)
        if x.ndim != 2 or x.shape[1] != 23 or a.shape != x.shape:
            raise ValueError('matching flat23 signals and observation mask required')
        for c in range(23):
            v = x[:, c][a[:, c] & np.isfinite(x[:, c])]
            n = len(v)
            if not n:
                continue
            old, total = self.counts[c], self.counts[c] + n
            delta = v.mean() - self.mean[c]
            self.m2[c] += np.square(v-v.mean()).sum() + delta**2 * old*n/total
            self.mean[c] += delta*n/total
            self.counts[c] = total

    def snapshot(self):
        variance = np.divide(self.m2, self.counts, out=np.zeros(23), where=self.counts > 0)
        return dict(counts=self.counts.copy(), mean=self.mean.copy(),
                    std=np.sqrt(np.maximum(variance, 0.)))


def _validate_statistics(statistics):
    for key in ('mean', 'std', 'counts'):
        v = np.asarray(statistics[key])
        if v.shape != (23,) or not np.isfinite(v).all():
            raise ValueError('finite flat23 device moments required')
    if (np.asarray(statistics['std']) < 0).any() or (np.asarray(statistics['counts']) < 0).any():
        raise ValueError('nonnegative device variance and counts required')


def _channel_selection(channels):
    if channels is None:
        return None
    if (not isinstance(channels, (tuple, list)) or not channels
            or any(not isinstance(c, str) or c not in io23.RAW23 for c in channels)
            or len(channels) != len(set(channels))):
        raise ValueError('explicit unique known channels required')
    return tuple(channels)


def source_coordinate_map(values, available, source, target, *, preserve_controls=False, channels=None):
    """mu_J + sigma_J*(x_E-mu_E)/sigma_E, THEN frozen parent preprocessing.

    The legacy all-channel experiment excluded the mode column by declaration;
    it was NOT a certified integer category (Shen's average mode is fractional).
    Explicit channel selection permits a separate numerical intervention there.
    Channels with missing data or unidentifiable constant variance retain the parent's
    original coordinate handling; their observation masks are NOT erased.
    Control preservation is an explicit ablation, not a physical unit claim.
    """
    _validate_statistics(source); _validate_statistics(target)
    channels = _channel_selection(channels)
    x, a = np.asarray(values, np.float64), np.asarray(available, bool)
    if x.ndim != 2 or x.shape[1] != 23 or a.shape != x.shape:
        raise ValueError('matching flat23 signals and observation mask required')
    mapped, statuses, aligned = x.copy(), [], np.zeros(23, bool)
    for c, name in enumerate(io23.RAW23):
        if channels is not None and name not in channels:
            status = 'source_coordinate_no_intervention'
        elif channels is None and name == 'mode_number_n':
            # Preserve archived values and legacy receipt text, not its old
            # categorical interpretation. New targeted experiments are explicit.
            status = 'discrete_source_normalization'
        elif preserve_controls and name in ('Bt', 'ip'):
            status = 'physical_level_preserved_source_normalization'
        elif target['counts'][c] == 0:
            status = 'unavailable_target'
        elif source['counts'][c] == 0:
            status = 'unavailable_source'
        elif target['std'][c] == 0:
            status = 'unidentifiable_target_variance'
        elif source['std'][c] == 0:
            status = 'unidentifiable_source_variance'
        else:
            status = 'device_zscore_in_source_coordinates'
            valid = a[:, c] & np.isfinite(x[:, c])
            # Equal moments must preserve the qualified source input exactly.
            if (source['mean'][c] != target['mean'][c]
                    or source['std'][c] != target['std'][c]):
                mapped[valid, c] = (source['mean'][c] + source['std'][c] *
                    (x[valid, c]-target['mean'][c])/target['std'][c])
            aligned[c] = True
        statuses.append(status)
    mapped = mapped.astype(np.float32)
    if not np.isfinite(mapped[a]).all():
        raise FloatingPointError('nonfinite device-normalized observed input')
    return mapped, dict(aligned=aligned, status=statuses)


class EfdReplay(EndedDensityReplay):
    """Same seal/reveal gate; fixed-prefix or cumulative ENDED target moments."""

    def __init__(self, source_center, source_scale, *, source_statistics,
                 statistics_mode='fixed', preserve_controls=False, channels=None, **kwargs):
        _validate_statistics(source_statistics)
        if statistics_mode not in ('fixed', 'cumulative') or type(preserve_controls) is not bool:
            raise ValueError('explicit moment update policy required')
        super().__init__(source_center, source_scale, preserve_observed_channels=True, **kwargs)
        self.source_statistics = {k:np.asarray(source_statistics[k]).copy()
                                  for k in ('counts', 'mean', 'std')}
        self.statistics_mode, self.preserve_controls = statistics_mode, preserve_controls
        self.channels = _channel_selection(channels)
        self.moments = ChannelMoments()

    def _consume_ended_signals(self, pending, ended):
        if self.statistics_mode == 'fixed' and len(ended) > self.calibration_count:
            return
        self.moments.update(pending['values'], pending['available'])
        if len(ended) >= self.calibration_count:
            self.statistics = dict(self.moments.snapshot(), calibration_ids=list(ended),
                statistics_mode=self.statistics_mode, definition='population_mean_std_full_valid_points',
                preserve_controls=self.preserve_controls, future_signals_used=False,
                revealed_class_used=False)

    def reencode(self, record):
        values = np.asarray(record.get('_raw_values', record['values'])).copy()
        available = np.asarray(record.get('_raw_available', record['available'])).copy()
        result = dict(record, values=values, available=available,
            _raw_values=values.copy(), _raw_available=available.copy(),
            calibration_ready=self.statistics is not None,
            alignment_active=self.aligned and self.statistics is not None)
        if result['alignment_active']:
            result['values'], receipt = source_coordinate_map(values, available,
                self.source_statistics, self.statistics, preserve_controls=self.preserve_controls,
                channels=self.channels)
            result['normalization_receipt'] = receipt
        if self.qa_coefficient is not None:
            from .qa_convention import rebuild_qa
            result = rebuild_qa(result, self.qa_coefficient)
        return result


def load_source_statistics(path, *, expected_parent_sha256, train_ids):
    result = json.loads(Path(path).read_text())
    p = result['protocol']
    if (p['parent_sha256'] != expected_parent_sha256 or p['channel_order'] != list(io23.RAW23)
            or p['source_role'] != 'J-TEXT TRAIN' or p['target_records_accessed'] != 0
            or set(p['train_ids']) != set(train_ids) or len(p['train_ids']) != len(train_ids)
            or p['definition'] != 'population_mean_std_full_valid_points'):
        raise ValueError('same-parent source TRAIN-only mean/std artifact required')
    statistics = {k:np.asarray(result[k]) for k in ('counts', 'mean', 'std')}
    _validate_statistics(statistics)
    return statistics, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_ids = simple_data.source_roles()['train']
    advanced = simple_data.advanced_times()
    state = ChannelMoments()
    for index, sid in enumerate(train_ids, 1):
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        # Source input contract only; no new target-specific MNM/CIII mask.
        state.update(shot['values'], shot['available'] & shot['valid'][:, None])
        if index % 200 == 0:
            print(json.dumps(dict(source_shots=index)), flush=True)
    protocol = dict(parent_sha256=_hash(args.checkpoint), channel_order=list(io23.RAW23),
        source_role='J-TEXT TRAIN', train_ids=train_ids, target_records_accessed=0,
        definition='population_mean_std_full_valid_points', ddof=0,
        source_time_scope='source training valid points (through disruption, no post-event points)',
        code_sha256=_hash(Path(__file__)), classifier_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _durable_json(args.output, dict(protocol=protocol,
        **{k:v.tolist() for k,v in state.snapshot().items()}))


if __name__ == '__main__':
    main()
