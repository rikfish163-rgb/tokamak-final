"""Causal density-coordinate control, NOT neural or physical calibration.

First N shots use the qualified source coordinates. Only after their whole
predictions have been durably sealed are their signals used for a one-time
numeric estimate. Subsequent EAST density values are mapped into the original
source coordinates; the model buffers and J-TEXT replay are never redefined.
"""
import json
from pathlib import Path

import numpy as np

from . import io23, simple_data
from .alignment import fit_ended_shot_statistics
from .causal_input import apply_input_contract
from .simple_online import _hash


class EndedDensityReplay:
    window = staticmethod(simple_data.window)

    def __init__(self, source_center, source_scale, *, output_dir, order,
                 calibration_count=30, aligned=True, data_api=simple_data,
                 qa_coefficient=None, preserve_observed_channels=False):
        self.source_center = np.asarray(source_center, np.float32).copy()
        self.source_scale = np.asarray(source_scale, np.float32).copy()
        if (self.source_center.shape != (23,) or self.source_scale.shape != (23,)
                or not np.isfinite(self.source_center).all()
                or not np.isfinite(self.source_scale).all()
                or (self.source_scale <= 0).any()):
            raise ValueError('finite flat23 source coordinates required')
        self.order = list(order)
        if (not self.order or any(type(s) is not int for s in self.order)
                or len(self.order) != len(set(self.order))
                or type(calibration_count) is not int
                or not 1 <= calibration_count < len(self.order)
                or type(aligned) is not bool or type(preserve_observed_channels) is not bool):
            raise ValueError('locked unique order and positive calibration prefix required')
        self.output_dir, self.data_api = Path(output_dir), data_api
        self.calibration_count, self.aligned = calibration_count, aligned
        # Explicit ablation: the archived route masked EVERY column lacking a
        # prefix IQR, even though only density is remapped. A constant observed
        # mode number has no estimable target scale but is not missing data.
        # Preserve that old route by default; this flag changes only masking.
        self.preserve_observed_channels = preserve_observed_channels
        if qa_coefficient is not None and (
                not np.isfinite(qa_coefficient) or qa_coefficient <= 0):
            raise ValueError('positive finite source QA convention required')
        self.qa_coefficient = qa_coefficient
        self.ended_ids, self.calibration_records = [], []
        self.pending, self.statistics = None, None

    def reencode(self, record):
        """Rebuild past replay from canonical raw data, never align it twice."""
        values = np.asarray(record.get('_raw_values', record['values'])).copy()
        available = np.asarray(record.get('_raw_available', record['available'])).copy()
        result = dict(record, values=values, available=available,
                      _raw_values=values.copy(), _raw_available=available.copy(),
                      calibration_ready=self.statistics is not None,
                      alignment_active=self.aligned and self.statistics is not None)
        if self.statistics is not None:
            if not self.preserve_observed_channels:
                result['available'] &= self.statistics['usable'][None, :]
            if self.aligned:
                c = io23.RAW23.index('ne_nG')
                center = float(self.statistics['center'][c])
                scale = max(float(self.statistics['scale'][c]), abs(center)*.01, 1e-6)
                # Keep the source encoder/forecast loss/source replay buffers
                # unchanged. This is a numerical mapping, not a unit claim.
                values[:, c] = ((values[:, c].astype(np.float64)-center)/scale
                                * float(self.source_scale[c])
                                + float(self.source_center[c])).astype(np.float32)
                if not np.isfinite(values[result['available']]).all():
                    raise FloatingPointError('nonfinite aligned input')
        if self.qa_coefficient is not None:
            from .qa_convention import rebuild_qa
            result = rebuild_qa(result, self.qa_coefficient)
        return result

    def read_shot(self, device, sid):
        if device != 'EAST':
            raise ValueError('online input supports EAST only')
        if self.pending is not None:
            raise ValueError('previous pending shot has not ended')
        index = len(self.ended_ids)
        if index == len(self.order) or sid != self.order[index]:
            raise ValueError('signals must follow the locked order')
        raw = apply_input_contract(self.data_api.read_shot(device, sid, rate_contract='nominal'))
        if (raw['shot'] != sid or raw['device'] != device
                or any(k in raw for k in ('y','down','is_disrupt'))):
            raise ValueError('label-free signal identity required')
        self.pending = dict(raw, values=raw['values'].copy(), available=raw['available'].copy())
        return self.reencode(raw)

    def attach_outcome(self, shot, device, advanced):
        if (self.pending is None or self.pending['shot'] != shot['shot']
                or device != 'EAST' or advanced):
            raise ValueError('only current pending EAST outcome may be revealed')
        paths = list(self.output_dir.glob(f'*_{shot["shot"]}_seal.json'))
        if len(paths) != 1:
            raise ValueError('one durable whole-shot seal required before reveal')
        receipt = json.loads(paths[0].read_text())
        npz = self.output_dir/receipt['npz']
        if receipt['shot'] != shot['shot'] or _hash(npz) != receipt['sha256']:
            raise ValueError('seal identity or hash mismatch before reveal')
        with np.load(npz, allow_pickle=False) as trace:
            if (int(trace['shot']) != shot['shot']
                    or not np.array_equal(trace['times'], self.pending['times'])
                    or trace['scores'].shape != self.pending['times'].shape):
                raise ValueError('seal must cover the whole pending shot')
        result = self.data_api.attach_outcome(shot, device, advanced)
        ended = [*self.ended_ids, shot['shot']]
        self._consume_ended_signals(self.pending, ended)
        self.ended_ids, self.pending = ended, None
        return result

    def _consume_ended_signals(self, pending, ended):
        """Called only after the durable whole-shot seal and natural reveal."""
        if len(ended) <= self.calibration_count:
            self.calibration_records.append(pending)
        if len(ended) == self.calibration_count:
            statistics = fit_ended_shot_statistics(self.calibration_records, ended_ids=ended)
            if not statistics['usable'][io23.RAW23.index('ne_nG')]:
                raise ValueError('ended prefix has no identifiable density scale')
            self.statistics = statistics
            self.calibration_records = []
