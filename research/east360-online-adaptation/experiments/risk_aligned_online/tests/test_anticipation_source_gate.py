from types import SimpleNamespace

import numpy as np

from experiments.risk_aligned_online.anticipation_source_gate import evaluate
from experiments.risk_aligned_online.simple_data import window
from experiments.risk_aligned_online.tests.test_anticipatory_readout import Toy
from experiments.risk_aligned_online.score_shift_audit import read_seals


def test_all_source_val_shots_are_paired_and_times_are_not_moved(tmp_path):
    calls = []
    times = np.arange(51)/1000.

    def read(device, sid):
        calls.append((device, sid))
        x = np.full((51, 23), 1. if sid == 1 else -1., np.float32)
        return dict(device=device, shot=sid, values=x, available=np.ones_like(x, bool), times=times)

    def label(s, device, advanced):
        assert device == 'J-TEXT'
        return dict(s, valid=np.ones(51, bool), is_disrupt=s['shot'] == 1,
                    down=.06 if s['shot'] == 1 else None)

    api = SimpleNamespace(source_roles=lambda: dict(val=[1, 2]), advanced_times=lambda: {},
                          read_shot=read, attach_outcome=label, window=window)
    result = evaluate(Toy(), output=tmp_path, threshold=.6, device='cpu', data_api=api)
    assert calls == [('J-TEXT', 1), ('J-TEXT', 2)]
    assert result['source_gate'] == dict(observed=True, hold=True, forecast=True)
    for arm in ('observed', 'hold', 'forecast'):
        assert result['metrics'][arm]['tp'] == 1
        assert result['metrics'][arm]['fp'] == 0
        seals = read_seals(tmp_path/arm)
        np.testing.assert_array_equal(seals[1]['times'], times)
        assert len(seals) == 2 and seals[1]['threshold'] == .6
    assert result['signal_error_mean']['direct_smooth_l1'] == 0.


def test_full_context_gate_never_suppresses_original_early_predictions(tmp_path):
    times = np.arange(51)/1000.
    def read(device, sid):
        x = np.full((51, 23), 1. if sid == 1 else -1., np.float32)
        return dict(device=device, shot=sid, values=x, available=np.ones_like(x, bool), times=times)
    def label(s, device, advanced):
        return dict(s, valid=np.ones(51, bool), is_disrupt=s['shot'] == 1,
                    down=.06 if s['shot'] == 1 else None)
    api = SimpleNamespace(source_roles=lambda: dict(val=[1, 2]), advanced_times=lambda: {},
                          read_shot=read, attach_outcome=label, window=window)
    evaluate(Toy(), output=tmp_path, threshold=.6, device='cpu', data_api=api, require_full_context=True)
    observed = read_seals(tmp_path/'observed')
    for arm in ('hold', 'forecast'):
        rows = read_seals(tmp_path/arm)
        for sid in (1, 2):
            np.testing.assert_array_equal(rows[sid]['scores'], observed[sid]['scores'])
            np.testing.assert_array_equal(rows[sid]['times'], times)
