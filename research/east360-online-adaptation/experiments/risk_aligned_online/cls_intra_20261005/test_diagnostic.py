import numpy as np

from ..ffn_intra_20261005.test_meta import case
from .model import ClsModel
from .diagnostic import diagnostic_group, audit_shot
from .stream import predict_shot


def test_group_strict_timely_and_valid_boundaries():
    shot = dict(is_disrupt=True, down=.2, times=np.array([.074, .075, .169, .170, .190]),
                valid=np.ones(5, dtype=bool))
    assert [diagnostic_group(shot, i) for i in range(5)] == ['D_lead_gt125', 'D_lead_30_125', 'D_lead_30_125', None, None]
    shot['is_disrupt'] = False
    assert diagnostic_group(shot, 4) == 'S'
    shot['valid'][4] = False
    assert diagnostic_group(shot, 4) is None


def test_audit_follows_actual_stream_without_changing_its_predictions():
    old, shot, _ = case(); model = ClsModel(old.base)
    shot = {k: v[:33].copy() if isinstance(v, np.ndarray) else v for k, v in shot.items()}
    scores, _ = predict_shot(model, shot, use_task=True, offline_source=True)
    rows = audit_shot(model, shot, use_task=True, threshold=.5)
    assert [r['query_index'] for r in rows] == [6, 26, 32]
    for row in rows:
        assert abs(row['probability_after'] - scores[row['query_index']]) <= 2e-6
        assert row['target_end'] < row['query_index']
