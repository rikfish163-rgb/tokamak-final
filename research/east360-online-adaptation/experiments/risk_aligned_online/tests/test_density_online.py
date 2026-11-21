import numpy as np
import pytest

from experiments.risk_aligned_online import density_online
from .test_ended_density_replay import provider, finish, DENSITY


def test_postshot_supervision_reencodes_old_experience_without_changing_seals(tmp_path, monkeypatch):
    p=provider(tmp_path/'a')
    first, current=finish(p,1), finish(p,2)
    first['_sealed_scores']=np.full(60,.1)
    current['_sealed_scores']=np.full(60,.2)
    before=first['values'].copy()
    def update(model, shot, replay, **kwargs):
        assert shot['shot']==2 and [s['shot'] for s in replay]==[1]
        assert not np.array_equal(replay[0]['values'][:,DENSITY], before[:,DENSITY])
        np.testing.assert_array_equal(replay[0]['_sealed_scores'],first['_sealed_scores'])
        assert kwargs['data_api'] is p
        return dict(accepted=True)
    monkeypatch.setattr(density_online,'postshot_update',update)
    # Match run_arm's real callback, including its data_api keyword. This
    # catches duplicate-keyword forwarding before a long GPU replay.
    result=density_online.update_in_coordinates(None,current,[first],coordinates=p,data_api=p)
    assert result['accepted']
    np.testing.assert_array_equal(first['values'],before)


def test_postshot_adapter_rejects_future_or_unsealed_experience(tmp_path):
    p=provider(tmp_path/'a'); current=finish(p,1)
    with pytest.raises(ValueError,match='sealed'):
        density_online.update_in_coordinates(None,current,[],coordinates=p,data_api=p)
    current['_sealed_scores']=np.full(60,.1)
    future=dict(current,shot=3)
    with pytest.raises(ValueError,match='history'):
        density_online.update_in_coordinates(None,current,[future],coordinates=p,data_api=p)


def test_cli_accepts_explicit_commissioning_order_and_roster():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-m',
        'experiments.risk_aligned_online.density_online', '--help'],
        capture_output=True, text=True, check=True)
    assert '--order' in result.stdout and '--roster' in result.stdout
