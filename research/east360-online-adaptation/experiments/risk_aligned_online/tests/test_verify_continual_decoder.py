import numpy as np
import pytest

from experiments.risk_aligned_online.verify_continual_decoder import changes, compare_curves


def row(sid, disruptive, timely=False, alarm=False):
    return dict(shot=sid, is_disrupt=disruptive, timely=timely, alarm=alarm)


def test_paired_gain_and_harm_are_separate():
    a = [row(1,True), row(2,True,True), row(3,False,alarm=True), row(4,False)]
    b = [row(1,True,True), row(2,True), row(3,False), row(4,False,alarm=True)]
    assert changes(a,b) == dict(rescued=[1],lost=[2],removed_fp=[3],new_fp=[4])
    with pytest.raises(AssertionError):
        changes(a,b[::-1])


def test_phase_comparison_keeps_original_stream_indices(tmp_path):
    left, right = tmp_path/'left', tmp_path/'right'
    left.mkdir(); right.mkdir()
    order = [{'shot':101},{'shot':102}]
    for index, item in enumerate(order,1):
        name = f'{index:04d}_{item["shot"]}_sealed.npz'
        np.savez(left/name,times=[0.,1.],scores=[.1,.2])
        np.savez(right/name,times=[0.,1.],scores=[.1,.3] if index==1 else [.1,.2])
    assert compare_curves(left,right,order,start=1) == 0.
    assert compare_curves(left,right,order) > 0.
    np.savez(right/'0002_102_sealed.npz',times=[0.,2.],scores=[.1,.2])
    with pytest.raises(AssertionError,match='time'):
        compare_curves(left,right,order,start=1)
