import numpy as np

from experiments.risk_aligned_online import commissioning_lag_clock_20261111 as clock


def test_interleave_uses_the_most_recent_available_action_without_backdating():
    original = np.array([153,185,217])
    offset = np.array([169,201,233])
    endpoints = np.array([152,153,168,169,184,185,200,201,232,233])
    use = clock.offset_active(original, offset, endpoints)
    assert use.tolist() == [False,False,False,True,True,False,False,True,False,True]
    np.testing.assert_array_equal(use[:7],clock.offset_active(original[:2],offset[:1],endpoints[:7]))


def test_no_supported_offset_feedback_means_original_exactly():
    ends = np.arange(211)
    assert not clock.offset_active(np.array([153,185]),np.empty(0,dtype=int),ends).any()


def test_first_offset_activation_is_after_complete_25_point_target():
    q = clock.offset_origins(230)
    assert q.tolist() == [143,175]
    assert (q+26).tolist() == [169,201]
    assert not len(clock.offset_origins(169))


def test_archive_stores_path_curves_separately_from_the_frozen_parent(tmp_path):
    parent, guard = (tmp_path/k for k in ('teacher','guard'))
    for root in (parent,guard):
        (root/'shots').mkdir(parents=True)
    name = '0021_123.npz'
    a,b = (root/'shots'/name for root in (parent,guard))
    t = np.arange(4)
    np.savez(a,times=t,**{clock.STATIC:t+1,clock.INHERITED:t+2})
    np.savez(b,times=t,release=[153],**{clock.OLD:t+3,clock.OLD_SHAM:t+4})
    time, curves, releases = clock.read_archived_curves(
        guard,parent,name,{name:clock.sha(b)},{name:clock.sha(a)},prepared=False)
    np.testing.assert_array_equal(time,t)
    np.testing.assert_array_equal(releases,[153])
    assert [curves[k][0] for k in clock.ARMS[:4]] == [1,2,3,4]
