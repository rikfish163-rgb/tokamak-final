from types import SimpleNamespace

import numpy as np

import run_joint_paired_direction_20261106 as exp


def test_hdf_timing_excludes_observed_exact_boundary_and_late_tail():
    t = np.arange(90)/1000.
    frames,mask,early,strict = exp.future_indices(t,63,.087,14.)
    assert frames[0] == 72 and frames[-1] == 89
    np.testing.assert_array_equal(frames[mask], [73,74,75,76])
    np.testing.assert_array_equal(frames[early], [72])
    np.testing.assert_array_equal(frames[strict], [72,73,74,75,76])


def test_two_directions_share_norm_grid_at_each_target():
    u = np.array([[[3.,0.],[0.,2.]], [[0.,2.],[-1.,0.]]])
    v = exp.matched_grid(u)
    expect = np.array([2.,1.])[:,None]*exp.SCALES
    np.testing.assert_allclose(np.linalg.norm(v[:,0],axis=-1),expect)
    np.testing.assert_allclose(np.linalg.norm(v[:,1],axis=-1),expect)


def test_paired_arm_adds_symmetric_history_and_all_safe_frames():
    gd = np.arange(12,dtype=float).reshape(6,2)
    gs = gd + 20
    zd = np.array([-.3,-.4,-.2,-.25,-.05,-.2])
    zs = np.array([-.3,-.4,-.2,-.25,-.08,-.25])
    a,b = exp.inequalities(gd,gs,zd,zs,np.array([True,False]),
        np.array([True,False]),0.,paired=False)
    c,d = exp.inequalities(gd,gs,zd,zs,np.array([True,False]),
        np.array([True,False]),0.,paired=True)
    np.testing.assert_array_equal(c[:len(a)],a)
    np.testing.assert_array_equal(d[:len(b)],b)
    assert len(a)==10 and len(c)==20
    np.testing.assert_array_equal(c[len(a):len(a)+4],gs[:4])
    np.testing.assert_array_equal(c[-2:],gs[4:])


def test_exact_replay_rejects_unsafe_other_sign_without_calling_it_a_missed_alarm():
    t = np.arange(100)/1000.
    d = np.full(len(t),exp.b.TAU-.20,dtype=np.float32)
    s = np.full(len(t),exp.b.TAU-.40,dtype=np.float32)
    q = SimpleNamespace(times=t,native=d,down=.090,horizon=25.)
    n = SimpleNamespace(times=t,native=s)
    frame = np.r_[exp.history(63),np.arange(72,96)]
    y = np.broadcast_to(d[frame],(1,2,1,len(frame))).copy()
    z = np.broadcast_to(s[frame],y.shape).copy()
    y[...,4] = exp.b.TAU+.025
    dm,sm = y[...,:4].copy(),z[...,:4].copy()
    sm[0,0,0,-1] += .15
    scored = exp.score_grid(q,n,frame,frame,np.zeros((1,2,1,2)),y,z,dm,sm)
    assert scored["danger_only"]["danger"]==1 and scored["danger_only"]["safe_reject"]==1
    assert scored["paired"]["neural_only"]==1 and scored["paired"]["scalar_any"]==0
