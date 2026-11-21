import json
from types import SimpleNamespace

import numpy as np

import run_prealarm_teaching_support_20261106 as study


def test_radiation_gain_preserves_observation_mask_ratios_and_clock():
    x = np.arange(1, 1 + 23*8, dtype=np.float32).reshape(8,23)
    a = np.ones_like(x, bool)
    a[0,study.SXR_MEAN] = False
    x[~a] = 0
    r = dict(values=x, available=a, times=np.arange(8)/1000)
    p = study.perturb(r, (2., 3.))
    np.testing.assert_array_equal(p["times"], r["times"])
    assert p["available"].sum() == a.sum() - len(a)
    for j,k in ((study.SXR_MEAN,2), (study.SXR_VAR,4), (study.XUV_VAR,9)):
        np.testing.assert_array_equal(p["values"][:,j], k*x[:,j])
    untouched = sorted(set(range(23)) - {study.SXR_MEAN,study.SXR_VAR,study.XUV_VAR,study.MNM})
    np.testing.assert_array_equal(p["values"][:,untouched], x[:,untouched])
    assert p["values"][0,study.SXR_MEAN] == 0
    assert a[:,study.MNM].all()


def test_donor_selection_uses_one_observed_pair_and_label_blind_order():
    rec = {k: (float(k),float(21-k)) for k in range(1,21)}
    picks = study.donors(rec)
    assert picks["low"]["shot"] != picks["high"]["shot"]
    for row in picks.values():
        np.testing.assert_allclose(np.square(row["gain"])*10.5, rec[row["shot"]])


def test_local_first_crossing_excludes_reached_and_post_disruption_frames():
    c = np.array([63])
    z = np.full((1, 2, 32), -1., np.float64)
    z[0,0,0] = z[0,0,29] = 1.  # reached sample, then a post-terminal sample
    z[0,1,8] = 1.
    t = np.arange(100)/1000.
    f = study.first_in_suffix(c,t,z,down=.09,tau=0.)
    np.testing.assert_array_equal(f, [[-1,72]])


def test_teaching_requires_true_timely_canonical_and_absent_scalar_alternative():
    c = np.array([63])
    t = np.arange(100)/1000.
    a = np.ones((1,3), bool)
    real = np.full((1,3,32), -1.)
    scalar = real.copy()
    real[0,1,8] = 1.  # lead 18ms, canonical
    p = study.teacher(c,t,real,scalar,a,down=.09,horizon=25.,tau=0.)
    assert p["canonical"] == 1 and p["strict"] == (0,1,72)
    scalar[0,2,9] = 1.  # another action can rescue by scalar alone
    q = study.teacher(c,t,real,scalar,a,down=.09,horizon=25.,tau=0.)
    assert q["canonical"] == 1 and q["strict"] is None
    p = study.teacher(c,t,real,real.copy()*0-1,a,down=.082,horizon=25.,tau=0.)
    assert p["canonical"] == 0  # precisely 10 ms is not timely


def test_gate_exports_json_with_numpy_fold_labels():
    bank = {"low":{"counts":{"protected":[10]}}}
    shots = [SimpleNamespace(sid=10,teacher={"baseline":{"timely":True}})]
    out = study.gate({},bank,shots,{10:np.int64(2)})
    assert json.loads(json.dumps(out))["stress_added_per_fold"] == [0,0,1,0,0]
