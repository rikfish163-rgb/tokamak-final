import numpy as np
import torch

from recovery_state_20261106 import recovery_field, matched_bias


def trace(block_values):
    return np.repeat(np.asarray(block_values, dtype=float), 4, axis=0)


def test_repeated_recovery_produces_a_delayed_direction():
    # Third training window has recovered before the recurring excursion.
    x = trace([[0.], [1.], [0.], [1.], [0.], [1.], [1.], [0.]])
    a = np.ones_like(x, dtype=bool)
    real, wrong, receipt = recovery_field(x,a,np.array([True]),first=11,stride=4)
    assert not real[:12].any() and not wrong[:12].any()
    assert real[23,0] < 0.
    assert receipt["recovery_triples"] > 0
    assert np.all(real[:,0] <= 1.) and np.all(real[:,0] >= -1.)


def test_persistent_drift_does_not_teach_restoration():
    x = trace([[float(i)] for i in range(9)])
    a = np.ones_like(x, dtype=bool)
    real, wrong, receipt = recovery_field(x,a,np.array([True]),first=11,stride=4)
    assert receipt["supported_triples"] > 0
    assert receipt["recovery_triples"] == 0
    assert not real.any() and not wrong.any()


def test_prefix_and_shot_reset():
    rng = np.random.default_rng(443)
    x = rng.normal(size=(80,4)).cumsum(0)
    a = rng.random(x.shape)>.15
    write = np.array([False,True,True,True])
    p = recovery_field(x,a,write,first=15,stride=4)
    for cutoff in (15,16,17,39,65):
        q = recovery_field(x[:cutoff],a[:cutoff],write,first=15,stride=4)
        for k in (0,1):
            np.testing.assert_allclose(p[k][:cutoff], q[k], atol=1e-12, rtol=0.)
    assert not p[0][:,0].any()
    reset = recovery_field(np.zeros_like(x),a,write,first=15,stride=4)
    assert not reset[0].any()


def test_embedding_energy_control_preserves_common_support():
    a = torch.zeros((3,4,64), dtype=torch.float64)
    b = torch.zeros_like(a)
    a[0,2,0], b[0,1,1] = 2., .2
    a[1,0,0], b[2,0,0] = 1., 1.
    u,v = matched_bias(a,b,.35)
    torch.testing.assert_close(u.norm(dim=(1,2)), torch.tensor([.35,0.,0.],dtype=u.dtype))
    torch.testing.assert_close(v.norm(dim=(1,2)), u.norm(dim=(1,2)))
    assert v[0,1,1] > 0 and u[0,2,0] > 0


def test_unobserved_channels_cannot_teach_or_be_written():
    x = trace([[0.,0.], [1.,2.], [0.,0.], [1.,2.], [0.,0.], [1.,2.], [1.,2.]])
    a = np.ones_like(x,dtype=bool)
    a[:,1] = False
    x[:,1] = np.nan
    real,wrong,_ = recovery_field(x,a,np.array([True,True]),first=11,stride=4)
    assert np.isfinite(real).all() and np.isfinite(wrong).all()
    assert not real[:,1].any() and not wrong[:,1].any()
