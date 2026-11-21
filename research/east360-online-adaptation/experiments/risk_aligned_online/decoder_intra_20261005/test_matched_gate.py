from copy import deepcopy

from experiments.risk_aligned_online.decoder_intra_20261005.prepare_matched import gate


def reference():
    return dict(maintenance=.2,whole=dict(balanced_loss=.3,metrics30=dict(tp=15,fp=2)),
                later=dict(mean_loss=.2))


def test_prediction_fit_alone_cannot_admit():
    initial=reference();candidate=deepcopy(initial);candidate['maintenance']=.1
    assert not gate(candidate,initial,initial)


def test_both_risk_losses_and_alarm_counts_required():
    initial=reference();candidate=deepcopy(initial)
    candidate['whole']['balanced_loss']=.29;candidate['later']['mean_loss']=.19
    assert gate(candidate,initial,initial)
    for key,value in [('tp',14),('fp',3)]:
        bad=deepcopy(candidate);bad['whole']['metrics30'][key]=value
        assert not gate(bad,initial,initial)
    bad=deepcopy(candidate);bad['later']['mean_loss']=.21
    assert not gate(bad,initial,initial)
    bad=deepcopy(candidate);bad['maintenance']=.221
    assert not gate(bad,initial,initial)
    better_static=reference();better_static['whole']['balanced_loss']=.28
    assert not gate(candidate,initial,better_static)
