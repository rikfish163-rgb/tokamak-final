"""Actual signal-gradient integration for the direction-only CPU diagnostic."""

from copy import deepcopy

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.risk_aligned_online.simple23 import Config, Simple23
from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_step_20261107 import MetaOrderedTrajectory
from analyze_risk_direction_geometry_20261108 import local_updates


def fixture():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(151)
    base = Simple23(Config(dim=8, heads=2), torch.zeros(23), torch.ones(23)).eval()
    decoder = MetaOrderedTrajectory(base.center, base.scale, dim=8, scenarios=4).eval()
    values = torch.randn(1, 256, 23) * .1
    available = torch.ones_like(values, dtype=torch.bool)
    available[:, :, 2] = False
    origins = [127, 159, 191, 223]
    episodes = [dict(x=values[:, o-127:o+1], a=available[:, o-127:o+1],
                     truth=values[:, o+1:o+26], mask=available[:, o+1:o+26]) for o in origins]
    return base, decoder, episodes, values[:, -128:], available[:, -128:]


def test_actual_four_steps_preserve_source_and_frozen_tensors():
    base, decoder, es, qx, qa = fixture()
    initial = deepcopy(base.state_dict())
    ginitial = deepcopy(decoder.state_dict())
    with sdpa_kernel(SDPBackend.MATH):
        for rule in ('adam', 'momentum_direction'):
            model = FeatureIntra(base, decoder)
            result = local_updates(model, es, qx, qa, torch.ones(1), feedback='signal', rule=rule)
            assert len(result['steps']) == 4
            assert result['weight_displacement_norm'] > 0
            assert all(row['gradient_norm'] > 0 for row in result['steps'])
            if rule == 'momentum_direction':
                assert max(row['step_length_error'] for row in result['steps']) < 1e-7
            for key, value in model.risk.state_dict().items():
                if key != 'encoder.layers.0.linear2.weight':
                    assert torch.equal(value, initial[key])
    assert all(torch.equal(v, initial[k]) for k, v in base.state_dict().items())
    assert all(torch.equal(v, ginitial[k]) for k, v in decoder.state_dict().items())


def test_signal_updates_do_not_use_the_posthoc_query_label():
    base, decoder, es, qx, qa = fixture()
    with sdpa_kernel(SDPBackend.MATH):
        for rule in ('adam', 'momentum_direction'):
            a, b = FeatureIntra(base, decoder), FeatureIntra(base, decoder)
            left = local_updates(a, es, qx, qa, torch.zeros(1), feedback='signal', rule=rule)
            right = local_updates(b, es, qx, qa, torch.ones(1), feedback='signal', rule=rule)
            assert torch.equal(a.weight, b.weight)
            assert left['after']['probability'] == right['after']['probability']
            assert left['after']['bce'] != right['after']['bce']
