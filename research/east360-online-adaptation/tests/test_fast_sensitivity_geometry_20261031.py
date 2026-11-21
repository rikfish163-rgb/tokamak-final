"""Validate the post-slow derivative against an actual finite change."""
import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.slow_fast_attribution_20261031.transitions import local_geometry


@pytest.mark.parametrize('zero', [False, True])
def test_geometry_matches_real_component_forward_and_small_step(zero):
    old_path = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(31101)
            base = Simple23(Config(dim=8, heads=2, fast_site='tokens'), torch.zeros(23), torch.ones(23))
            model = ContextAdapter(base).double().eval()
            h = torch.randn(2, 17, 8, dtype=torch.float64)
            valid = torch.ones(2, 17, dtype=torch.bool)
            delta = torch.randn(2, 92, dtype=torch.float64)*.1
            if zero:
                delta.zero_()
            states = {name:dict(slow=torch.randn(8, 8, dtype=torch.float64)*.1,
                               fast_output=torch.randn(8, 92, dtype=torch.float64)*.1)
                      for name in ('cold', 'both')}
            case = dict(at_same_endpoint={})
            for s in states:
                model.restore(states[s])
                case['at_same_endpoint'][s+'_off'] = dict(logit=float(model.risk(h, valid, delta*0)[0]))
                for o in states:
                    model.restore(dict(slow=states[s]['slow'], fast_output=states[o]['fast_output']))
                    arm = {('cold','cold'):'cold_on', ('cold','both'):'cold_slow_both_readout',
                           ('both','cold'):'both_slow_cold_readout', ('both','both'):'both_on'}[s,o]
                    case['at_same_endpoint'][arm] = dict(logit=float(model.risk(h, valid, delta)[0]))
            before = {k:v.clone() for k,v in model.state_dict().items()}
            h0, d0 = h.clone(), delta.clone()
            result = local_geometry(model, h, valid, delta, states, 0, case)
            assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
            assert torch.equal(h, h0) and torch.equal(delta, d0)
            assert all(p.grad is None for p in model.parameters())
            assert -1.000000001 <= result['gradient_direction_cosine'] <= 1.000000001
            for s in states:
                assert result[s]['gradient_norm'] > 0
                assert result[s]['layernorm1_denominator'] > 0
                for o in states:
                    row = result[s]['readouts'][o]
                    model.restore(dict(slow=states[s]['slow'], fast_output=states[o]['fast_output']))
                    step = 1e-5
                    zero_z = model.risk(h, valid, delta*0)[0]
                    plus = model.risk(h, valid, delta*step)[0]
                    minus = model.risk(h, valid, -delta*step)[0]
                    finite = float((plus-minus)/(2*step))
                    assert finite == pytest.approx(row['first_order_logit_change'], rel=1e-5, abs=1e-8)
                    assert row['interpolation'][0]['logit'] == float(zero_z)
                    if zero:
                        assert row['first_order_logit_change'] == row['actual_logit_change'] == 0
                        assert row['action_norm'] == 0
    finally:
        torch.backends.mha.set_fastpath_enabled(old_path)
