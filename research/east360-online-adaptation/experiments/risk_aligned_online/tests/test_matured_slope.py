import torch

from experiments.risk_aligned_online.expected_dynamics_20261028.matured_slope import SlopeMemory


def example():
    torch.manual_seed(131)
    x = torch.randn(36, 23, dtype=torch.float64).cumsum(0)
    a = torch.ones_like(x, dtype=torch.bool)
    a[9:13, 5] = False
    a[:, 2] = False
    prior = torch.randn(4, 23, dtype=torch.float64) / 4
    ridge = torch.ones(4, 23, dtype=torch.float64)
    return x, a, SlopeMemory(64, prior, ridge)


def test_matches_independent_matured_window_loop():
    x, a, model = example()
    result = model.trace(x, a, mode='memory', return_before=True)
    for t in range(len(x)):
        for h in range(1, 5):
            for c in (0, 2, 5, 22):
                numerator = model.ridge[h-1, c] * model.prior[h-1, c]
                denominator = model.ridge[h-1, c].clone()
                for arrival in range(max(0, t-15), t+1):
                    o = arrival-h
                    if o >= 1 and a[o-1, c] and a[o, c] and a[arrival, c]:
                        d = x[o, c]-x[o-1, c]
                        numerator = numerator+d*(x[arrival, c]-x[o, c])
                        denominator = denominator+d*d
                dnow = x[t, c]-x[t-1, c] if t and a[t, c] and a[t-1, c] else 0.
                torch.testing.assert_close(result['forecast'][t, h-1, c], numerator/denominator*dnow)
    assert torch.equal(result['delta'][:2], torch.zeros_like(result['delta'][:2]))
    assert result['updates'] == len(x)-2


def test_prefix_causal_reset_and_missing_values_inert():
    x, a, model = example()
    full = model.trace(x, a, mode='memory')
    torch.testing.assert_close(full['forecast'][:17], model.trace(x[:17], a[:17], mode='memory')['forecast'], rtol=0, atol=0)
    changed = x.clone(); changed[17:] += 1000; changed[~a] = float('nan')
    torch.testing.assert_close(full['forecast'][:17], model.trace(changed, a, mode='memory')['forecast'][:17], rtol=0, atol=0)
    torch.testing.assert_close(full['forecast'], model.trace(x, a, mode='memory')['forecast'], rtol=0, atol=0)
    assert torch.count_nonzero(full['forecast'][:, :, 2]) == 0
    assert torch.count_nonzero(model.trace(x, a, mode='off')['delta']) == 0


def test_short_constant_and_wrong_feedback_are_explicit():
    x, a, model = example()
    for n in (0, 1, 2, 3, 4):
        r = model.trace(x[:n], a[:n], mode='memory')
        assert r['forecast'].shape == (n, 4, 23)
        assert torch.isfinite(r['forecast_loss'])
    const = model.trace(torch.ones_like(x), a, mode='memory')
    assert torch.count_nonzero(const['forecast']) == 0
    real = model.trace(x, a, mode='memory')
    wrong = model.trace(x, a, mode='reverse')
    assert not torch.equal(real['forecast'], wrong['forecast'])
    assert torch.count_nonzero(model.trace(x, a, mode='hold')['forecast']-real['forecast']) > 0


def test_delayed_release_uses_old_coefficients_on_current_input():
    from experiments.risk_aligned_online.expected_dynamics_20261028.slope_release import delayed_forecast
    x, a, model = example()
    t = model.trace(x, a, mode='memory', return_before=True)
    f = delayed_forecast(t)
    for j in range(1, len(x)):
        d = torch.where(a[j]&a[j-1], x[j]-x[j-1], 0.)
        torch.testing.assert_close(f[j], t['coefficients'][j-1]*d, rtol=1e-12, atol=1e-12)
    changed = x.clone(); changed[20] += 100
    u = model.trace(changed, a, mode='memory', return_before=True)
    torch.testing.assert_close(t['coefficients'][19], u['coefficients'][19], rtol=0, atol=0)
    assert not torch.equal(f[20], delayed_forecast(u)[20])
