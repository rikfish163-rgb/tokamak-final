import torch

from motion_metric_span_20261107 import coefficients


def test_signed_weights_recover_a_known_reachable_direction():
    matrix = torch.zeros(1, 4, 3, dtype=torch.float64)
    matrix[0, :3] = torch.diag(torch.tensor([2., 3., 4.], dtype=torch.float64))
    target = torch.tensor([[1., -2., 3., 7.]], dtype=torch.float64)
    w, rank, fraction = coefficients(matrix, target)
    assert int(rank[0]) == 3
    assert w.abs().max() <= 2
    prediction = matrix @ w[:, :, None]
    expected = target.clone()
    expected[:, -1] = 0
    torch.testing.assert_close(prediction.squeeze(-1), expected)
    torch.testing.assert_close(fraction, torch.sqrt(torch.tensor([14/63], dtype=torch.float64)))


def test_numerically_tiny_unstable_direction_is_not_counted_as_capacity():
    matrix = torch.zeros(1, 4, 3, dtype=torch.float64)
    matrix[0, :3] = torch.diag(torch.tensor([1., .01, .00001], dtype=torch.float64))
    target = torch.tensor([[1., 0., 1., 0.]], dtype=torch.float64)
    w, rank, fraction = coefficients(matrix, target)
    assert int(rank[0]) == 2
    assert w[0, 2] == 0
    torch.testing.assert_close(fraction, torch.tensor([2**-.5], dtype=torch.float64))


def test_no_supported_gradient_does_not_create_an_oracle_direction():
    w, rank, fraction = coefficients(torch.zeros(2, 4, 3), torch.ones(2, 4))
    assert not w.any() and not rank.any() and not fraction.any()
