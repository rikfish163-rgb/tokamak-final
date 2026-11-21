import pytest
import torch


@pytest.fixture
def adam_step():
    from risk_meta_adam_20261108 import adam_step

    return adam_step


_BASE_GRADIENTS = (
    ((0.8, -0.4, 0.0), (0.3, -0.2, 0.5)),
    ((-0.5, 0.1, 0.0), (0.6, 0.4, -0.7)),
    ((0.2, -0.8, 0.0), (-0.4, 0.7, 0.1)),
    ((-0.3, 0.5, 0.0), (0.1, -0.6, 0.8)),
)


def _gradient_sequence(case, dtype):
    base = [torch.tensor(values, dtype=dtype) for values in _BASE_GRADIENTS]
    if case == "zero":
        return [torch.zeros_like(gradient) for gradient in base]
    if case == "small":
        return [gradient * 1e-10 for gradient in base]
    if case == "large":
        return [gradient * scale for gradient, scale in zip(base, (80, 30, 50, 100))]
    if case == "alternating":
        return [base[0] * sign for sign in (0.1, -0.1, 0.1, -0.1)]
    raise AssertionError(f"unknown gradient case: {case}")


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("case", ("zero", "small", "large", "alternating"))
@pytest.mark.parametrize("learning_rate", (1e-4, 3e-3))
def test_four_steps_match_torch_adamw_and_its_state(adam_step, dtype, case, learning_rate):
    original = torch.tensor([[1.25, -0.8, 0.0], [0.5, -1.0, 0.3]], dtype=dtype)
    weight = original.clone()
    reference = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.AdamW(
        [reference], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, foreach=False,
    )
    history = ()
    weight_atol = 2e-7 if dtype == torch.float32 else 2e-14
    state_rtol = 3e-6 if dtype == torch.float32 else 2e-13

    for step_number, gradient in enumerate(_gradient_sequence(case, dtype), 1):
        old_weight = weight.clone()
        old_gradient = gradient.clone()
        old_history = tuple(item.clone() for item in history)
        kwargs = {} if learning_rate == 1e-4 else {"lr": learning_rate}
        actual = adam_step(weight, gradient, history, **kwargs)
        reference.grad = gradient.clone()
        torch.nn.utils.clip_grad_norm_([reference], 1, error_if_nonfinite=True)
        optimizer.step()

        assert set(actual) == {"weight", "history", "exp_avg", "exp_avg_sq"}
        assert isinstance(actual["history"], tuple)
        assert len(actual["history"]) == step_number
        assert all(new is old for new, old in zip(actual["history"], history))
        torch.testing.assert_close(weight, old_weight, rtol=0, atol=0)
        torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
        for item, old_item in zip(history, old_history):
            torch.testing.assert_close(item, old_item, rtol=0, atol=0)
        torch.testing.assert_close(
            actual["history"][-1], reference.grad, rtol=state_rtol, atol=1e-30,
        )
        torch.testing.assert_close(actual["weight"], reference, rtol=0, atol=weight_atol)
        state = optimizer.state[reference]
        assert state["step"].item() == step_number
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[key], state[key], rtol=state_rtol, atol=1e-30)
        weight, history = actual["weight"], actual["history"]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_four_zero_steps_have_finite_analytic_first_derivatives(adam_step, dtype):
    lr, eps, beta1 = 1e-4, 1e-8, 0.9
    initial_weight = torch.zeros((2, 3), dtype=dtype, requires_grad=True)
    gradients = tuple(torch.zeros_like(initial_weight, requires_grad=True) for _ in range(4))
    weight, history = initial_weight, ()
    for gradient in gradients:
        result = adam_step(weight, gradient, history, lr=lr)
        weight, history = result["weight"], result["history"]

    derivatives = torch.autograd.grad(weight.sum(), (initial_weight, *gradients))
    torch.testing.assert_close(derivatives[0], torch.ones_like(initial_weight))
    for index, derivative in enumerate(derivatives[1:]):
        expected = -lr / eps * sum(
            (1 - beta1) * beta1 ** (step - 1 - index) / (1 - beta1 ** step)
            for step in range(index + 1, 5)
        )
        assert torch.isfinite(derivative).all()
        torch.testing.assert_close(derivative, torch.full_like(derivative, expected))


def test_zero_coordinate_with_nonzero_other_coordinates_has_finite_derivative(adam_step):
    lr, eps, beta1 = 1e-4, 1e-8, 0.9
    gradients = tuple(gradient.requires_grad_()
                      for gradient in _gradient_sequence("large", torch.float64))
    weight, history = torch.zeros_like(gradients[0]), ()
    for gradient in gradients:
        result = adam_step(weight, gradient, history, lr=lr)
        weight, history = result["weight"], result["history"]

    derivatives = torch.autograd.grad(weight[0, 2], gradients)
    for index, (gradient, derivative) in enumerate(zip(gradients, derivatives)):
        clip = min(1.0, 1.0 / (torch.linalg.vector_norm(gradient).item() + 1e-6))
        expected = -lr / eps * clip * sum(
            (1 - beta1) * beta1 ** (step - 1 - index) / (1 - beta1 ** step)
            for step in range(index + 1, 5)
        )
        assert torch.isfinite(derivative).all()
        torch.testing.assert_close(derivative[0, 2], derivative.new_tensor(expected))


def test_history_alone_retains_gradient_path_to_earlier_step(adam_step):
    first = torch.tensor([[0.12, -0.34], [0.23, -0.15]], dtype=torch.float64,
                         requires_grad=True)
    second = torch.tensor([[0.32, 0.14], [-0.12, 0.08]], dtype=torch.float64,
                          requires_grad=True)
    result1 = adam_step(torch.zeros_like(first), first)
    # The second input weight is independent of the first result, so any path
    # to the first gradient must go through the returned gradient history.
    result2 = adam_step(torch.zeros_like(second), second, result1["history"])
    first_derivative, second_derivative = torch.autograd.grad(
        result2["weight"].sum(), (first, second),
    )
    assert result2["history"][0] is result1["history"][0]
    assert torch.isfinite(first_derivative).all()
    assert torch.isfinite(second_derivative).all()
    assert torch.count_nonzero(first_derivative) > 0
    assert torch.count_nonzero(second_derivative) > 0


def test_four_step_derivatives_match_central_finite_differences(adam_step):
    values = (
        ((1.2, -0.7, 0.3), (0.5, -0.4, 0.8)),
        ((-0.4, 0.3, 0.2), (0.6, 0.5, -0.7)),
        ((0.2, -0.8, 0.4), (-0.3, 0.7, 0.1)),
        ((-0.3, 0.5, -0.6), (0.1, -0.6, 0.8)),
    )
    gradients = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
                      for value in values)
    coefficients = torch.tensor([[0.7, -0.3, 0.2], [-0.4, 0.5, 0.9]], dtype=torch.float64)

    def objective(sequence):
        weight, history = torch.zeros_like(coefficients), ()
        for gradient in sequence:
            result = adam_step(weight, gradient, history)
            weight, history = result["weight"], result["history"]
        return (weight * coefficients).sum()

    derivatives = torch.autograd.grad(objective(gradients), gradients)
    difference_step = 1e-6
    for index, derivative in enumerate(derivatives):
        numerical = torch.empty_like(derivative)
        for row in range(derivative.shape[0]):
            for column in range(derivative.shape[1]):
                plus = [gradient.detach().clone() for gradient in gradients]
                minus = [gradient.detach().clone() for gradient in gradients]
                plus[index][row, column] += difference_step
                minus[index][row, column] -= difference_step
                numerical[row, column] = (objective(plus) - objective(minus)) / (2 * difference_step)
        assert torch.isfinite(derivative).all()
        torch.testing.assert_close(derivative, numerical, rtol=2e-5, atol=2e-10)


def test_four_steps_propagate_through_differentiated_inner_losses(adam_step):
    targets = tuple(torch.tensor(value, dtype=torch.float64) for value in _BASE_GRADIENTS)
    features = torch.tensor([[0.3, -0.2, 0.6], [-0.1, 0.5, -0.4]], dtype=torch.float64)
    coefficients = torch.tensor([[0.7, -0.3, 0.2], [-0.4, 0.5, 0.9]], dtype=torch.float64)

    def objective(meta_parameter):
        weight = torch.tensor([[0.2, -0.1, 0.4], [0.6, -0.5, 0.3]],
                              dtype=torch.float64, requires_grad=True)
        history = ()
        for target in targets:
            inner_loss = (weight - target).square().sum() / 2
            inner_loss = inner_loss + meta_parameter * (weight * features).sum()
            gradient, = torch.autograd.grad(inner_loss, weight, create_graph=True)
            result = adam_step(weight, gradient, history, lr=3e-3)
            weight, history = result["weight"], result["history"]
        return (weight * coefficients).sum()

    meta_parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    derivative, = torch.autograd.grad(objective(meta_parameter), meta_parameter)
    difference_step = 1e-5
    numerical = (objective(meta_parameter + difference_step)
                 - objective(meta_parameter - difference_step)) / (2 * difference_step)
    assert torch.isfinite(derivative)
    assert derivative.abs() > 1e-6
    torch.testing.assert_close(derivative, numerical, rtol=2e-5, atol=2e-10)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), 3e38))
def test_nonfinite_total_norm_is_rejected_without_mutation(adam_step, bad_value):
    weight = torch.ones((2, 2), dtype=torch.float32)
    gradient = torch.tensor([[bad_value, bad_value], [0.0, 0.0]], dtype=torch.float32)
    original_weight, original_gradient = weight.clone(), gradient.clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        adam_step(weight, gradient)
    torch.testing.assert_close(weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(gradient, original_gradient, rtol=0, atol=0, equal_nan=True)
