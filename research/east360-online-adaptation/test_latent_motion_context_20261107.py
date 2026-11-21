import inspect

import pytest
import torch

from latent_motion_context_20261107 import assemble


KINDS = ("summary", "level", "ordered", "stale", "swap", "global")


@pytest.fixture(params=(torch.float32, torch.float64), ids=("float32", "float64"))
def inputs(request):
    dtype = request.param
    raw = torch.arange(1, 139, dtype=dtype, device="cpu").reshape(2, 69)

    def state(starts, steps):
        starts = torch.tensor(starts, dtype=dtype, device=raw.device).reshape(2, 1)
        steps = torch.tensor(steps, dtype=dtype, device=raw.device).reshape(2, 1)
        return starts + steps * torch.arange(64, dtype=dtype, device=raw.device)

    return (
        raw,
        state((5, -3), (1, -1)),
        state((2, 7), (2, 2)),
        state((-1, 4), (-1, 3)),
    )


@pytest.mark.parametrize(
    ("kind", "expected_blocks"),
    (
        ("summary", (((0, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 0)))),
        ("level", (((5, 1), (-3, -1)), ((5, 1), (-3, -1)), ((0, 0), (0, 0)))),
        ("ordered", (((5, 1), (-3, -1)), ((2, 2), (7, 2)), ((3, -1), (-10, -3)))),
        ("stale", (((2, 2), (7, 2)), ((-1, -1), (4, 3)), ((3, 3), (3, -1)))),
        ("swap", (((2, 2), (7, 2)), ((5, 1), (-3, -1)), ((-3, 1), (10, 3)))),
        ("global", (((0, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 0)))),
    ),
)
def test_exact_blocks_and_unchanged_inputs(inputs, kind, expected_blocks):
    before = tuple(tensor.clone() for tensor in inputs)
    result = assemble(*inputs, kind)

    assert result.shape == (2, 261)
    assert result.dtype == inputs[0].dtype
    assert result.device == inputs[0].device
    expected_raw = torch.zeros_like(inputs[0]) if kind == "global" else inputs[0]
    torch.testing.assert_close(result[:, :69], expected_raw, rtol=0, atol=0)
    for index, rows in enumerate(expected_blocks):
        starts = torch.tensor(
            [row[0] for row in rows], dtype=inputs[0].dtype, device=inputs[0].device
        ).reshape(2, 1)
        steps = torch.tensor(
            [row[1] for row in rows], dtype=inputs[0].dtype, device=inputs[0].device
        ).reshape(2, 1)
        expected = starts + steps * torch.arange(64, dtype=inputs[0].dtype, device=inputs[0].device)
        torch.testing.assert_close(
            result[:, 69 + index * 64 : 69 + (index + 1) * 64],
            expected,
            rtol=0,
            atol=0,
        )
    for tensor, original in zip(inputs, before):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_equal_states_collapse_ordered_level_and_stale(inputs):
    raw, now, _, _ = inputs
    ordered = assemble(raw, now, now, now, "ordered")
    torch.testing.assert_close(ordered, assemble(raw, now, now, now, "level"), rtol=0, atol=0)
    torch.testing.assert_close(ordered, assemble(raw, now, now, now, "stale"), rtol=0, atol=0)


def test_swap_and_ordered_have_same_full_row_l2(inputs):
    ordered = assemble(*inputs, "ordered")
    swap = assemble(*inputs, "swap")
    torch.testing.assert_close(
        torch.linalg.vector_norm(ordered, dim=1),
        torch.linalg.vector_norm(swap, dim=1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_no_input_gradient_path(inputs, kind):
    leaves = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    result = assemble(*leaves, kind)
    assert not result.requires_grad
    assert result.grad_fn is None

    downstream_weight = torch.ones(
        261, 1, dtype=result.dtype, device=result.device, requires_grad=True
    )
    (result @ downstream_weight).sum().backward()
    assert downstream_weight.grad is not None
    assert all(tensor.grad is None for tensor in leaves)


def test_api_rejects_extra_future_tensor(inputs):
    assert list(inspect.signature(assemble).parameters) == [
        "raw", "now", "previous", "older", "kind"
    ]
    future = torch.zeros_like(inputs[1])
    with pytest.raises(TypeError):
        assemble(*inputs, "ordered", future)
    with pytest.raises(TypeError):
        assemble(*inputs, "ordered", future=future)


@pytest.mark.parametrize(
    ("index", "shape"),
    (
        (0, (69,)),
        (0, (2, 68)),
        (0, (2, 70)),
        (1, (2, 63)),
        (1, (2, 64, 1)),
        (2, (2, 63)),
        (2, (2, 64, 1)),
        (3, (2, 63)),
        (3, (2, 64, 1)),
    ),
)
def test_rejects_wrong_shape(inputs, index, shape):
    tensors = list(inputs)
    tensors[index] = torch.zeros(shape, dtype=inputs[0].dtype, device=inputs[0].device)
    with pytest.raises(ValueError, match="shape"):
        assemble(*tensors, "ordered")


@pytest.mark.parametrize("index", range(4))
def test_rejects_wrong_batch(inputs, index):
    tensors = list(inputs)
    width = 69 if index == 0 else 64
    tensors[index] = torch.zeros(3, width, dtype=inputs[0].dtype, device=inputs[0].device)
    with pytest.raises(ValueError, match="batch"):
        assemble(*tensors, "ordered")


@pytest.mark.parametrize("index", range(4))
def test_rejects_mixed_dtypes(inputs, index):
    tensors = list(inputs)
    other_dtype = torch.float64 if inputs[0].dtype == torch.float32 else torch.float32
    tensors[index] = tensors[index].to(other_dtype)
    with pytest.raises(ValueError, match="dtype"):
        assemble(*tensors, "ordered")


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_rejects_nonfinite_even_in_unused_inputs(inputs, index, value):
    tensors = list(inputs)
    tensors[index] = tensors[index].clone()
    tensors[index][0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        assemble(*tensors, "global")


def test_rejects_unknown_kind(inputs):
    with pytest.raises(ValueError, match="kind"):
        assemble(*inputs, "future")
