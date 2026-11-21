import math
import unittest

import torch

from layer0_arrival_step_20261106 import bounded_step


class BoundedStepTests(unittest.TestCase):
    def test_actual_quadratic_picks_first_improving_step_per_row(self):
        target = torch.tensor([-1.0, -0.25, -0.0625, -1.0, -1.0, 0.0], dtype=torch.float64)
        gradient = torch.zeros((6, 128), dtype=torch.float64)
        gradient[:, 0] = -2 * target
        gradient.requires_grad_()
        radius = torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.float64, requires_grad=True)
        zero_loss = target.square().requires_grad_()
        support = torch.tensor([True, True, True, False, True, True])
        calls = []

        def evaluate(indices, action):
            calls.append((indices.clone(), action.clone()))
            return (action[:, 0] - target[indices]).square()

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, radius, zero_loss, evaluate, support=support
        )

        expected_actions = torch.zeros_like(gradient)
        expected_actions[0, 0] = -1
        expected_actions[1, 0] = -0.25
        torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
        torch.testing.assert_close(after_loss, torch.tensor([0, 0, 1 / 256, 1, 1, 0], dtype=torch.float64), rtol=0, atol=0)
        self.assertTrue(torch.equal(accepted, torch.tensor([True, True, False, False, False, False])))
        self.assertIsNone(anchor_final)
        self.assertEqual([indices.tolist() for indices, _ in calls], [[0, 1, 2], [1, 2], [1, 2], [2]])
        for indices, action in calls:
            self.assertTrue(torch.all(torch.linalg.vector_norm(action, dim=1) <= radius.detach()[indices]))
        for output in (actions, after_loss, accepted):
            self.assertFalse(output.requires_grad)
            self.assertIsNone(output.grad_fn)
        self.assertEqual(gradient[0, 0].item(), 2)
        self.assertEqual(zero_loss[0].item(), 1)

    def test_improvement_must_strictly_exceed_both_tolerance_scales(self):
        gradient = torch.zeros((2, 128), dtype=torch.float64)
        gradient[:, 0] = 1
        zero_loss = torch.tensor([10000.0, 1e-12], dtype=torch.float64)
        calls = []

        def evaluate(indices, action):
            calls.append(indices.tolist())
            large_row_loss = torch.where(action[:, 0] == -1, 9999.0, 9998.75)
            return torch.where(indices == 0, large_row_loss, 0.0)

        actions, after_loss, accepted, _ = bounded_step(
            gradient, torch.ones(2, dtype=torch.float64), zero_loss, evaluate,
            support=torch.ones(2, dtype=torch.bool),
        )
        self.assertEqual(actions[0, 0].item(), -0.5)
        self.assertTrue(torch.equal(actions[1], torch.zeros(128, dtype=torch.float64)))
        self.assertTrue(torch.equal(accepted, torch.tensor([True, False])))
        self.assertEqual(after_loss[0].item(), 9998.75)
        self.assertEqual(after_loss[1].item(), zero_loss[1].item())
        self.assertEqual(calls, [[0, 1], [0, 1], [1], [1]])

    def test_unsupported_and_zero_cases_do_not_call_either_callback(self):
        gradient = torch.zeros((4, 128), dtype=torch.float64)
        gradient[[0, 1, 3], 0] = 1
        radius = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        zero_loss = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
        support = torch.tensor([False, True, True, True])
        anchor_zero = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)

        def unexpected(*_args):
            self.fail("ineligible rows must not be scored")

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, radius, zero_loss, unexpected, support=support,
            anchor_gradient=gradient, anchor_zero=anchor_zero, anchor_eval=unexpected,
        )
        self.assertTrue(torch.equal(actions, torch.zeros_like(gradient)))
        self.assertTrue(torch.equal(after_loss, zero_loss))
        self.assertFalse(accepted.any().item())
        self.assertTrue(torch.equal(anchor_final, anchor_zero.detach()))
        self.assertFalse(anchor_final.requires_grad)

    def test_nonlinear_actual_anchor_is_corrected_twice_and_rechecked(self):
        gradient = torch.zeros((1, 128), dtype=torch.float64)
        gradient[0, 1] = 2
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[0, 0] = 1
        anchor_zero = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        risk_calls = []
        loss_calls = []

        def risk(indices, action):
            self.assertEqual(indices.tolist(), [0])
            risk_calls.append(action.clone())
            return 2 + action[:, 0] + 0.2 * action[:, 1].square()

        def loss(indices, action):
            self.assertEqual(indices.tolist(), [0])
            loss_calls.append(action.clone())
            return (action[:, 1] + 1).square()

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient.requires_grad_(), torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64), loss, support=torch.tensor([True]),
            anchor_gradient=anchor_gradient.requires_grad_(), anchor_zero=anchor_zero,
            anchor_eval=risk,
        )
        self.assertTrue(accepted.item())
        self.assertEqual(len(risk_calls), 3)
        self.assertEqual(len(loss_calls), 1)
        self.assertGreater(abs((2 + risk_calls[0][0, 0] + 0.2 * risk_calls[0][0, 1] ** 2).item() - 2), 1e-4)
        for candidate in risk_calls:
            self.assertLessEqual(torch.linalg.vector_norm(candidate[0]).item(), 1.0)
        actual_risk = 2 + actions[:, 0] + 0.2 * actions[:, 1].square()
        self.assertLessEqual(abs(actual_risk.item() - anchor_zero.item()), 1e-4)
        torch.testing.assert_close(anchor_final, actual_risk, rtol=0, atol=0)
        torch.testing.assert_close(after_loss, (actions[:, 1] + 1).square(), rtol=0, atol=0)
        torch.testing.assert_close(actions, loss_calls[0], rtol=0, atol=0)
        self.assertNotEqual(actions[0, 0].item(), 0)
        self.assertFalse(anchor_final.requires_grad)

    def test_parallel_projection_residue_does_not_trigger_scoring(self):
        gradient = torch.zeros((2, 128), dtype=torch.float64)
        gradient[:, 0] = 2
        gradient[:, 1] = torch.tensor([1e-11, 1e-9], dtype=torch.float64)
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[:, 0] = 1
        calls = []

        def risk(indices, action):
            calls.append(("risk", indices.tolist()))
            return action[:, 0]

        def loss(indices, action):
            calls.append(("loss", indices.tolist()))
            return 1e-8 + (gradient[indices] * action).sum(dim=1)

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, torch.ones(2, dtype=torch.float64),
            torch.full((2,), 1e-8, dtype=torch.float64), loss,
            support=torch.ones(2, dtype=torch.bool), anchor_gradient=anchor_gradient,
            anchor_zero=torch.zeros(2, dtype=torch.float64), anchor_eval=risk,
        )
        self.assertTrue(torch.equal(actions[0], torch.zeros(128, dtype=torch.float64)))
        self.assertEqual(actions[1, 1].item(), -1)
        self.assertTrue(torch.equal(accepted, torch.tensor([False, True])))
        self.assertEqual(after_loss[0].item(), 1e-8)
        self.assertTrue(torch.equal(anchor_final, torch.zeros(2, dtype=torch.float64)))
        self.assertEqual(calls, [("risk", [1]), ("loss", [1])])

    def test_zero_anchor_gradient_cannot_hide_infeasible_nonlinear_risk(self):
        gradient = torch.zeros((1, 128), dtype=torch.float64)
        gradient[0, 1] = 2
        risk_actions = []

        def risk(_indices, action):
            risk_actions.append(action.clone())
            return action[:, 1].square()

        def unexpected_loss(*_args):
            self.fail("loss must not be scored at an infeasible anchor")

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, torch.ones(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            unexpected_loss, support=torch.tensor([True]), anchor_gradient=torch.zeros_like(gradient),
            anchor_zero=torch.zeros(1, dtype=torch.float64), anchor_eval=risk,
        )
        self.assertEqual([a[0, 1].item() for a in risk_actions], [-1, -0.5, -0.25, -0.125])
        self.assertTrue(torch.equal(actions, torch.zeros_like(gradient)))
        self.assertEqual(after_loss.item(), 1)
        self.assertFalse(accepted.item())
        self.assertEqual(anchor_final.item(), 0)

    def test_failed_finite_corrections_backtrack_with_no_extra_policy(self):
        gradient = torch.zeros((1, 128), dtype=torch.float64)
        gradient[0, 1] = 2
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[0, 0] = 1
        risk_calls = []
        loss_calls = []

        def risk(_indices, action):
            risk_calls.append(action.clone())
            return action[:, 0] + 10 * action[:, 0].square() + 0.2 * action[:, 1].square()

        def loss(_indices, action):
            loss_calls.append(action.clone())
            return (action[:, 1] + 1).square()

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, torch.ones(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            loss, support=torch.tensor([True]), anchor_gradient=anchor_gradient,
            anchor_zero=torch.zeros(1, dtype=torch.float64), anchor_eval=risk,
        )
        self.assertTrue(accepted.item())
        self.assertEqual(len(risk_calls), 11)
        self.assertEqual(len(loss_calls), 1)
        self.assertEqual(actions[0, 1].item(), -0.125)
        self.assertLessEqual(abs(anchor_final.item()), 1e-4)
        self.assertLess(after_loss.item(), 1)
        for candidate in risk_calls:
            self.assertLessEqual(torch.linalg.vector_norm(candidate[0]).item(), 1.0)

    def test_nonfinite_scores_are_not_accepted_or_used_for_corrections(self):
        gradient = torch.zeros((1, 128), dtype=torch.float64)
        gradient[0, 0] = 1
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[0, 1] = 1
        risk_calls = []

        def risk(_indices, action):
            risk_calls.append(action.clone())
            return torch.full((1,), float("nan"), dtype=torch.float64)

        def unexpected_loss(*_args):
            self.fail("nonfinite risk is infeasible")

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, torch.ones(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            unexpected_loss, support=torch.tensor([True]), anchor_gradient=anchor_gradient,
            anchor_zero=torch.zeros(1, dtype=torch.float64), anchor_eval=risk,
        )
        self.assertEqual(len(risk_calls), 4)
        self.assertTrue(torch.equal(actions, torch.zeros_like(gradient)))
        self.assertEqual(after_loss.item(), 1)
        self.assertFalse(accepted.item())
        self.assertEqual(anchor_final.item(), 0)

    def test_mixed_anchor_batch_tracks_each_row_independently(self):
        gradient = torch.zeros((4, 128), dtype=torch.float64)
        gradient[:, 1] = 2
        gradient[3, 1] = 0
        gradient[3, 0] = 2
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[:, 0] = 1
        curvature = torch.tensor([0.0, 10.0, 0.0, 0.0], dtype=torch.float64)
        risk_rows = []
        loss_rows = []

        def risk(indices, action):
            risk_rows.append(indices.tolist())
            return action[:, 0] + curvature[indices] * action[:, 0].square() + 0.2 * action[:, 1].square()

        def loss(indices, action):
            loss_rows.append(indices.tolist())
            return (action[:, 1] + 1).square()

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64),
            loss, support=torch.tensor([True, True, False, True]),
            anchor_gradient=anchor_gradient, anchor_zero=torch.zeros(4, dtype=torch.float64),
            anchor_eval=risk,
        )
        self.assertTrue(torch.equal(accepted, torch.tensor([True, True, False, False])))
        self.assertEqual(actions[1, 1].item(), -0.125)
        self.assertTrue(torch.equal(actions[2:], torch.zeros((2, 128), dtype=torch.float64)))
        self.assertTrue(torch.equal(after_loss[2:], torch.ones(2, dtype=torch.float64)))
        actual_risk = actions[:, 0] + curvature * actions[:, 0].square() + 0.2 * actions[:, 1].square()
        torch.testing.assert_close(anchor_final, actual_risk, rtol=0, atol=0)
        self.assertTrue(torch.all(actual_risk.abs() <= 1e-4))
        self.assertTrue(torch.all(torch.linalg.vector_norm(actions, dim=1) <= 1))
        self.assertEqual(loss_rows, [[0], [1]])
        self.assertTrue(all(set(rows) <= {0, 1} for rows in risk_rows))

    def test_anchor_callback_owned_tensor_is_not_mutated(self):
        gradient = torch.zeros((1, 128), dtype=torch.float64)
        gradient[0, 0] = 1
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[0, 1] = 1
        cached_score = torch.zeros(1, dtype=torch.float64)
        calls = []

        def risk(_indices, action):
            calls.append(action.clone())
            if len(calls) == 1:
                return cached_score  # Actual risk at (-1, 0).
            return 1 + action[:, 0] + 2 * action[:, 1]

        bounded_step(
            gradient, torch.ones(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            lambda _indices, action: 1 + action[:, 0], support=torch.tensor([True]),
            anchor_gradient=anchor_gradient, anchor_zero=torch.ones(1, dtype=torch.float64),
            anchor_eval=risk,
        )
        self.assertGreater(len(calls), 1)
        self.assertEqual(cached_score.item(), 0)

    def test_nonfinite_loss_backtracks_then_preserves_exact_zero_on_failure(self):
        gradient = torch.zeros((2, 128), dtype=torch.float64)
        gradient[:, 0] = 1
        calls = []

        def loss(indices, action):
            calls.append(indices.tolist())
            first = action[:, 0] == -1
            return torch.where((indices == 0) & ~first, 0.5, float("nan"))

        actions, after_loss, accepted, _ = bounded_step(
            gradient, torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64),
            loss, support=torch.ones(2, dtype=torch.bool),
        )
        self.assertEqual(actions[0, 0].item(), -0.5)
        self.assertTrue(torch.equal(actions[1], torch.zeros(128, dtype=torch.float64)))
        self.assertTrue(torch.equal(after_loss, torch.tensor([0.5, 1], dtype=torch.float64)))
        self.assertTrue(torch.equal(accepted, torch.tensor([True, False])))
        self.assertEqual(calls, [[0, 1], [0, 1], [1], [1]])

    def test_float32_returned_actions_respect_original_radius(self):
        gradient = torch.ones((1, 128), dtype=torch.float32)
        radius = torch.tensor([0.1], dtype=torch.float32)

        def loss(indices, action):
            return 1 + (gradient[indices] * action).sum(dim=1)

        actions, after_loss, accepted, _ = bounded_step(
            gradient, radius, torch.ones(1), loss, support=torch.tensor([True])
        )
        self.assertTrue(accepted.item())
        self.assertEqual(actions.dtype, torch.float32)
        self.assertLessEqual(torch.linalg.vector_norm(actions.double(), dim=1).item(), radius.double().item())
        torch.testing.assert_close(after_loss, loss(torch.tensor([0]), actions), rtol=0, atol=0)

    def test_extreme_finite_gradient_norms_still_form_a_direction(self):
        gradient = torch.zeros((2, 128), dtype=torch.float64)
        gradient[:, 0] = torch.tensor([1e-200, 1e200], dtype=torch.float64)
        zero_loss = torch.tensor([1.0, 1e200], dtype=torch.float64)

        def loss(indices, action):
            x = action[:, 0]
            return torch.where(indices == 0, 1 + 1e-200 * x - 0.1 * x.square(), 1e200 * (1 + x))

        actions, after_loss, accepted, _ = bounded_step(
            gradient, torch.ones(2, dtype=torch.float64), zero_loss, loss,
            support=torch.ones(2, dtype=torch.bool),
        )
        self.assertTrue(accepted.all().item())
        self.assertTrue(torch.equal(actions[:, 0], torch.tensor([-1.0, -1.0], dtype=torch.float64)))
        torch.testing.assert_close(after_loss, torch.tensor([0.9, 0.0], dtype=torch.float64))

    def test_many_float64_actions_stay_inside_the_actual_radius(self):
        rng = torch.Generator().manual_seed(20261106)
        count = 1024
        gradient = torch.randn((count, 128), generator=rng, dtype=torch.float64)
        radius = torch.rand(count, generator=rng, dtype=torch.float64) + 0.01
        zero_loss = torch.full((count,), 100.0, dtype=torch.float64)

        def loss(indices, action):
            return zero_loss[indices] + (gradient[indices] * action).sum(dim=1)

        actions, after_loss, accepted, _ = bounded_step(
            gradient, radius, zero_loss, loss, support=torch.ones(count, dtype=torch.bool),
        )
        self.assertTrue(accepted.all().item())
        self.assertTrue(torch.all(torch.linalg.vector_norm(actions, dim=1) <= radius).item())
        precise_norms = [
            math.sqrt(math.fsum(value * value for value in row))
            for row in actions.tolist()
        ]
        self.assertTrue(all(norm <= cap for norm, cap in zip(precise_norms, radius.tolist())))
        self.assertTrue(torch.all((gradient * actions).sum(dim=1) < 0).item())
        torch.testing.assert_close(after_loss, loss(torch.arange(count), actions), rtol=0, atol=0)

    def test_extreme_finite_anchor_norms_still_project(self):
        gradient = torch.zeros((2, 128), dtype=torch.float64)
        gradient[:, 0] = 1
        gradient[:, 1] = 2
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[:, 0] = torch.tensor([1e-200, 1e200], dtype=torch.float64)

        def risk(indices, action):
            return anchor_gradient[indices, 0] * action[:, 0]

        def loss(_indices, action):
            return 1 + action[:, 0] + 2 * action[:, 1] + action[:, 1].square()

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64),
            loss, support=torch.ones(2, dtype=torch.bool), anchor_gradient=anchor_gradient,
            anchor_zero=torch.zeros(2, dtype=torch.float64), anchor_eval=risk,
        )
        self.assertTrue(accepted.all().item())
        self.assertTrue(torch.equal(actions[:, 0], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(actions[:, 1], -torch.ones(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(after_loss, torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(anchor_final, torch.zeros(2, dtype=torch.float64)))

    def test_direction_exists_even_when_full_gradient_norm_overflows(self):
        gradient = torch.full((1, 128), 1e308, dtype=torch.float64)
        zero_loss = torch.tensor([1e308], dtype=torch.float64)
        radius = torch.tensor([0.01], dtype=torch.float64)

        def loss(indices, action):
            return zero_loss[indices] + (gradient[indices] * action).sum(dim=1)

        actions, after_loss, accepted, _ = bounded_step(
            gradient, radius, zero_loss, loss, support=torch.tensor([True]),
        )
        self.assertTrue(accepted.item())
        self.assertTrue(torch.all(actions < 0).item())
        self.assertLess(after_loss.item(), zero_loss.item())
        self.assertLessEqual(math.hypot(*actions[0].tolist()), radius.item())

    def test_anchor_projection_exists_when_full_anchor_norm_overflows(self):
        gradient = torch.zeros((1, 128), dtype=torch.float64)
        gradient[0, 0] = 1e308
        anchor_gradient = torch.full_like(gradient, 1e308)
        radius = torch.tensor([1e-308], dtype=torch.float64)

        def risk(indices, action):
            return (anchor_gradient[indices] * action).sum(dim=1)

        def loss(indices, action):
            return 10 + (gradient[indices] * action).sum(dim=1)

        actions, after_loss, accepted, anchor_final = bounded_step(
            gradient, radius, torch.tensor([10.0], dtype=torch.float64), loss,
            support=torch.tensor([True]), anchor_gradient=anchor_gradient,
            anchor_zero=torch.zeros(1, dtype=torch.float64), anchor_eval=risk,
        )
        self.assertTrue(accepted.item())
        self.assertLess(actions[0, 0].item(), 0)
        self.assertTrue(torch.all(actions[0, 1:] > 0).item())
        self.assertLessEqual(math.hypot(*actions[0].tolist()), radius.item())
        self.assertLessEqual(abs(risk(torch.tensor([0]), actions).item()), 1e-4)
        torch.testing.assert_close(anchor_final, risk(torch.tensor([0]), actions))
        torch.testing.assert_close(after_loss, loss(torch.tensor([0]), actions))

    def test_callback_exceptions_propagate_and_partial_anchor_is_rejected(self):
        class ScoringError(RuntimeError):
            pass

        gradient = torch.zeros((1, 128), dtype=torch.float64)
        gradient[0, 0] = 1
        anchor_gradient = torch.zeros_like(gradient)
        anchor_gradient[0, 1] = 1
        radius = torch.ones(1, dtype=torch.float64)
        zero_loss = torch.ones(1, dtype=torch.float64)
        support = torch.tensor([True])

        def fail(*_args):
            raise ScoringError("real scoring failure")

        with self.assertRaisesRegex(ScoringError, "real scoring failure"):
            bounded_step(gradient, radius, zero_loss, fail, support=support)
        with self.assertRaisesRegex(ScoringError, "real scoring failure"):
            bounded_step(
                gradient, radius, zero_loss, fail, support=support,
                anchor_gradient=anchor_gradient, anchor_zero=zero_loss, anchor_eval=fail,
            )
        with self.assertRaises(ValueError):
            bounded_step(gradient, radius, zero_loss, fail, support=support, anchor_zero=zero_loss)


if __name__ == "__main__":
    unittest.main()
