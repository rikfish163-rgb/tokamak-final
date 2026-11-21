import unittest

import numpy as np

import run_patch_sustained_margin_20261106 as trial


class SustainedMarginTests(unittest.TestCase):
    def test_single_spike_and_durable_pair_choose_different_states(self):
        a = np.array([[-0.15, -1.0, -0.18, -0.18]])
        g = np.array([[[1.0, 0, 4], [0, 0, 4], [0, 0.8, 4], [0, 0.8, 4]]])
        guard = np.array([[[0.0, 0, 1]]])
        state, frame = trial.optimized_states(a, g, guard, np.ones(a.shape, bool), 0.35)
        np.testing.assert_allclose(state, [[[0.35, 0, 0], [0, 0.35, 0]]], atol=1e-12)
        np.testing.assert_array_equal(frame, [[0, 2]])

    def test_maximin_respects_offsets_and_matches_dense_ball_search(self):
        a, h = np.array([[0.13, -0.10]]), np.array([[[1.5, 0.1], [-0.4, 0.9]]])
        radius = 0.35
        u, _ = trial.optimized_states(a, h, np.zeros((1, 1, 2)), np.ones_like(a, bool), radius)
        theta = np.linspace(0, 2 * np.pi, 12001)
        ball_edge = radius * np.column_stack((np.cos(theta), np.sin(theta)))
        expected = np.min(a + ball_edge @ h[0].T, axis=1).max()
        actual = np.min(a[0] + h[0] @ u[0, 1])
        self.assertGreaterEqual(actual, expected - 1e-8)
        self.assertLessEqual(np.linalg.norm(u[0, 1]), radius + 1e-12)

    def test_opposing_gradients_can_have_an_interior_optimum(self):
        a = np.array([[0.20, 0.00]])
        g = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
        state, _ = trial.optimized_states(a, g, np.zeros((1, 1, 2)), np.ones_like(a, bool), 0.35)
        np.testing.assert_allclose(state[0, 1], [-0.10, 0.0], atol=1e-12)

    def test_neither_padding_nor_timely_gaps_can_form_a_pair(self):
        native = np.arange(78) / 1000
        grid, valid = trial.target_frames(native, np.array([63]), down=0.084)
        np.testing.assert_array_equal(grid[0, :4], [72, 73, 74, 75])
        np.testing.assert_array_equal(np.flatnonzero(valid[0]), [0, 1])
        only_isolated = np.array([[True, False, True]])
        state, frame = trial.optimized_states(np.zeros((1, 3)), np.ones((1, 3, 2)),
                                              np.zeros((1, 1, 2)), only_isolated, 0.35)
        np.testing.assert_array_equal(frame, [[-1, -1]])
        self.assertFalse(state.any())

    def test_both_arms_and_signs_share_the_first_exactly_safe_scale(self):
        shape = (2, 2, 2, len(trial.SCALES), 4)
        shift = np.zeros(shape)
        shift[0, 1, 1, 0, -1] = trial.BOUND + 1e-4
        shift[1, 0, 0, :, 0] = -trial.BOUND - 1e-4
        factor, actual = trial.common_scale(shift)
        np.testing.assert_array_equal(factor, [trial.SCALES[1], 0])
        self.assertFalse(actual.any())

    def test_write_starts_after_eight_observations_and_stops_at_tail(self):
        native = np.arange(77, dtype=np.float32)
        blocks = trial.identity_blocks(native, np.array([63]))
        menu = trial.install_suffix(blocks, np.full((1, 2, 2, 24), -999, np.float32),
                                    np.array([63]), len(native), np.array([True]))
        np.testing.assert_array_equal(menu[0, 0, 1, :8], native[64:72])
        self.assertTrue((menu[:, 0, 1:, 8:13] == -999).all())
        self.assertFalse(menu[:, 0, :, 13:].any())
        np.testing.assert_array_equal(menu[1, 0, 0, :13], native[64:77])


if __name__ == "__main__":
    unittest.main()
