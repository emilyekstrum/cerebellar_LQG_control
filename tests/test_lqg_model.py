"""Tests for lqg_model.py

Covers: tvar helper, lqe (Kalman filter), lqr (LQ regulator), and LQG.sample().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from model.lqg_model import LQG, lqe, lqr, tvar



# Shared helpers

def _make_1d_lqg(horizon=50, target=(10.0, 0.0)):
    """LQG for 1-D motion: state=[pos, vel], control=[accel]"""
    dt = 0.01
    model = LQG(horizon)
    model.set_target(list(target))
    model.define("A", np.matrix([[1, dt], [0, 1]]))
    model.define("B", np.matrix([[0], [dt]]))
    model.define("C", np.matrix(np.eye(2)))
    model.define("Q", np.matrix(np.diag([100.0, 1.0])))
    model.define("R", np.matrix([[0.001]]))
    model.define("V", np.matrix(0.01 * np.eye(2)))
    model.define("W", np.matrix(1.0 * np.eye(2)))
    model.define("X", np.matrix(0.1 * np.eye(2)))
    return model


def _lqe_inputs(horizon=10):
    """Return (A, C, V, W, X0) list inputs for a 2-state system.
    """
    dt = 0.01
    return (
        tvar(np.matrix([[1, dt], [0, 1]]), horizon),
        tvar(np.matrix(np.eye(2)), horizon),
        tvar(np.matrix(0.1 * np.eye(2)), horizon),  # V — will be squared
        tvar(np.matrix(1.0 * np.eye(2)), horizon),  # W — will be squared
        [np.matrix(0.1 * np.eye(2))],               # X0
    )



# tvar

class TestTvar:
    def test_wraps_matrix_into_list_of_correct_length(self):
        m = np.matrix(np.eye(3))
        result = tvar(m, 7)
        assert isinstance(result, list)
        assert len(result) == 7
        for elem in result:
            np.testing.assert_array_equal(elem, m)

    def test_returns_existing_list_unchanged(self):
        lst = [np.matrix(np.eye(2))] * 4
        assert tvar(lst, 4) is lst

    def test_idx_places_value_only_at_that_index(self):
        m = np.matrix(np.diag([5.0, 5.0]))
        result = tvar(m, 4, idx=-1)
        assert np.any(result[-1] != 0)
        for i in range(len(result) - 1):
            np.testing.assert_array_equal(result[i], np.zeros((2, 2)))


# lqe (Kalman filter)

class TestLQE:
    def test_output_list_lengths(self):
        horizon = 8
        A, C, V, W, X0 = _lqe_inputs(horizon)
        L, X0_out, X1 = lqe(A, C, V, W, X0)
        assert len(L) == horizon
        assert len(X0_out) == horizon
        assert len(X1) == horizon

    def test_kalman_gain_shape(self):
        horizon = 5
        A, C, V, W, X0 = _lqe_inputs(horizon)
        L, _, _ = lqe(A, C, V, W, X0)
        # C is (2,2) eye; Kalman gain shape = (n_states, n_obs) = (2, 2)
        for t in range(horizon):
            assert np.shape(L[t]) == (2, 2)

    def test_measurement_update_reduces_uncertainty(self):
        # trace(X1[t]) <= trace(X0[t]) at every step
        horizon = 10
        A, C, V, W, X0 = _lqe_inputs(horizon)
        L, X0_out, X1 = lqe(A, C, V, W, X0)
        for t in range(horizon):
            assert np.trace(X1[t]) <= np.trace(X0_out[t]) + 1e-10

    def test_covariance_matrices_are_positive_semidefinite(self):
        horizon = 10
        A, C, V, W, X0 = _lqe_inputs(horizon)
        L, X0_out, X1 = lqe(A, C, V, W, X0)
        for t in range(horizon):
            assert np.all(np.linalg.eigvalsh(X0_out[t]) >= -1e-10)
            assert np.all(np.linalg.eigvalsh(X1[t]) >= -1e-10)

    def test_kalman_gain_eigenvalues_between_zero_and_one(self):
        # For C=I: L = X0*(X0+W)^{-1}, eigenvalues in [0,1]
        horizon = 5
        A, C, V, W, X0 = _lqe_inputs(horizon)
        L, _, _ = lqe(A, C, V, W, X0)
        for t in range(horizon):
            eigs = np.linalg.eigvalsh(L[t])
            assert np.all(eigs >= -1e-10)
            assert np.all(eigs <= 1.0 + 1e-10)

    def test_higher_measurement_noise_gives_smaller_kalman_gain(self):
        # more measurement noise -> trust observations less -> smaller gain
        horizon = 5
        A_l, C_l, V_l, W_low, X0_l = _lqe_inputs(horizon)
        A_h, C_h, V_h, W_high, X0_h = _lqe_inputs(horizon)
        for t in range(horizon):
            W_high[t] = W_high[t] * 10  # 10× noisier (before squaring in lqe)

        L_low, _, _ = lqe(A_l, C_l, V_l, W_low, X0_l)
        L_high, _, _ = lqe(A_h, C_h, V_h, W_high, X0_h)

        total_low = sum(float(np.sum(np.abs(L_low[t]))) for t in range(horizon))
        total_high = sum(float(np.sum(np.abs(L_high[t]))) for t in range(horizon))
        assert total_low > total_high


# lqr (LQ regulator)

class TestLQR:
    def setup_method(self):
        self.h = 10
        dt = 0.01
        self.A = tvar(np.matrix([[1, dt], [0, 1]]), self.h)
        self.B = tvar(np.matrix([[0], [dt]]), self.h)
        self.Q = tvar(np.matrix(np.diag([100.0, 1.0])), self.h, -1)
        self.R = tvar(np.matrix([[0.001]]), self.h)

    def test_output_list_lengths(self):
        K, P = lqr(self.A, self.B, self.Q, self.R)
        # K has h+1 entries: h computed gains + 1 zero initialiser at the end
        assert len(K) == self.h + 1
        assert len(P) == self.h

    def test_control_gain_shape(self):
        K, P = lqr(self.A, self.B, self.Q, self.R)
        # Usable gains are K[:-1]; shape = (n_controls, n_states) = (1, 2)
        for k in K[:-1]:
            assert np.shape(k) == (1, 2)

    def test_cost_to_go_positive_semidefinite(self):
        K, P = lqr(self.A, self.B, self.Q, self.R)
        for p in P:
            eigvals = np.linalg.eigvalsh(p)
            assert np.all(eigvals >= -1e-10), f"P not PSD, min eigenvalue: {eigvals.min()}"

    def test_higher_state_cost_increases_control_gains(self):
        Q_low = tvar(np.matrix(np.diag([1.0, 0.01])), self.h, -1)
        Q_high = tvar(np.matrix(np.diag([1000.0, 10.0])), self.h, -1)
        K_low, _ = lqr(self.A, self.B, Q_low, self.R)
        K_high, _ = lqr(
            tvar(np.matrix([[1, 0.01], [0, 1]]), self.h),
            tvar(np.matrix([[0], [0.01]]), self.h),
            Q_high,
            tvar(np.matrix([[0.001]]), self.h),
        )
        sum_abs = lambda K: sum(float(np.sum(np.abs(k))) for k in K[:-1])
        assert sum_abs(K_high) > sum_abs(K_low)

    def test_terminal_initializer_is_zero(self):
        # K[-1] is the zero matrix used to seed the backward recursion
        K, _ = lqr(self.A, self.B, self.Q, self.R)
        np.testing.assert_array_equal(K[-1], np.zeros_like(K[-1]))



# LQG class

class TestLQGDefineAndTarget:
    def test_define_populates_variable_as_list(self):
        model = LQG(10)
        A = np.matrix([[1, 0.01], [0, 1]])
        model.define("A", A)
        assert model.var["A"] is not None
        assert len(model.var["A"]) == 10
        for a in model.var["A"]:
            np.testing.assert_array_equal(a, A)

    def test_set_target_reshapes_to_column_vector(self):
        model = LQG(10)
        model.set_target([5.0, 0.0])
        assert model.target.shape == (2, 1)
        assert model.target[0, 0] == 5.0
        assert model.target[1, 0] == 0.0


class TestLQGSample:
    def test_output_has_required_keys(self):
        np.random.seed(0)
        data = _make_1d_lqg(horizon=30).sample(n=1)
        for key in ("x", "y", "u", "kf", "noise", "target", "control"):
            assert key in data

    def test_trajectory_list_lengths(self):
        np.random.seed(0)
        horizon = 30
        data = _make_1d_lqg(horizon=horizon).sample(n=1)
        assert len(data["x"]) == horizon
        assert len(data["y"]) == horizon
        assert len(data["u"]) == horizon
        assert len(data["kf"]["x0"]) == horizon
        assert len(data["kf"]["x1"]) == horizon

    def test_state_vector_shape_single_sample(self):
        np.random.seed(0)
        data = _make_1d_lqg(horizon=20).sample(n=1)
        for t in range(20):
            assert data["x"][t].shape == (2, 1)
            assert data["kf"]["x1"][t].shape == (2, 1)

    def test_state_matrix_shape_multiple_samples(self):
        np.random.seed(0)
        n = 5
        data = _make_1d_lqg(horizon=20).sample(n=n)
        for t in range(20):
            assert data["x"][t].shape == (2, n)

    def test_convergence_to_target_position(self):
        # after enough steps the controller should bring position near target
        np.random.seed(42)
        data = _make_1d_lqg(horizon=500, target=(10.0, 0.0)).sample(n=1)
        final_pos = data["x"][-1][0, 0]
        assert abs(final_pos - 10.0) < 3.0

    def test_late_trajectory_closer_to_target_than_early(self):
        np.random.seed(7)
        data = _make_1d_lqg(horizon=400, target=(10.0, 0.0)).sample(n=1)
        target_pos = 10.0
        early_err = abs(np.mean([data["x"][t][0, 0] for t in range(20, 40)]) - target_pos)
        late_err = abs(np.mean([data["x"][t][0, 0] for t in range(360, 400)]) - target_pos)
        assert late_err < early_err

    def test_estimate_tracks_true_state(self):
        np.random.seed(0)
        data = _make_1d_lqg(horizon=200, target=(10.0, 0.0)).sample(n=1)
        errors = [abs(data["x"][t][0, 0] - data["kf"]["x1"][t][0, 0]) for t in range(50, 200)]
        assert np.mean(errors) < 5.0

    def test_observation_noise_bounds_estimate_error(self):
        # with low measurement noise the estimate should track true state more tightly
        np.random.seed(0)
        dt = 0.01
        horizon = 200

        def run(W_scale):
            model = LQG(horizon)
            model.set_target([10.0, 0.0])
            model.define("A", np.matrix([[1, dt], [0, 1]]))
            model.define("B", np.matrix([[0], [dt]]))
            model.define("C", np.matrix(np.eye(2)))
            model.define("Q", np.matrix(np.diag([100.0, 1.0])))
            model.define("R", np.matrix([[0.001]]))
            model.define("V", np.matrix(0.01 * np.eye(2)))
            model.define("W", np.matrix(W_scale * np.eye(2)))
            model.define("X", np.matrix(0.1 * np.eye(2)))
            np.random.seed(0)
            data = model.sample(n=1)
            return np.mean([abs(data["x"][t][0, 0] - data["kf"]["x1"][t][0, 0]) for t in range(50, horizon)])

        err_low = run(0.1)
        err_high = run(5.0)
        assert err_low < err_high

    def test_no_control_no_observation_sets_fields_to_none(self):
        np.random.seed(0)
        model = LQG(10)
        model.define("A", np.matrix([[1, 0.01], [0, 1]]))
        model.define("V", np.matrix(0.01 * np.eye(2)))
        data = model.sample(n=1)
        assert data["control"] is False
        assert data["y"] is None
        assert data["u"] is None

    def test_observation_only_no_control(self):
        np.random.seed(0)
        horizon = 20
        model = LQG(horizon)
        model.define("A", np.matrix([[1, 0.01], [0, 1]]))
        model.define("C", np.matrix(np.eye(2)))
        model.define("V", np.matrix(0.01 * np.eye(2)))
        model.define("W", np.matrix(1.0 * np.eye(2)))
        model.define("X", np.matrix(0.1 * np.eye(2)))
        data = model.sample(n=1)
        assert data["control"] is False
        assert data["y"] is not None
        assert len(data["y"]) == horizon

    def test_provided_x0_initialises_near_given_mean(self):
        # with small initial covariance, x[0] should be close to x0
        np.random.seed(99)
        data = _make_1d_lqg(horizon=50, target=(10.0, 0.0)).sample(n=1, x0=[3.0, 0.5])
        assert abs(data["x"][0][0, 0] - 3.0) < 1.0
        assert abs(data["x"][0][1, 0] - 0.5) < 1.0

    def test_control_is_active_flag(self):
        np.random.seed(0)
        data = _make_1d_lqg(horizon=20).sample(n=1)
        assert data["control"] is True
