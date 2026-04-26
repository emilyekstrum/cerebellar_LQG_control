""" LQG control model implementation 

true state dynamics: x[t+1] = A*x[t] + B*u[t] + V*v[t]
        - x[t] = true state (pos, vel)
        - u[t] = control input (acceleration)
        - v[t] = process noise (acceleration noise)
        - V = process noise gain (maps process noise to state space)

observations: y[t] = C*x[t] + W*w[t]
        - y[t] = observation (pos, vel)
        - w[t] = measurement noise (observation noise)
        - W = measurement noise gain (maps measurement noise to observation space)

L: Kalman gain (maps measurement residual to state estimate correction - how much to update the state estimate based on observations)
K: control/feedback gain (maps state estimate to control input - how much to control based on the state estimate)
"""

import numpy as np


# linear quadratic estimator (Kalman filter)
def lqe(A, C, V, W, X0):
    """estiamtes true state x[t]

    A: state transition matrix (maps state at time t to state at time t+1)
    C: observation matrix (maps state to observation)
    V: process noise gain (maps process noise to state space)
    W: measurement noise gain (maps measurement noise to observation space)
    X0: initial error covariance matrix (uncertainty about the initial state)
    """

    horizon = len(A)  # number of time steps
    for t in range(horizon):
        V[t] = V[t] * V[t].T  # ensure symmetric positive-definite covariance matrices
        W[t] = W[t] * W[t].T
    X0 = [
        X0[0] * X0[0].T
    ]  # ensure symmetric positive-definite initial error covariance matrix

    X1 = []  # list of error covariance matrices (uncertainty) after measurement update
    L = []  # list of Kalman gain matrices

    # Kalman filter recursion
    # K is computed by minimizing the trace of X1, which is the error covariance after measurement update
    # X0 is the error covariance after time update, which is computed by propagating X1 through the system dynamics and adding process noise V
    for t in range(horizon):
        L.append(
            X0[t] * C[t].T * np.linalg.pinv(C[t] * X0[t] * C[t].T + W[t])
        )  # Kalman gain computation using the error covariance X0 and the measurement noise W

        X1.append(
            X0[t] - L[t] * C[t] * X0[t]
        )  # Measurement update of the error covariance, which reduces the uncertainty based on the measurement noise W and the Kalman gain K

        if t < horizon - 1:
            X0.append(
                A[t] * X1[t] * A[t].T + V[t]
            )  # Time update of the error covariance, which propagates the uncertainty through the system dynamics and adds process noise V
            
    return L, X0, X1


# linear quadratic regulator
def lqr(A, B, Q, R):
    """computes optimal control gains L[t] for a finite horizon control problem with quadratic cost function"""

    horizon = len(A)
    K = [
        np.matrix(np.zeros((2, 2)))
    ]  # list of control gain matrices, initialized with zero matrix for the last time step (no control)
    P = [
        Q[horizon - 1]
    ]  # list of cost-to-go matrices, initialized with the terminal cost matrix Q at the last time step

    # LQR recursion
    # L is computed by minimizing the quadratic cost function, which consists of the state cost (x'*Q*x) and the control cost (u'*R*u)
    # P is the cost-to-go matrix, which is computed by working backwards through the system dynamics and
    # adding the state cost Q and the control cost R weighted by the control gain L
    for t in reversed(range(horizon)):
        K.append(
            -np.linalg.pinv(R[t] + B[t].T * P[horizon - (t + 1)] * B[t])
            * B[t].T
            * P[horizon - (t + 1)]
            * A[t]
        )  # Control gain computation using the cost-to-go matrix P and the control cost R
        if t > 0:
            P.append(
                A[t].T * P[horizon - (t + 1)] * (A[t] + B[t] * K[horizon - t]) + Q[t]
            )  # Cost-to-go update, which propagates the cost backward through the system dynamics and adds the state cost Q and the control cost R weighted by the control gain L
    return list(reversed(K)), list(reversed(P))


# transforms time-invariant into constant time-varying (list of) matrices
def tvar(var, horizon, idx=None):
    """helper to transform time-invariant matrices into time-varying (list of) matrices - basically evolve as time progresses."""
    if not type(var) is list:
        temp = []
        if not idx:
            for t in range(horizon):
                temp.append(var)
        else:
            for t in range(horizon):
                temp.append(np.matrix(np.zeros(np.shape(var))))
            temp[idx] = var
        var = temp
    return var


# main LQG class definition - LQR and LQE
class LQG:
    def __init__(self, horizon, target=None):
        self.horizon = horizon
        self.target = target
        self.var = {
            "A": None,
            "B": None,
            "C": None,
            "P": None,
            "Q": None,
            "R": None,
            "V": None,
            "W": None,
            "X0": None,
            "X1": None,
        }

    def define(self, string, val):
        """define mdoel paramters"""
        for char in string:
            if char in "ABCRVW":
                self.var[char] = tvar(val, self.horizon)
            elif char == "Q":
                self.var["Q"] = tvar(val, self.horizon, -1)
            elif char == "X":
                self.var["X0"] = tvar(val, 1)

    def set_target(self, target):
        """Set the target state [position, velocity]"""
        self.target = np.matrix(target).reshape(-1, 1)

    def kalman(self):
        """compute Kalman gain K[t] and error covariance matrices X0[t], X1[t] for the LQE (Kalman filter)"""
        self.var["L"], self.var["X0"], self.var["X1"] = lqe(
            self.var["A"], self.var["C"], self.var["V"], self.var["W"], self.var["X0"]
        )

    def control(self):
        """compute control gain L[t] and cost-to-go matrices P[t] for the LQR"""
        self.var["K"], self.var["P"] = lqr(
            self.var["A"], self.var["B"], self.var["Q"], self.var["R"]
        )

    def sample(
        self, n=1, x0=None, x=None, u=None, v=None, w=None, target=None, xhat=None
    ):
        """Sample trajectories from the LQG model. If x0, x, u, v, w are not provided, they are sampled from the model.

        Basic LQG trajectory algorithm:
        1. predict state estimate x0[t+1] = A*xhat[t] + B*u[t]
        2. sample true state x[t+1] = A*x[t] + B*u[t] + V*v[t]
        3. sample observation y[t] = C*x[t] + W*w[t]
        4. update state estimate xhat[t+1] = x0[t+1] + K[t+1]*(y[t+1] - C*x0[t+1])
        5. compute control input u[t+1] = L[t+1]*xhat[t+1]

        output: data dictionary with keys 'x', 'y', 'u', 'kf', 'noise', 'cost'
        - x: list of true state trajectories (pos, vel)
        - y: list of observation trajectories (pos, vel)
        - u: list of control input trajectories (acceleration)
        - kf: dictionary with keys 'x0' (list of state estimates before measurement update) and 'x1' (list of state estimates after measurement update)
        - noise: dictionary with keys 'x' (initial state error), 'v' (process noise), 'w' (measurement noise)
        - cost: dictionary with keys 'state' (list of state costs x'*Q*x) and 'control' (list of control costs u'*R*u)
        """

        y = None

        a = np.shape(self.var["A"][0])

        if self.var["X0"] is None:
            self.var["X0"] = [np.matrix(np.zeros(np.shape((a, a))))]

        # Set target: use parameter if provided, else use instance target, else default to zero
        if target is not None:
            target_state = np.matrix(target).reshape(-1, 1)
        elif self.target is not None:
            target_state = self.target
        else:
            target_state = np.matrix(np.zeros((a[0], 1)))

        if not self.var["C"] is None:
            c = np.shape(self.var["C"][0])
            self.kalman()  # compute Kalman gain and error covariance matrices for the LQE (Kalman filter)
            obs = True
        else:
            obs = False

        if not self.var["B"] is None:
            self.control()  # compute control gain and cost-to-go matrices for the LQR
            ctrl = True
        else:
            ctrl = False

        if x0 is None:  # default zero
            x0 = [np.matrix(np.zeros((a[0], n)))]
        elif min(np.shape(x0)) == 1 or type(x0) is list:  # provided mean
            x0 = np.reshape(np.matrix(x0), (max(np.shape(x0)), 1))
            x0 = [np.tile(x0, (1, n))]
        assert np.shape(x0[0]) == (a[0], n)

        if x is None:  # sample initial error
            e = np.random.randn(a[0], n)
            x = [x0[0] + self.var["X0"][0] * e]
        else:  # provided initial states
            assert np.shape(x) == (a[0], n)
            x = [x]
            e = np.linalg.pinv(self.var["X0"][0]) * (x[0] - x0[0])

        if w is None and obs:
            w = [np.random.randn(c[0], n)]
            for t in range(self.horizon - 1):
                w.append(np.random.randn(c[0], n))
        if v is None:
            v = [np.random.randn(a[0], n)]
            for t in range(self.horizon - 1):
                v.append(np.random.randn(a[0], n))

        if obs:
            y = [self.var["C"][0] * x[0] + self.var["W"][0] * w[0]]
            if xhat is None:
                x1 = [x0[0] + self.var["L"][0] * (y[0] - self.var["C"][0] * x0[0])]
            else:
                if min(np.shape(xhat)) == 1 or type(xhat) is list:
                    xhat = np.reshape(np.matrix(xhat), (max(np.shape(xhat)), 1))
                    xhat = np.tile(xhat, (1, n))
                assert np.shape(xhat) == (
                    a[0],
                    n,
                ), f"xhat shape {np.shape(xhat)} doesn't match expected shape {(a[0], n)}"
                x1 = [xhat]
        else:
            if xhat is None:
                x1 = [x0[0]]
            else:
                if min(np.shape(xhat)) == 1 or type(xhat) is list:
                    xhat = np.reshape(np.matrix(xhat), (max(np.shape(xhat)), 1))
                    xhat = np.tile(xhat, (1, n))
                assert np.shape(xhat) == (
                    a[0],
                    n,
                ), f"xhat shape {np.shape(xhat)} doesn't match expected shape {(a[0], n)}"
                x1 = [xhat]
        if u is None and ctrl:
            u = [
                self.var["K"][0] * (x1[0] - target_state)
            ]  # compute control input for the first time step using the control gain and the initial state estimate

        # sample trajectories for the remaining time steps using the LQG dynamics and control policy
        for t in range(self.horizon - 1):
            if ctrl:
                x0.append(self.var["A"][t] * x1[t] + self.var["B"][t] * u[t])
                x.append(
                    self.var["A"][t] * x[t]
                    + self.var["B"][t] * u[t]
                    + self.var["V"][t] * v[t]
                )
            else:
                x0.append(self.var["A"][t] * x1[t])
                x.append(self.var["A"][t] * x[t] + self.var["V"][t] * v[t])
            if obs:
                y.append(self.var["C"][t] * x[t] + self.var["W"][t] * w[t])
                x1.append(x0[t] + self.var["L"][t] * (y[t] - self.var["C"][t] * x0[t]))
            else:
                x1.append(x0[t])
            if ctrl:
                if len(u) <= t + 1:
                    u.append(self.var["K"][t] * (x1[t] - target_state))

        noise = {"x": e, "v": v, "w": w}
        kf = {"x0": x0, "x1": x1}
        data = {
            "control": ctrl,
            "x": x,
            "y": y,
            "u": u,
            "kf": kf,
            "noise": noise,
            "cost": {},
            "target": target_state,
        }

        return data
