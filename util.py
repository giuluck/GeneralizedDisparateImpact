import numpy as np
from moving_targets.masters import SingleTargetRegression
from moving_targets.metrics import Metric


class EffectCancellation(SingleTargetRegression):
    """We define A as the (N, K) matrix of inputs related to the features to be excluded, where <N> is the number of
    data samples and <K> is the number of excluded features, so that A[i, j] represents the i-th value of the j-th
    excluded feature in the dataset. We then try to limit the casual relationship between the each excluded feature and
    the output by imposing the constraints "abs(w) <= theta$ with:
        - <w> being a K-sized vector so that "A @ w = z" holds (simple linear regression);
        - <theta> being a K-sized vector of threshold values.

    Since the linear system "A @ w = z" has no solution due to the fact that usually N > M, we need to minimize the
    residual error using the least-squares method, which leads to the formulation:
        (A.T @ A) @ w = A.T @ z
    from which we can derive that:
        w = ((A.T @ A)^-1 @ A.T) @ z
    and since our constraints are:
        abs(w) <= theta
    we can rewrite them
        -theta <= w <= theta
    and subsequently as:
        -theta <= ((A.T @ A)^-1 @ A.T) @ z <= theta
    or else:
        -(A.T @ A) @ theta <= A.T @ z <= (A.T @ A) @ theta
    which is our final set of (2 *) M constraints.
    """

    def __init__(self, features, theta, backend='gurobi', alpha=1.0, beta=1.0, loss='mse', stats=True):
        # noinspection PyUnusedLocal,PyPep8Naming
        def satisfied(x, y, p):
            A = x[features].values
            lhs = np.dot(A.T, p)
            rhs = np.dot(A.T, A).dot(theta)
            return np.all(np.abs(lhs) <= rhs)

        super(EffectCancellation, self).__init__(satisfied=satisfied, backend=backend, alpha=alpha, beta=beta,
                                                 lb=0.0, ub=float('inf'), y_loss=loss, p_loss=loss, stats=stats)

        self.features, self.theta = features, theta

    # noinspection PyPep8Naming
    def build(self, x, y, p):
        A = x[self.features].values
        var = super(EffectCancellation, self).build(x, y, p)
        rhs = np.dot(A.T, A).dot(self.theta)
        lhs = np.array([self.backend.dot(row, var) for row in A.T])
        self.backend.add_constraints([left <= right for left, right in zip(lhs, rhs)])
        self.backend.add_constraints([left >= -right for left, right in zip(lhs, rhs)])
        return var


class ZeroWeightCorrelation(Metric):
    def __init__(self, features, name='constraint'):
        super(ZeroWeightCorrelation, self).__init__(name=name)
        self.features = features

    def __call__(self, x, y, p):
        w, _, _, _ = np.linalg.lstsq(x[self.features].values, p, rcond=None)
        return np.abs(w).sum()


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res
