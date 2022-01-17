import numpy as np
from moving_targets.masters import SingleTargetRegression
from moving_targets.metrics import CausalIndependence


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

    def __init__(self, features, theta, backend='gurobi', alpha=1.0, loss='mse', lb=0.0, ub=float('inf'), stats=True):
        self.theta = theta
        self.features = features
        self.metric = CausalIndependence(features=self.features, aggregation=None)
        super(EffectCancellation, self).__init__(
            satisfied=lambda x, y, p: np.all([w <= t for w, t in zip(self.metric(x, y, p).values(), self.theta)]),
            backend=backend, alpha=alpha, beta=None, lb=lb, ub=ub, y_loss=loss, p_loss=loss, stats=stats
        )

    # noinspection PyPep8Naming
    def build(self, x, y, p):
        A = x[self.features].values
        variables = super(EffectCancellation, self).build(x, y, p)
        # add auxiliary variables for the regressor weights and impose constraints representing the regression task
        weights = self.backend.add_continuous_variables(A.shape[1], name='w')
        linear_regression_lhs = self.backend.dot(A.T @ A, weights)
        linear_regression_rhs = self.backend.dot(A.T, variables)
        self.backend.add_constraints([lrl == lrr for lrl, lrr in zip(linear_regression_lhs, linear_regression_rhs)])
        # add model constraints on the regressor weights
        self.backend.add_constraints([w <= t for w, t in zip(weights, self.theta)])
        self.backend.add_constraints([w >= -t for w, t in zip(weights, self.theta)])
        return variables


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res
