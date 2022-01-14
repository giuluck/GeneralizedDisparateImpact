import numpy as np
from moving_targets.masters import SingleTargetRegression
from moving_targets.metrics import Metric
from scipy.stats import pearsonr


class CausalExclusion(SingleTargetRegression):
    """Causal Exclusion Master.

    We define A as the (N, K) matrix of inputs to be excluded, where <N> is the number of data samples and <K> is the
    number of excluded features, so that A[i, j] represents the i-th value of the j-th excluded feature in the dataset.
    We then try to limit the casual relationship between the each excluded feature and the output by imposing the
    constraints "abs(w) <= theta$ with:
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
            A = x[:, features]
            lhs = np.dot(A.T, p)
            rhs = np.dot(A.T, A).dot(theta)
            return np.all(np.abs(lhs) <= rhs)

        super(CausalExclusion, self).__init__(satisfied=satisfied, backend=backend, alpha=alpha, beta=beta,
                                              lb=0.0, ub=float('inf'), y_loss=loss, p_loss=loss, stats=stats)

        self.features, self.theta = features, theta

    # noinspection PyPep8Naming
    def build(self, x, y, p):
        A = x[:, self.features]
        var = super(CausalExclusion, self).build(x, y, p)
        rhs = np.dot(A.T, A).dot(self.theta)
        lhs = np.array([self.backend.dot(row, var) for row in A.T])
        self.backend.add_constraints([left <= right for left, right in zip(lhs, rhs)])
        self.backend.add_constraints([left >= -right for left, right in zip(lhs, rhs)])
        return var


class CausalExclusionCovariance(SingleTargetRegression):
    """Causal Exclusion Master with Constraints on Covariance.

    Here, we rely on covariance as a metric for constraint satisfaction. However, since the covariance is not defined
    within a normalized interval, in order to set the <theta> values in an easier way we use a "percentage covariance",
    which is scaled on the covariance between the excluded features vector and the original targets (same as in the
    percentage DIDI used in fairness tasks).

    The set of constraints, thus, will be:
        abs(cov(A_i, z) / cov(A_i, y)) <= theta, for each column A_i in A
    where:
        - A_i is a vector containing the data of the i-th excluded feature
        - cov(a, b) is computed as (a - a.mean()) @ (b - b.mean())
    """

    def __init__(self, features, theta, backend='gurobi', alpha=1.0, beta=1.0, loss='mse', stats=True):
        self.features, self.theta = list(features), np.array(theta)

        # noinspection PyUnusedLocal
        def satisfied(x, y, p):
            percentage_covariances = [np.cov(f, p) / np.cov(f, y) for f in x.T[self.features]]
            return np.all(np.abs(percentage_covariances) <= theta)

        super(CausalExclusionCovariance, self).__init__(satisfied=satisfied, backend=backend, alpha=alpha, beta=beta,
                                                        lb=0.0, ub=float('inf'), y_loss=loss, p_loss=loss, stats=stats)

    def build(self, x, y, p):
        # retrieve variables from super class
        variables = super(CausalExclusionCovariance, self).build(x, y, p)
        # add auxiliary variable for the variable mean to speedup the constraints creation
        v_mean = self.backend.add_continuous_variable(lb=-float('inf'), ub=float('inf'))
        self.backend.add_constraint(len(variables) * v_mean == self.backend.sum(variables))
        # compute variances of model variables and original targets
        variance_v, variance_y = variables - v_mean, y - y.mean()
        for f, t in zip(x.T[self.features], self.theta):
            # compute variance of features vector and compute features/targets covariance
            variance_f = f - f.mean()
            covariance_yv = np.dot(variance_f, variance_y)
            # add auxiliary variable for features/variables covariance to speedup the constraints creation
            covariance_fv = self.backend.add_continuous_variable(lb=-float('inf'), ub=float('inf'))
            self.backend.add_constraint(covariance_fv == self.backend.dot(variance_f, variance_v))
            # add constraints on percentage covariance
            covariance_pct = covariance_fv / covariance_yv
            self.backend.add_constraints([covariance_pct <= t, covariance_pct >= -t])
        # return variables
        return variables


class PearsonCorrelation(Metric):
    def __init__(self, feature, name):
        super(PearsonCorrelation, self).__init__(name=name)
        self.feature = feature

    def __call__(self, x, y, p):
        return pearsonr(x[:, self.feature], p)[0]


class ZeroWeightCorrelation(Metric):
    def __init__(self, features, name):
        super(ZeroWeightCorrelation, self).__init__(name=name)
        self.features = features

    # noinspection PyPep8Naming
    def __call__(self, x, y, p):
        A = x[:, self.features]
        w, _, _, _ = np.linalg.lstsq(A, p, rcond=None)
        return np.abs(w).sum()


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res
