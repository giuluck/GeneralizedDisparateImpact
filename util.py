import numpy as np
from moving_targets.masters import SingleTargetRegression

from moving_targets.metrics import Metric
from scipy.stats import pearsonr


# noinspection PyUnusedLocal,PyPep8Naming
class CausalExclusion(SingleTargetRegression):
    def __init__(self, excluded_features, theta, backend='gurobi', alpha=1.0, beta=1.0, loss='mse', stats=True):
        self.excluded_features, self.theta = list(excluded_features), np.array(theta)

        def satisfied(x, y, p):
            # Given the vector <theta> containing the thresholds of the M excluded features, and the NxM matrix <A>
            # representing the weight of these excluded features for each of the N samples, the predictions <p> satisfy
            # the constraint if all of the following M (dis)equivalences hold:
            #   -(A.T @ A) @ theta <= A.T @ p <= (A.T @ A) @ theta
            # which can be rewritten as:
            #   abs(A.T @ p) <= (A.T @ A) @ theta
            A = x[self.excluded_features].values
            thresholds = np.dot(np.dot(A.T, A), self.theta)
            return np.all(np.abs(np.dot(A.T, p)) <= thresholds)

        super(CausalExclusion, self).__init__(satisfied=satisfied, backend=backend, alpha=alpha, beta=beta,
                                              lb=0.0, ub=float('inf'), y_loss=loss, p_loss=loss, stats=stats)

    def build(self, x, y, p):
        variables = super(CausalExclusion, self).build(x, y, p)
        # The formula:
        #   -(A.T @ A) @ theta <= A.T @ variables <= (A.T @ A) @ theta,
        # with <variables> the array of variables and <A> the matrix of features weights, can be unpacked in:
        #   -t <= r @ variables <= t, for i in {1, ..., M},
        # with:
        #   - <r> being the i-th row of the MxN matrix <A.T>;
        #   - <t> being the i-th element of the M-sized vector <(A.T @ A) @ theta>, i.e., the i-th threshold;
        # thus we can iterate over the M pairs of (vectors, threshold) and add the constraints accordingly
        A = x[self.excluded_features].values
        thresholds = np.dot(np.dot(A.T, A), self.theta)
        # self.backend.add_constraints([self.backend.dot(r, variables) >= -t for r, t in zip(A.T, thresholds)])
        # self.backend.add_constraints([self.backend.dot(r, variables) <= t for r, t in zip(A.T, thresholds)])
        return variables


class PearsonCorrelation(Metric):
    def __init__(self, feature, name=None):
        super(PearsonCorrelation, self).__init__(name=f'pearson_{feature}' if name is None else name)
        self.feature = feature

    def __call__(self, x, y, p):
        return pearsonr(x[self.feature], p)[0]


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res
