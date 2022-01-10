import numpy as np
from moving_targets.masters import SingleTargetRegression


class CausalExclusion(SingleTargetRegression):
    def __init__(self, excluded_features, theta, backend='gurobi', alpha=1.0, beta=1.0, loss='mse', stats=True):
        # Given the vector <theta> containing the thresholds of the M excluded features, and the NxM matrix <A>
        # representing the weight of these excluded features for each of the N samples, the predictions satisfy the
        # constraint if it holds all of the M (dis)equivalences in:
        #   A.T @ p <= (A.T @ A) @ theta
        # which can be rewritten as a set of N constraints like:
        #   p <= A @ theta
        super(CausalExclusion, self).__init__(
            satisfied=lambda x, y, p: np.all(p <= np.dot(x[excluded_features], theta)),
            backend=backend, alpha=alpha, beta=beta, y_loss=loss, p_loss=loss, stats=stats
        )

        self.excluded_features = excluded_features
        self.theta = theta

    def build(self, x, y, p):
        variables = super(CausalExclusion, self).build(x, y, p)
        # The formula:
        #   V <= A @ theta,
        # with <V> the array of variables and <A> the matrix of features weights, can be unpacked in:
        #   v <= a @ theta, for i in {1, ..., N},
        # with <v> being the i-th element of the vector <V>, and <a> being the i-th row of the matrix <A>,
        # thus we can iterate the N pairs of variable/row and add the constraints accordingly
        A = x[self.excluded_features].values
        ATA = np.dot(A.T, A)

        for i in range(len(self.excluded_features)):
            self.backend.add_constraint(self.backend.sum(A.T[i] * variables) <= self.backend.sum(ATA[i] * self.theta))
        return variables
