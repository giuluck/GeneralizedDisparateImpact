import numpy as np
from moving_targets.masters import RegressionMaster
from moving_targets.metrics import Metric
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures


class Pearson(Metric):
    def __init__(self, feature, name='pearson'):
        super(Pearson, self).__init__(name=name)
        self.feature = feature

    def __call__(self, x, y, p):
        return pearsonr(x[self.feature], p)[0]


class CausalIndependence(Metric):
    def __init__(self, feature, name='independence'):
        super(CausalIndependence, self).__init__(name=name)
        self.feature = feature

    def __call__(self, x, y, p):
        a = PolynomialFeatures(degree=1).fit_transform(x[[self.feature]])
        w, _, _, _ = np.linalg.lstsq(a, p, rcond=None)
        return w[1]


class EffectCancellation(RegressionMaster):
    def __init__(self, features, theta, backend='gurobi', loss='mse', lb=-float('inf'), ub=float('inf'), stats=True):
        super().__init__(backend=backend, loss=loss, alpha='harmonic', lb=lb, ub=ub, stats=stats)
        self.features = features
        self.theta = theta

    def build(self, x, y, p):
        v = super(EffectCancellation, self).build(x, y, p)
        v_mean = self.backend.mean(v)
        for f in self.features:
            # for each feature vector a, we have that w = cov(a, v) / var(a), and since we want |w| <= theta,
            # we can impose the constraint -var(a) * theta <= cov(a, v) <= var(a) * theta
            a = x[f].values
            a_var = a.var()
            av_cov = self.backend.mean(a * v) - v_mean * a.mean()
            self.backend.add_constraints([av_cov >= -a_var * self.theta, av_cov <= a_var * self.theta])
        return v
