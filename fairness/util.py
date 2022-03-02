import numpy as np
from moving_targets.masters import Master
from moving_targets.masters.losses import MAE, MSE, HammingDistance, CrossEntropy
from moving_targets.metrics import DIDI

from moving_targets.metrics import Metric
from moving_targets.util import probabilities


class LrWeight(Metric):
    def __init__(self, protected='race', aggregate=False, percentage=True, name='weight'):
        super(LrWeight, self).__init__(name=name)
        self.protected = protected
        self.aggregate = aggregate
        self.percentage = percentage

    def __call__(self, x, y, p):
        f = [c for c in x.columns if c.startswith(self.protected)]
        x = np.concatenate((np.ones((len(x), 1)), x[f]), axis=1)

        p_weights = np.linalg.lstsq(x, p, rcond=None)[0][1:]
        y_weights = np.linalg.lstsq(x, y, rcond=None)[0][1:] if self.percentage else np.ones_like(p_weights)

        # TODO: rollback
        # from sklearn.linear_model import LogisticRegression
        # p_weights = LogisticRegression(fit_intercept=False)
        # p_weights.fit(x, p.round())
        # p_weights = p_weights.coef_[1:]
        # y_weights = LogisticRegression(fit_intercept=False)
        # y_weights.fit(x, p.round())
        # y_weights = y_weights.coef_[1:]

        weights = p_weights / y_weights
        if self.aggregate:
            max_abs = np.argmax(np.abs(weights))
            return weights[max_abs]
        else:
            return {f: w for f, w in zip(f, weights)}


class FairnessMaster(Master):
    def __init__(self, classification, backend, loss, alpha, stats, protected):
        self.protected = protected
        self.classification = classification
        super(FairnessMaster, self).__init__(backend=backend, loss=loss, alpha=alpha, stats=stats)

    def _get_loss(self, loss):
        if loss == 'hd':
            assert self.classification, "HammingDistance is not a valid regression loss"
            return HammingDistance(labelling=False)
        elif loss == 'ce':
            assert self.classification, "CrossEntropy is not a valid regression loss"
            return CrossEntropy(binary=True)
        elif loss == 'mae':
            return MAE(binary=self.classification)
        elif loss == 'mse':
            return MSE(binary=self.classification)
        else:
            raise AssertionError(f"Unknown loss '{loss}'")

    def build(self, x, y, p):
        assert y.ndim == 1, f"Target vector must be one-dimensional, got shape {y.shape}"
        lb, ub = (0, 1) if self.classification else (0, float('inf'))
        vtype = 'binary' if self.loss.binary else 'continuous'
        return self.backend.add_variables(len(y), vtype=vtype, lb=lb, ub=ub, name='y')


class DidiMaster(FairnessMaster):
    def __init__(self, classification, backend, loss, alpha, stats, protected='race', violation=0.2):
        super(DidiMaster, self).__init__(classification, backend, loss, alpha, stats, protected=protected)
        self.violation = violation

    def build(self, x, y, p):
        variables = super(DidiMaster, self).build(x, y, p)
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected=self.protected)
        deviations = self.backend.add_continuous_variables(len(indicator_matrix), lb=0.0, name='deviations')
        total_avg = self.backend.mean(variables)
        for g, protected_group in enumerate(indicator_matrix):
            protected_vars = variables[protected_group]
            if len(protected_vars) > 0:
                protected_avg = self.backend.mean(protected_vars)
                self.backend.add_constraint(deviations[g] >= total_avg - protected_avg)
                self.backend.add_constraint(deviations[g] >= protected_avg - total_avg)
        # in classification scenarios the didi is sum(sum(devs[:, i])), which for binary tasks equals to 2 * sum(devs)
        if self.classification:
            didi = 2 * self.backend.sum(deviations)
            train_didi = DIDI.classification_didi(indicator_matrix=indicator_matrix, targets=y)
        else:
            didi = self.backend.sum(deviations)
            train_didi = DIDI.regression_didi(indicator_matrix=indicator_matrix, targets=y)
        self.backend.add_constraint(didi <= self.violation * train_didi)
        return variables

    def adjust_targets(self, x, y, p, sample_weight=None):
        z = super(DidiMaster, self).adjust_targets(x, y, p, sample_weight)
        # print(z.dtype)
        # if not np.allclose(z, z.round()):
        #     for i, v in enumerate(z):
        #         print(i, '-->', v)
        return z


class LrMaster(FairnessMaster):
    def __init__(self, classification, backend, loss, alpha, stats, protected='race', theta=1.0):
        super(LrMaster, self).__init__(classification, backend, loss, alpha, stats, protected)
        self.theta = theta

    def build(self, x, y, p):
        v = super(LrMaster, self).build(x, y, p)
        f = [c for c in x.columns if c.startswith(self.protected)]
        x = np.concatenate((np.ones((len(x), 1)), x[f]), axis=1)
        v_weights = self.backend.add_continuous_variables(x.shape[1], name='w')
        y_weights = np.abs(np.linalg.lstsq(x, y, rcond=None)[0])
        left_hand_sides = np.atleast_1d(self.backend.dot(x.T @ x, v_weights))
        right_hand_sides = np.atleast_1d(self.backend.dot(x.T, v))
        self.backend.add_constraints(
            [lrl == lrr for lrl, lrr in zip(left_hand_sides, right_hand_sides)] +
            [vw >= -self.theta * yw for vw, yw in zip(v_weights[1:], y_weights[1:])] +
            [vw <= self.theta * yw for vw, yw in zip(v_weights[1:], y_weights[1:])]
        )

        # self.w = v_weights / y_weights

        return v

    def adjust_targets(self, x, y, p, sample_weight=None):
        z = super(LrMaster, self).adjust_targets(x, y, p, sample_weight)
        return probabilities.get_classes(z) if self.classification else z
