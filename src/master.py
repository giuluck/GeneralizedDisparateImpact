from typing import Dict, Union, List, Optional

import numpy as np

from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import MSE, HammingDistance
from moving_targets.masters.optimizers import Harmonic


class CausalExclusionMaster(Master):
    """Custom master problem for causal exclusion of features up to a certain polynomial degree. In order to limit the
    numerical errors, instead of explicitly building the regression model, this implementation relies on the explicit
    mathematical formulation involving variances and covariances."""

    def __init__(self,
                 classification: bool,
                 features: Union[str, List[str]],
                 thresholds: Union[float, List[float]] = 0.0,
                 degrees: Union[int, List[int]] = 1,
                 time_limit: float = 30,
                 stats: Union[bool, List[str]] = False):
        """
        :param features:
            The features to exclude.

        :param thresholds:
            Either a common threshold or a list of thresholds for each of the features to exclude.

        :param degrees:
            Either a common kernel degree or a list of kernel degrees for each of the features to exclude.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param time_limit:
            Time limit for the Gurobi backend.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'nabla_term', 'squared_term', 'objective', 'elapsed_time'] whose statistics must be logged.

        """
        # handle binary vs continuous
        lb, ub, vtype = (0, 1, 'binary') if classification else (-float('inf'), float('inf'), 'continuous')

        # handle features
        if isinstance(features, list):
            n = len(features)
            degrees = degrees if isinstance(degrees, list) else [degrees] * n
            thresholds = thresholds if isinstance(thresholds, list) else [thresholds] * n
            assert len(degrees) == n, f"There are {n} features, but {len(degrees)} degrees were passed"
            assert len(thresholds) == n, f"There are {n} features, but {len(thresholds)} thresholds were passed"
        else:
            assert not isinstance(degrees, list), "If features is a single value, degrees cannot be a list"
            assert not isinstance(thresholds, list), "If features is a single value, thresholds cannot be a list"
            features, degrees, thresholds = [features], [degrees], [thresholds]

        self.features: Dict[str, (float, int)] = {f: (t, d) for f, t, d in zip(features, thresholds, degrees)}
        """The list of features to exclude with the respective exclusion threshold and kernel degree."""

        self.classification: bool = classification
        """Whether we are dealing with a binary classification or a regression task."""

        self.lb: float = lb
        """The model variables lower bounds."""

        self.ub: float = ub
        """The model variables upper bounds."""

        self.vtype: str = vtype
        """The model variables vtypes."""

        super().__init__(backend=GurobiBackend(time_limit=time_limit),
                         loss=HammingDistance() if classification else MSE(),
                         alpha=Harmonic(1.0),
                         stats=stats)

    def build(self,
              x,
              y: np.ndarray,
              p: np.ndarray) -> np.ndarray:
        assert y.ndim == 1, f"Target vector must be one-dimensional, got shape {y.shape}"
        v = self.backend.add_variables(len(y), vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')
        for feature, (threshold, degree) in self.features.items():
            # retrieve the column vector of the feature to exclude
            z = x[feature].values
            # compute the first-order slope of the shadow linear regressor and constraint it wrt the threshold
            cov_zy = self.backend.mean(z * v) - z.mean() * self.backend.mean(v)
            var_z = np.var(z)
            theta = cov_zy / var_z
            self.backend.add_constraints([theta >= -threshold, theta <= threshold])
            # compute the higher-order slopes of the shadow regressor and constraint them according to the formulation
            for d in np.arange(1, degree) + 1:
                zd = z ** d
                cov_zdz = np.cov(zd, z, bias=True)[0, 1]
                cov_zdy = self.backend.mean(zd * v) - zd.mean() * self.backend.mean(v)
                # instead of forcing:
                #   cov(z, y) / var(z) = theta_1 == theta_d = cov(z^d, y) / cov(z^d, z)
                # we use the equivalent constraint:
                #   cov(z, y) * cov(z^d, z) == cov(z^d, y) * var(z)
                # which is less subject to numerical error since there is no fraction
                self.backend.add_constraint(cov_zy * cov_zdz == cov_zdy * var_z)
        return v

    def adjust_targets(self,
                       x,
                       y: np.ndarray,
                       p: Optional[np.ndarray],
                       sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        # due to numerical tolerances, targets may be returned as <z + eps>, therefore we round binary targets in order
        # to remove these numerical errors and make sure that they will not cause problems to the learners
        z = super(CausalExclusionMaster, self).adjust_targets(x, y, p)
        return z.round() if self.classification else z
