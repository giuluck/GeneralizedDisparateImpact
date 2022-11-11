from typing import Dict, Union, List, Optional, Any

import numpy as np
import pandas as pd
from torch import nn

from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.learners import LinearRegression, LogisticRegression, RandomForestClassifier, \
    RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from moving_targets.learners.torch_learners import TorchMLP
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import MSE, HammingDistance
from moving_targets.masters.optimizers import Harmonic
from moving_targets.metrics import Metric
from moving_targets.util.typing import Dataset
from src.models.model import Model


class CausalExclusionMaster(Master):
    """Custom master problem for causal exclusion of features up to a certain polynomial degree. In order to limit the
    numerical errors, instead of explicitly building the regression model, this implementation relies on the explicit
    mathematical formulation involving variances and covariances."""

    def __init__(self,
                 classification: bool,
                 features: Union[str, List[str]],
                 degrees: Union[int, List[int]] = 1,
                 thresholds: Union[float, List[float]] = 0.0,
                 relative_thresholds: bool = True):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param features:
            The features to exclude.

        :param degrees:
            Either a common kernel degree or a list of kernel degrees for each of the features to exclude.

        :param thresholds:
            Either a common threshold or a list of thresholds for each of the features to exclude.

        :param relative_thresholds:
            Whether the thresholds have to be considered as absolute values or a percentage with respect to the weights
            computed on the training data using a linear kernel (i.e., the reference value is cov(z, y) / var(z)).
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

        self.relative_thresholds: bool = relative_thresholds
        """Whether the thresholds have to be considered as absolute values or a percentage with respect to the weights
        computed on the training data using a linear kernel."""

        self.lb: float = lb
        """The model variables lower bounds."""

        self.ub: float = ub
        """The model variables upper bounds."""

        self.vtype: str = vtype
        """The model variables vtypes."""

        super().__init__(backend=GurobiBackend(time_limit=30),
                         loss=HammingDistance() if classification else MSE(),
                         alpha=Harmonic(1.0))

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
            # if relative thresholds are used, we need to compute a "relative" theta which is obtained by dividing
            # theta for the respective value it has on the training data, i.e.:
            #   training_theta = cov(z, y) / var(z)
            # the value of the "relative" theta will then be:
            #   relative_theta = theta / training_theta = cov(z, v) / var(z) * var(z) / cov(z, y)
            # which means that we ca simply compute our theta as:
            #   <absolute> theta = cov(z, v) / var(z)
            #   <relative> theta = cov(z, v) / cov(z, y)
            # in the end, since our constraint is:
            #   abs(theta) <= threshold
            # we can rewrite it as:
            #   abs(cov(z, v) / <denominator>) <= threshold
            # where <denominator> is either var(z) or cov(z, y) depending on the requirements, thus:
            #   abs(cov(z, v)) / abs(<denominator>) <= threshold
            # which becomes:
            #   abs(cov(z, v)) <= threshold * abs(<denominator>)
            # which can be eventually split into:
            #   -threshold * abs(<denominator>) <= cov(z, v) <= threshold * abs(<denominator>)
            denominator = np.cov(z, y, bias=True)[0, 1] if self.relative_thresholds else var_z
            threshold *= abs(denominator)
            self.backend.add_constraints([cov_zy >= -threshold, cov_zy <= threshold])
            # compute the higher-order slopes of the shadow regressor and constraint them according to the formulation
            for d in np.arange(1, degree) + 1:
                zd = z ** d
                cov_zdz = np.cov(zd, z, bias=True)[0, 1]
                cov_zdy = self.backend.mean(zd * v) - zd.mean() * self.backend.mean(v)
                # instead of forcing:
                #   cov(z, y) / var(z) = theta_1 == theta_d = cov(z^d, y) / cov(z^d, z)
                # we use the equivalent constraint:
                #   cov(z, y) * cov(z^d, z) == cov(z^d, y) * var(z)
                # which is less subject to numerical errors since there is no fraction
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


class MovingTargets(Model):
    def __init__(self,
                 learner: str,
                 classification: bool,
                 excluded: Union[str, List[str]],
                 thresholds: Union[float, List[float]] = 0.0,
                 degrees: Union[int, List[int]] = 1,
                 iterations: int = 10,
                 metrics: List[Metric] = (),
                 fold: Optional[Dataset] = None,
                 callbacks: List[Callback] = (),
                 verbose: Union[bool, int] = False,
                 history: Union[bool, Dict[str, Any]] = False,
                 **learner_kwargs):
        """
        :param learner:
            The learner alias.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The features to be excluded.

        :param thresholds:
            Either a common threshold or a list of thresholds for each of the features to be excluded.

        :param degrees:
            Either a common kernel degree or a list of kernel degrees for each of the features to be excluded.

        :param iterations:
            The number of Moving Targets' iterations.

        :param metrics:
            The Moving Targets' metrics.

        :param fold:
            An (optional) validation fold.

        :param callbacks:
            The Moving Targets' callbacks.

        :param verbose:
            The Moving Targets' verbosity level.

        :param history:
            Either a boolean value representing whether or not to plot the Moving Targets' history or a dictionary of
            parameters to pass to the History's plot function.

        :param learner_kwargs:
            Additional arguments to be passed to the Learner constructor.
        """
        super(MovingTargets, self).__init__(
            name=f'mt {learner}',
            classification=classification,
            excluded=excluded,
            thresholds=thresholds,
            degrees=degrees,
            iterations=iterations,
            **learner_kwargs
        )

        if learner == 'lr':
            lrn = LogisticRegression(max_iter=10000) if classification else LinearRegression()
        elif learner == 'rf':
            lrn = RandomForestClassifier() if classification else RandomForestRegressor()
        elif learner == 'gb':
            lrn = GradientBoostingClassifier() if classification else GradientBoostingRegressor()
        elif learner == 'nn':
            lrn = TorchMLP(
                loss=nn.BCELoss() if classification else nn.MSELoss(),
                output_activation=nn.Sigmoid() if classification else None,
                verbose=False,
                **learner_kwargs
            )
        else:
            raise AssertionError(f"Unknown learner alias '{learner}'")

        mst = CausalExclusionMaster(
            classification=classification,
            thresholds=thresholds,
            features=excluded,
            degrees=degrees
        )

        self.macs: MACS = MACS(init_step='pretraining', learner=lrn, master=mst, metrics=metrics)
        """The MACS instance."""

        self.fit_params: Dict[str, Any] = {
            'iterations': iterations,
            'callbacks': list(callbacks),
            'verbose': verbose,
            'val_data': fold
        }
        """The MACS fitting parameters."""

        self.history: Union[bool, Dict[str, Any]] = history
        """Either a boolean value representing whether or not to plot the Moving Targets' history or a dictionary of
        parameters to pass to the History's plot function."""

    def add_callback(self, callback: Callback):
        self.fit_params['callbacks'].append(callback)

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        history = self.macs.fit(x, y, **self.fit_params)
        if isinstance(self.history, dict):
            history.plot(**self.history)
        elif self.history is True:
            history.plot()

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.macs.predict(x)
