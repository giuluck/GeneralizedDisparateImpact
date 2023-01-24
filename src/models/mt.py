from typing import Dict, Union, List, Optional, Any

import numpy as np
import pandas as pd
from torch import nn

from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.learners import LinearRegression, LogisticRegression, RandomForestClassifier, \
    RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, TorchMLP
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import MSE, HammingDistance
from moving_targets.masters.optimizers import Harmonic
from moving_targets.metrics import Metric
from moving_targets.util.typing import Dataset
from src.metrics import GeneralizedDIDI as gDIDI
from src.models.model import Model


class AbstractMaster(Master):
    """Master problem for causal exclusion of features up to a certain polynomial degree."""

    def __init__(self, classification: bool, excluded: str, degree: int, threshold: float, relative: Union[bool, int]):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The feature to exclude.

        :param degree:
            The kernel degree up to which to exclude the feature.

        :param threshold:
            The threshold up to which to exclude the feature.

        :param relative:
            If a positive integer k is passed, it computes the relative value with respect to the indicator computed on
            the original targets with kernel k. If True is passed, it assumes k = 1. Otherwise, if False is passed, it
            simply computes the absolute value of the indicator.
        """
        assert degree > 0 and isinstance(degree, int), f"Degree should be a positive integer number, got {degree}"
        assert threshold >= 0.0, f"Threshold should be a non-negative number, got {threshold}"

        # handle binary vs continuous
        lb, ub, vtype = (0, 1, 'binary') if classification else (-float('inf'), float('inf'), 'continuous')

        self.classification: bool = classification
        """Whether we are dealing with a binary classification or a regression task."""

        self.excluded: str = excluded
        """The feature to exclude."""

        self.degree: int = degree
        """The kernel degree up to which to exclude the feature."""

        self.threshold: float = threshold
        """The threshold up to which to exclude the feature."""

        self.relative: int = int(relative) if isinstance(relative, bool) else relative
        """The kernel degree to use to compute the metric in relative value, or 0 for absolute value."""

        self.lb: float = lb
        """The model variables lower bounds."""

        self.ub: float = ub
        """The model variables upper bounds."""

        self.vtype: str = vtype
        """The model variables vtypes."""

        super().__init__(backend=GurobiBackend(WorkLimit=60),
                         loss=HammingDistance() if classification else MSE(),
                         alpha=Harmonic(1.0))

    def _formulation(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, v: np.ndarray):
        """Implements the formulation of the constraint.

        :param x:
            The vector representing the feature to be excluded.

        :param y:
            The vector of original targets.

        :param p:
            The vector of model predictions.

        :param v:
            The vector of variables of the solver.
        """
        raise NotImplementedError("Please implement abstract method '_formulation'")

    def build(self, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        assert y.ndim == 1, f"Target vector must be one-dimensional, got shape {y.shape}"
        v = self.backend.add_variables(len(y), vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')
        # avoid degenerate case (all zeros or all ones) in classification scenario
        if self.classification:
            sum_v = self.backend.sum(v)
            self.backend.add_constraints([sum_v >= 1, sum_v <= len(v) - 1])
        # build the model according to the implemented formulation and return the variables
        self._formulation(x=x[self.excluded].values, y=y, p=p, v=v)
        return v

    def adjust_targets(self,
                       x,
                       y: np.ndarray,
                       p: Optional[np.ndarray],
                       sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        # due to numerical tolerances, targets may be returned as <z + eps>, therefore we round binary targets in order
        # to remove these numerical errors and make sure that they will not cause problems to the learners
        z = super(AbstractMaster, self).adjust_targets(x, y, p)
        return z.round() if self.classification else z


class GeneralizedDIDIMaster(AbstractMaster):
    """Master formulation for the standard generalized DIDI constraint relying on a shadow linear regression model."""

    def _formulation(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, v: np.ndarray):
        # now we compute the values tilde{alpha} respectively to the adjusted targets; in order to do that, we define
        # a vector <alpha> of free variables which are computed by posing the linear system equations as constraints
        phi = np.stack([x ** d - (x ** d).mean() for d in np.arange(self.degree) + 1], axis=1)
        psi = v - self.backend.mean(v, aux=True)
        alpha = self.backend.add_continuous_variables(self.degree)
        left_hand_sides = np.atleast_1d(self.backend.dot(phi.T @ phi, alpha))
        right_hand_sides = np.atleast_1d(self.backend.dot(phi.T, psi))
        self.backend.add_constraints([lhs == rhs for lhs, rhs in zip(left_hand_sides, right_hand_sides)])
        # eventually, our indicator is given by the norm 1 of the alpha vector, optionally constrained with respect to
        # the indicator computed on the original targets with degree k in case relative is a positive integer k, i.e.:
        #    didi_v <= threshold * didi(x, y, kernel=relative), if relative > 0
        #    didi_v <= threshold = threshold * 1, if relative == 0
        didi_v = self.backend.norm_1(alpha)
        didi_y = gDIDI.generalized_didi(x, y, degree=self.relative) if self.relative > 0 else 1
        self.backend.add_constraints([didi_v <= self.threshold * didi_y])


class FirstOrderMaster(AbstractMaster):
    """Master formulation where the constraint on generalized DIDI is obtained by excluding the higher orders and
    keeping just the first-order correlation. This allows to impose easier constraints without the need to compute the
    actual slopes of the shadow linear regression model.
    """

    def _formulation(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, v: np.ndarray):
        # compute the generalized didi for the adjusted targets value by assuming that tilde{alpha}_i = 0 for each
        # i != 1; the value of the didi becomes the value of tilde{alpha}_1, which can be computed from the first
        # equation of the usual linear system as:
        #   var(x) alpha_1 = cov(x, v) --> alpha_1 = cov(x, v) / var(x) --> didi_v = | cov(x, v) | / var(x)
        # and since we might optionally want to constraint our indicator with respect to the indicator computed on the
        # original targets with degree k in case relative is a positive integer k, we get that:
        #    didi_v <= threshold * didi(x, y, kernel=relative), if relative > 0
        #    didi_v <= threshold = threshold * 1, if relative == 0
        # which translates to:
        #    | cov(x, v) | <= threshold * var(x) * didi(x, y, kernel=relative), if relative > 0
        #    | cov(x, v) | <= threshold * var(x) * 1, if relative == 0
        var_x = x.var()
        cov_xv = self.backend.mean(x * v) - x.mean() * self.backend.mean(v)
        didi_y = gDIDI.generalized_didi(x, y, degree=self.relative) if self.relative > 0 else 1
        threshold = self.threshold * didi_y * var_x
        self.backend.add_constraints([cov_xv >= -threshold, cov_xv <= threshold])
        # additionally, we have to add further constraints to ensure that the higher orders have null weight, i.e.:
        #    cov(x^d, x) * cov(x, v) == var(x) * cov(x^d, v)
        # for each d in {2, ..., k}
        for d in np.arange(1, self.degree) + 1:
            xd = x ** d
            cov_xdx = np.cov(xd, x, bias=True)[0, 1]
            cov_xdv = self.backend.mean(xd * v) - xd.mean() * self.backend.mean(v)
            self.backend.add_constraint(cov_xv * cov_xdx == cov_xdv * var_x)


class MovingTargets(Model):
    def __init__(self,
                 master: str,
                 learner: str,
                 classification: bool,
                 excluded: str,
                 degree: int,
                 threshold: float,
                 relative: Union[bool, int],
                 iterations: int = 10,
                 metrics: List[Metric] = (),
                 fold: Optional[Dataset] = None,
                 callbacks: List[Callback] = (),
                 verbose: Union[bool, int] = False,
                 history: Union[bool, Dict[str, Any]] = False,
                 **learner_kwargs):
        """
        :param master:
            The master alias, either 'coarse' for regular GeDI or 'fine' for fine-grained GeDI on first order.

        :param learner:
            The learner alias, one in 'lr', 'rf', 'gb', and 'nn'.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The feature to exclude.

        :param degree:
            The kernel degree up to which to exclude the feature.

        :param threshold:
            The threshold up to which to exclude the feature.

        :param relative:
            If a positive integer k is passed, it computes the relative value with respect to the indicator computed on
            the original targets with kernel k. If True is passed, it assumes k = 1. Otherwise, if False is passed, it
            simply computes the absolute value of the indicator.

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
        if learner == 'lr':
            lrn = LogisticRegression(max_iter=10000) if classification else LinearRegression()
        elif learner == 'rf':
            lrn = RandomForestClassifier() if classification else RandomForestRegressor()
        elif learner == 'gb':
            lrn = GradientBoostingClassifier() if classification else GradientBoostingRegressor()
        elif learner == 'nn':
            lrn = TorchMLP(
                loss=nn.BCELoss() if classification else nn.MSELoss(),
                activation=nn.Sigmoid() if classification else None,
                verbose=False,
                **learner_kwargs
            )
        else:
            raise AssertionError(f"Unknown learner alias '{learner}'")

        if master == 'coarse':
            mst = GeneralizedDIDIMaster
        elif master == 'fine':
            mst = FirstOrderMaster
        else:
            raise AssertionError(f"Unknown master alias '{master}'")
        mst = mst(classification, excluded, degree=degree, threshold=threshold, relative=relative)

        super(MovingTargets, self).__init__(
            name=f'mt {learner} {master}',
            classification=classification,
            excluded=excluded,
            degree=degree,
            threshold=threshold,
            relative=relative,
            iterations=iterations,
            learner=lrn.__class__.__name__,
            master=mst.__class__.__name__,
            **learner_kwargs
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
