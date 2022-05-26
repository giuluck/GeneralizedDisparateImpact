from typing import Dict, Union, List, Callable, Optional

import numpy as np
from moving_targets.masters import Master
from moving_targets.masters.backends import Backend
from moving_targets.masters.losses import Loss, HammingDistance, aliases
from moving_targets.masters.optimizers import Optimizer
from sklearn.preprocessing import PolynomialFeatures

from src.constraints import Constraint


class Shape:
    def __init__(self,
                 feature: str,
                 constraints: Union[Dict[int, Constraint], List[Optional[Constraint]]],
                 kernel: Union[None, int, Callable] = 1):
        """
        :param feature:
            The name of the feature having the desired shape.

        :param constraints:
            Either a list of either `Constraint` objects (or None for no constraint) or a dictionary of such objects
            indexed via the kernel index (e.g., if the kernel is f(x) -> [1 | x | x^2], the dictionary {2: Null()} will
            only impose the constraint x^2 == 0).

        :param kernel:
            If None is passed, the input feature is not transformed. If an integer is passed, the input feature is
            transformed via a polynomial kernel up to the given degree, with bias included (the default behaviour is,
            indeed, to transform feature x into [1|x]). Otherwise, an explicit kernel function must be passed.
        """
        # handle constraints
        # - if a set of constrains is passed as a list, transforms it into a dictionary for compatibility
        if isinstance(constraints, list):
            constraints = {i: c for i, c in enumerate(constraints) if c is not None}

        # handle default kernel function
        # - if no kernel, uses the identity function (i.e., simply retrieves the feature)
        # - if int, builds a polynomial kernel and applies it
        if kernel is None:
            kernel = lambda x: x
        elif isinstance(kernel, int):
            kernel = PolynomialFeatures(degree=kernel, include_bias=True).fit_transform

        self.feature: str = feature
        """The feature name."""

        self.constraints: Dict[int, Constraint] = constraints
        """The dictionary of constraints indexed via the kernel index."""

        self.kernel: Callable = kernel
        """A callable kernel function that transforms the input feature."""


class ShapeConstrainedMaster(Master):
    """Common Interface for Shape Constrained Master."""

    def __init__(self,
                 backend: Union[str, Backend],
                 binary: bool = False,
                 loss: Union[str, Loss] = 'auto',
                 alpha: Union[str, float, Optimizer] = 'harmonic',
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param binary:
            Whether the output variables should be binary (0/1 for binary classification tasks) or continuous (ranging
            from -infinity to +infinity).

        :param loss:
            Either a string representing a `Loss` alias or the actual `Loss` instance used to compute the objective.

        :param alpha:
            Either a floating point for a constant alpha value, a string representing an `Optimizer` alias,  or an
            actual `Optimizer` instance which implements the strategy to dynamically change the alpha value.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'nabla_term', 'squared_term', 'objective', 'elapsed_time'] whose statistics must be logged.
        """

        # handle binary vs continuous
        lb, ub, vtype = (0, 1, 'binary') if binary else (-float('inf'), float('inf'), 'continuous')

        self.binary: bool = binary
        """Whether the output variables should be binary or continuous."""

        self.lb = lb
        """The model variables lower bounds."""

        self.ub = ub
        """The model variables upper bounds."""

        self.vtype = vtype
        """The model variables vtypes."""

        super().__init__(backend=backend, loss=loss, alpha=alpha, stats=stats)

    def _build(self, x, y, p, v):
        raise NotImplementedError("Please implement abstract method _build")

    def _get_loss(self, loss):
        loss = ('hd' if self.binary else 'mse') if loss == 'auto' else loss
        loss_class = aliases.get(loss)
        assert loss_class is not None, f"Unknown loss '{loss}'"
        if loss_class == HammingDistance:
            assert self.binary, "HammingDistance cannot deal with continuous variables"
            return loss_class(labelling=False)
        else:
            return loss_class(binary=self.binary)

    # noinspection PyPep8Naming
    def build(self, x, y, p):
        assert y.ndim == 1, f"Target vector must be one-dimensional, got shape {y.shape}"
        v = self.backend.add_variables(len(y), vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')
        self._build(x, y, p, v)
        return v

    def adjust_targets(self, x, y, p, sample_weight=None):
        # due to numerical tolerances, targets may be returned as <z + eps>, therefore we round binary targets in order
        # to remove these numerical errors and make sure that they will not cause problems to the learners
        z = super(ShapeConstrainedMaster, self).adjust_targets(x, y, p)
        return z.round() if self.binary else z


class DefaultMaster(ShapeConstrainedMaster):
    """Handle partial (soft) shape constraints in regression or binary classification tasks by building a surrogate
    linear regression model and forcing its weights to be in certain user-specified intervals.

    The model can deal with multiple kind of constraints (e.g., positive/negative, greater/lower than, within a certain
    interval or outside that interval, ...), which allows to model different properties ranging from monotonicity,
    convexity/concavity, sensitivity, independence, and so on. Moreover, the constraints can be imposed (independently)
    on multiple features, and on different transformations of each feature via the kernel function.

    For example, let us suppose that we have two input features (x1, x2) and we want to constraint the first feature
    so to have a symmetric concave response stronger than a certain threshold (e.g., 0.2) with respect to the target y.
    Then we should pass `[Shape(feature='x1', constraints=[None, Null(), Greater(0.2)], kernel=2)]` a shape parameter,
    so that the kernel function will transform feature 'x1' into A = [1 | x1 | x1^2] and the master will eventually
    compute the weights for the linear regression model constraint its weights accordingly, i.e., no constraint on the
    intercept, no influence from the linear term ("x1 == 0") and concavity stronger than the threshold ("x1^2 >= 0.2").
    """

    def __init__(self,
                 shapes: List[Shape],
                 backend: Union[str, Backend],
                 binary: bool = False,
                 loss: Union[str, Loss] = 'auto',
                 alpha: Union[str, float, Optimizer] = 'harmonic',
                 stats: Union[bool, List[str]] = False):
        self.shapes: List[Shape] = shapes
        """The list of desired shapes for input features."""

        super().__init__(backend=backend, binary=binary, loss=loss, alpha=alpha, stats=stats)

    # noinspection PyPep8Naming
    def _build(self, x, y, p, v):
        for shape in self.shapes:
            # build linear regression model: A.T @ A @ w = A.T @ v
            A = shape.kernel(x[[shape.feature]])
            w = self.backend.add_continuous_variables(A.shape[1])
            left_hand_sides = np.atleast_1d(self.backend.dot(A.T @ A, w))
            right_hand_sides = np.atleast_1d(self.backend.dot(A.T, v))
            self.backend.add_constraints([lhs == rhs for lhs, rhs in zip(left_hand_sides, right_hand_sides)])
            # constraint (transformed) features f with respective shapes s
            for i, c in shape.constraints.items():
                c.add(expression=w[i], backend=self.backend)


class ExplicitZerosMaster(ShapeConstrainedMaster):
    """Optimized master for shape constraints involving the causal exclusion of a single features up to a certain
    polynomial degree that the higher degrees to zero in order to limit the numerical errors."""

    def __init__(self,
                 feature: str,
                 constraint: Constraint,
                 backend: Union[str, Backend],
                 binary: bool = False,
                 degree: int = 1,
                 loss: Union[str, Loss] = 'auto',
                 alpha: Union[str, float, Optimizer] = 'harmonic',
                 stats: Union[bool, List[str]] = False):
        # check data entry
        assert degree > 0, f"'degree' should be a positive integer, got {degree}"

        self.feature: str = feature
        """The feature on which to impose the constraints."""

        self.constraint: Constraint = constraint
        """The constraint to enforce on the first order."""

        self.degree: int = degree
        """The degree of the polynomial kernel."""

        super().__init__(backend=backend, binary=binary, loss=loss, alpha=alpha, stats=stats)

    # noinspection PyPep8Naming
    def _build(self, x, y, p, v):
        A = PolynomialFeatures(degree=self.degree).fit_transform(x[[self.feature]])
        w = self.backend.add_continuous_variables(2)
        w = np.concatenate((w, [0] * (self.degree - 1)))
        left_hand_sides = np.atleast_1d(self.backend.dot(A.T @ A, w))
        right_hand_sides = np.atleast_1d(self.backend.dot(A.T, v))
        self.constraint.add(expression=w[1], backend=self.backend)
        self.backend.add_constraints([lhs == rhs for lhs, rhs in zip(left_hand_sides, right_hand_sides)])


class CovarianceBasedMaster(ShapeConstrainedMaster):
    """Optimized master for shape constraints involving the causal exclusion of a single features up to a certain
    polynomial degree that relies on the explicit formulation (involving variances and covariances) in order to limit
    the numerical errors."""

    def __init__(self,
                 feature: str,
                 constraint: Constraint,
                 backend: Union[str, Backend],
                 binary: bool = False,
                 degree: int = 1,
                 loss: Union[str, Loss] = 'auto',
                 alpha: Union[str, float, Optimizer] = 'harmonic',
                 stats: Union[bool, List[str]] = False):
        # check data entry
        assert degree > 0, f"'degree' should be a positive integer, got {degree}"

        self.feature: str = feature
        """The feature on which to impose the constraints."""

        self.constraint: Constraint = constraint
        """The constraint to enforce on the first order."""

        self.degree: int = degree
        """The degree of the polynomial kernel."""

        super().__init__(backend=backend, binary=binary, loss=loss, alpha=alpha, stats=stats)

    def _build(self, x, y, p, v):
        x = x[self.feature].values
        thetas = []
        for d in np.arange(self.degree) + 1:
            xd = x ** d
            cov_xd = np.cov(x, xd, bias=True)[0, 1]
            cov_xy = self.backend.mean(xd * v) - xd.mean() * self.backend.mean(v)
            thetas.append(cov_xy / cov_xd)
        self.constraint.add(expression=thetas[0], backend=self.backend)
        self.backend.add_constraints([t == thetas[0] for t in thetas[1:]])
