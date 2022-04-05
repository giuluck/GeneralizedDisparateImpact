from typing import Dict, Union, List, Callable, Any, Optional

import numpy as np
from moving_targets.masters import Master
from moving_targets.masters.backends import Backend
from moving_targets.masters.losses import Loss, HammingDistance, aliases
from moving_targets.masters.optimizers import Optimizer
from sklearn.preprocessing import PolynomialFeatures

from src.constraints import Constraint


class ShapeConstrainedMaster(Master):
    """Handle partial (soft) shape constraints in regression or binary classification tasks by building a surrogate
    linear regression model and forcing its weights to be in certain user-specified intervals.

    The model can deal with multiple kind of constraints (e.g., positive/negative, greater/lower than, within a certain
    interval or outside that interval, ...), which allows to model different properties ranging from monotonicity,
    convexity/concavity, sensitivity, independence, and so on. Moreover, the constraints can be imposed (independently)
    on multiple features, and on different transformations of each feature via the kernel function.

    For example, let us suppose that we have two input features (x1, x2) and we want to constraint the first feature
    so that its concavity stronger than a certain threshold (e.g., 0.2) with respect to the target y. Then we should
    build a kernel function f(x, feature) which transforms the feature x into A = [1 | x | x^2] and eventually build
    the master with constraints {'x1': [None, Null(), Shape.Greater(0.2)]}. The master will indeed compute the weights
    for the linear regression model constraint its weights accordingly, i.e., the it no constraint one the intercept,
    but the monotonicity should have no influence while the concavity should be greater than the threshold.
    """

    def __init__(self,
                 backend: Union[str, Backend],
                 constraints: Dict[Any, Union[Dict[int, Constraint], List[Optional[Constraint]]]],
                 kernels: Union[None, int, Callable] = 1,
                 binary: bool = False,
                 loss: Union[str, Loss] = 'mse',
                 alpha: Union[str, float, Optimizer] = 'harmonic',
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param constraints:
            A dictionary which associates to each constrained feature either a list of either `Constraint` objects (or
            None for no constraint) or a dictionary of such indexed via feature number, which are used to constraint
            the i-th weight of the linear regression model built on the kernel transformation of that feature.

        :param kernels:
            If None is passed, the input feature is not transformed. If an integer is passed, the input feature is
            transformed via a polynomial kernel up to the given degree, with bias included (the default behaviour is,
            indeed, to transform feature f into [1|f]). Otherwise, an explicit kernel function must be passed.

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
        # handle constraints
        # - if a set of constrains is passed as a list, transforms it into a dictionary for compatibility
        for feature, constraint in constraints.items():
            if isinstance(constraint, list):
                constraints[feature] = {i: c for i, c in enumerate(constraint) if c is not None}

        # handle default kernel function
        # - if no kernel, uses the identity function (i.e., simply retrieves the feature)
        # - if int, builds a polynomial kernel and applies it
        if kernels is None:
            kernels = lambda x, f: x[[f]]
        elif isinstance(kernels, int):
            kernel = PolynomialFeatures(degree=kernels, include_bias=True)
            kernels = lambda x, f: kernel.fit_transform(x[[f]])

        # handle binary vs continuous
        lb, ub, vtype = (0, 1, 'binary') if binary else (-float('inf'), float('inf'), 'continuous')

        self.constraints: Dict[Any, Dict[int, Constraint]] = constraints
        """The shapes of each constrained feature respectively to the transformation."""

        self.kernels: Callable = kernels
        """The kernel function to transform each constrained feature."""

        self.binary: bool = binary
        """Whether the output variables should be binary or continuous."""

        self.lb = lb
        """The model variables lower bounds."""

        self.ub = ub
        """The model variables upper bounds."""

        self.vtype = vtype
        """The model variables vtypes."""

        super().__init__(backend=backend, loss=loss, alpha=alpha, stats=stats)

    def _get_loss(self, loss):
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
        for feature, constraint in self.constraints.items():
            # build linear regression model: A.T @ A @ w = A.T @ v
            A = self.kernels(x, feature)
            w = self.backend.add_continuous_variables(A.shape[1])
            left_hand_sides = np.atleast_1d(self.backend.dot(A.T @ A, w))
            right_hand_sides = np.atleast_1d(self.backend.dot(A.T, v))
            self.backend.add_constraints([lhs == rhs for lhs, rhs in zip(left_hand_sides, right_hand_sides)])
            # constraint (transformed) features f with respective shapes s
            for i, c in constraint.items():
                c.add(expression=w[i], backend=self.backend)
        return v
