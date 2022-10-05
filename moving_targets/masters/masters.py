"""Basic Master Interface."""
import time
from abc import ABC
from typing import Set, Optional, Tuple
from typing import Union, List

import numpy as np

from moving_targets.callbacks import StatsLogger
from moving_targets.masters import optimizers, losses, backends
from moving_targets.masters.backends import Backend
from moving_targets.masters.losses import MAE, MSE, Loss
from moving_targets.masters.optimizers import Constant, Optimizer
from moving_targets.util import probabilities
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.scalers import Scaler
from moving_targets.util.typing import Dataset


class Master(StatsLogger):
    """Basic interface for a Moving Targets Master."""

    _Loss = Union[str, Loss]
    """Type alias."""

    @staticmethod
    def _parameters() -> Set[str]:
        return {'alpha', 'nabla_term', 'squared_term', 'objective', 'elapsed_time'}

    def __init__(self,
                 backend: Union[str, Backend],
                 loss: Union[_Loss, Tuple[_Loss, _Loss]],
                 alpha: Union[str, float, Optimizer],
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param loss:
            Either a single loss element for both y_loss and p_loss, or two separate loss elements. Each loss element
            can be a string representing a `Loss` alias or the actual `Loss` instance used to compute the objective.

        :param alpha:
            Either a floating point for a constant alpha value, a string representing an `Optimizer` alias,  or an
            actual `Optimizer` instance which implements the strategy to dynamically change the alpha value.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
           'nabla_term', 'squared_term', 'objective', 'elapsed_time'] whose statistics must be logged.
        """

        super(Master, self).__init__(stats=stats, name='Master')

        # handle alpha optimizer
        if isinstance(alpha, str):
            alpha = self._get_optimizer(optimizer=alpha)
        elif not isinstance(alpha, Optimizer):
            alpha = Constant(base=alpha)

        # losses
        y_loss, p_loss = loss if isinstance(loss, tuple) else (loss, loss)

        self.backend: Backend = backend if isinstance(backend, Backend) else self._get_backend(backend=backend)
        """The `Backend` instance."""

        self.y_loss: Loss = y_loss if isinstance(y_loss, Loss) else self._get_loss(loss=y_loss)
        """The `Loss` instance for the targets."""

        self.p_loss: Loss = p_loss if isinstance(p_loss, Loss) else self._get_loss(loss=p_loss)
        """The `Loss` instance for the predictions."""

        self.alpha: Optimizer = alpha
        """The alpha `Optimizer` instance."""

        self.x_scaler: Optional[Scaler] = Scaler(default_method=x_scaler) if isinstance(x_scaler, str) else x_scaler
        """The (optional) scaler for the input data."""

        self.y_scaler: Optional[Scaler] = Scaler(default_method=y_scaler) if isinstance(y_scaler, str) else y_scaler
        """The (optional) scaler for the output data."""

        self._macs: Optional = None
        """Reference to the MACS object encapsulating the `Master`."""

        self._time: Optional[float] = None
        """An auxiliary variable to keep track of the elapsed time between iterations."""

    # noinspection PyMethodMayBeStatic
    def _get_backend(self, backend: str) -> Backend:
        """Implements the strategy to get default backends from the class type retrieved using the backend alias.

        :param backend:
            The backend alias.

        :return:
            The correct `Optimizer` instance.
        """
        backend_class = backends.aliases.get(backend)
        assert backend_class is not None, f"Unknown backend alias '{backend}'"
        return backend_class()

    def _get_loss(self, loss: str) -> Loss:
        """Implements the strategy to get default losses from the class type retrieved using the loss alias.

        :param loss:
            The loss alias.

        :return:
            The correct `Loss` instance.
        """
        loss_class = losses.aliases.get(loss)
        assert loss_class is not None, f"Unknown loss alias '{loss}'"
        return loss_class(name=loss)

    # noinspection PyMethodMayBeStatic
    def _get_optimizer(self, optimizer: str) -> Optimizer:
        """Implements the strategy to get default optimizers from the class type retrieved using the optimizer alias.

        :param optimizer:
            The optimizer alias.

        :return:
            The correct `Optimizer` instance.
        """
        optimizer_class = optimizers.aliases.get(optimizer)
        assert optimizer_class is not None, f"Unknown alpha optimizer alias '{optimizer}'"
        return optimizer_class(base=1)

    def log(self, **cache):
        self._macs.log(**cache)

    def on_adjustment_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._macs = macs
        self._time = time.time()

    def on_adjustment_end(self, macs, x, y: np.ndarray, z: np.ndarray, val_data: Optional[Dataset]):
        self._log_stats(elapsed_time=time.time() - self._time)
        self._time = None
        self._macs = None

    def on_backend_solved(self, x, y: np.ndarray, p: np.ndarray, z: np.ndarray):
        """Routine called after the master problem is solved and the adjusted targets are retrieved.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param p:
            The vector of learners predictions.

        :param z:
            The vector of adjusted labels.
        """
        pass

    def build(self, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Creates the model variables and adds the problem constraints.

        :param x:
            The matrix/dataframe of data samples.

        :param y:
            The vector of original labels.

        :param p:
            The vector of learner predictions.

        :return:
            The model variables.
        """
        raise NotImplementedError(not_implemented_message(name='build'))

    def adjust_targets(self,
                       x,
                       y: np.ndarray,
                       p: Optional[np.ndarray],
                       sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Core function of the `Master` object which builds the model and returns the adjusted targets.

        :param x:
            The matrix/dataframe of data samples.

        :param y:
            The vector of original labels.

        :param p:
            The (optional) vector of learner predictions, which will be None in the first iteration with 'projection'.

        :param sample_weight:
            The (optional) vector of sample weights.

        :return:
            The vector of adjusted targets.
        """
        # use the original targets when no predictions are available due to the initial projection step
        p = y if p is None else p
        # transform data if scalers are passed
        x = x if self.x_scaler is None else self.x_scaler.fit_transform(data=x)
        if self.y_scaler is not None:
            y = self.y_scaler.transform(data=y)
            p = self.y_scaler.transform(data=p)
        # build backend and model
        self.backend.build()
        v = self.build(x=x, y=y, p=p)
        alpha = self.alpha(x=x, y=y, p=p)
        y_loss = self.y_loss(backend=self.backend, model_variables=v, numeric_variables=y, sample_weight=sample_weight)
        p_loss = self.p_loss(backend=self.backend, model_variables=v, numeric_variables=p, sample_weight=sample_weight)
        self.backend.minimize(cost=alpha * y_loss + p_loss)
        # if the problem is infeasible return None, otherwise log stats and return the adjusted labels
        if self.backend.solve().solution is not None:
            self._log_stats(alpha=alpha,
                            y_loss=self.backend.get_value(y_loss),
                            p_loss=self.backend.get_value(p_loss),
                            objective=self.backend.get_objective())
            adjusted = self.backend.get_values(expressions=v)
        else:
            self._log_stats(alpha=alpha)
            adjusted = None
        self.on_backend_solved(x=x, y=y, p=p, z=adjusted)
        self.backend.clear()
        # invert transformation on output targets if scaler is passed
        return adjusted if self.y_scaler is None else self.y_scaler.inverse_transform(adjusted)


class RegressionMaster(Master, ABC):
    """An abstract Moving Targets Master for Regression Tasks."""

    def __init__(self, backend: Union[str, Backend],
                 loss: Union[str, Loss] = 'mse',
                 alpha: Union[str, float, Optimizer] = 'harmonic',
                 lb: float = -float('inf'),
                 ub: float = float('inf'),
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param loss:
            Either a string representing a `Loss` alias or the actual `Loss` instance used to compute the objective.

        :param alpha:
            Either a floating point for a constant alpha value, a string representing an `Optimizer` alias,  or an
            actual `Optimizer` instance which implements the strategy to dynamically change the alpha value.

        :param lb:
            The model variables lower bounds.

        :param ub:
            The model variables upper bounds.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
            'nabla_term', 'squared_term', 'objective', 'elapsed_time'] whose statistics must be logged.
        """
        super(RegressionMaster, self).__init__(backend=backend,
                                               loss=loss,
                                               alpha=alpha,
                                               x_scaler=x_scaler,
                                               y_scaler=y_scaler,
                                               stats=stats)

        self.lb = lb
        """The model variables lower bounds."""

        self.ub = ub
        """The model variables upper bounds."""

    def _get_loss(self, loss: str) -> Loss:
        loss_class = losses.aliases.get(loss)
        assert loss_class is not None, f"Unknown loss alias '{loss}'"
        assert loss_class in [MAE, MSE], f"Loss '{loss} is not a valid regression loss."
        return loss_class(name=loss)

    def build(self, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')


class ClassificationMaster(Master, ABC):
    """An abstract Moving Targets Master for Classification Tasks."""

    def __init__(self,
                 backend: Union[str, Backend],
                 loss: Union[str, Loss] = 'crossentropy',
                 alpha: Union[str, float, Optimizer] = 'harmonic',
                 types: str = 'auto',
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False):
        """
        :param backend:
            The `Backend` instance encapsulating the optimization solver.

        :param loss:
            Either a string representing a `Loss` alias or the actual `Loss` instance used to compute the objective.

        :param alpha:
            Either a floating point for a constant alpha value, a string representing an `Optimizer` alias,  or an
            actual `Optimizer` instance which implements the strategy to dynamically change the alpha value.

        :param types:
            The variables and adjustments types, which must be in ['discrete', 'discretized', 'continuous']:
            - if 'discrete' is chosen, the model will use binary variables to represent class targets, and the returned
            adjustments will be discrete;
            - if 'discretized' is chosen, the model will use continuous variables to represent class probabilities, but
            the returned adjustments will be discretized in order to match the same return type of 'discrete';
            - if 'continuous' is chosen, the model will use continuous variables to represent class probabilities, and
            the returned adjustments will be continuous as well.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters in ['alpha',
           'nabla_term', 'squared_term', 'objective', 'elapsed_time'] whose statistics must be logged.
        """
        if types == 'discrete':
            binary, classes = True, True
        elif types == 'discretized':
            binary, classes = False, True
        elif types == 'continuous':
            binary, classes = False, False
        else:
            raise AssertionError("'vtype' should be one in 'discrete', 'discretized', or 'continuous', got '{vtype}'")

        self.binary: bool = binary
        """Whether to use binary or continuous model variables."""

        self.classes: bool = classes
        """Whether to return class targets or class probabilities."""

        super(ClassificationMaster, self).__init__(backend=backend,
                                                   loss=loss,
                                                   alpha=alpha,
                                                   x_scaler=x_scaler,
                                                   y_scaler=y_scaler,
                                                   stats=stats)

    def _get_loss(self, loss: str) -> Loss:
        loss_class = losses.aliases.get(loss)
        assert loss_class is not None, f"Unknown loss alias '{loss}'"
        # check for continuous/binary targets in case of regression losses
        if loss_class in [MAE, MSE]:
            return loss_class(binary=self.binary, name=loss)
        else:
            return loss_class(name=loss)

    def build(self, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # we can simply build variables by shape (the y targets will be already onehot encoded for multiclass tasks)
        vtype = 'binary' if self.binary else 'continuous'
        variables = self.backend.add_variables(*y.shape, vtype=vtype, lb=0, ub=1, name='y')
        return variables

    def adjust_targets(self,
                       x,
                       y: np.ndarray,
                       p: Optional[np.ndarray],
                       sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        # due to numerical tolerances, targets may be returned as <z + eps>, therefore we round binary targets in order
        # to remove these numerical errors and make sure that they will not cause problems to the learners
        z = super(ClassificationMaster, self).adjust_targets(x, y, p, sample_weight)
        z = z.round() if self.binary else z
        # eventually, we return class/label targets if we are asked so, or class/label probabilities otherwise
        return probabilities.get_classes(z) if self.classes else z
