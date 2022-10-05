"""Moving Targets Losses."""
from typing import Optional, Any, Callable

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util import probabilities
from moving_targets.util.errors import not_implemented_message


class Loss:
    """Basic interface for a Moving Targets Master Loss."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the loss.
        """

        self.__name__: str = name
        """The name of the loss."""

    def __call__(self,
                 backend: Backend,
                 numeric_variables: np.ndarray,
                 model_variables: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None) -> Any:
        """Core method used to compute the master loss.

        :param backend:
            The `Backend` instance.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the model predictions).

        :param sample_weight:
            The sample weights associated to each sample.

        :return:
            The loss expression.
        """
        losses = self._losses(backend=backend, numeric_variables=numeric_variables, model_variables=model_variables)
        if sample_weight is not None:
            # normalize weights then multiply partial losses per respective weight
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
            losses = sample_weight * losses
        return backend.mean(losses)

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        """Computes the partial losses computed over the pairs of variables.

        :param backend:
            The `Backend` instance.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the model predictions).

        :return:
            The array of partial losses computed over the pairs of variables.
        """
        raise NotImplementedError(not_implemented_message(name='_losses'))


class MAE(Loss):
    """Mean Absolute Error Loss for Univariate Targets."""

    def __init__(self, binary: bool = False, name: str = 'mean_absolute_error'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

        :param name:
            The name of the loss.
        """
        super(MAE, self).__init__(name=name)

        # given the model variables <m> and the numeric variables <n>, if <m> is binary we have that:
        #     abs(m, n) = m * (1 - n) + (1 - m) * n = m - 2mn + n
        self._abs: Callable = (lambda b, m, n: m - 2 * m * n + n) if binary else (lambda b, m, n: b.abs(m - n))

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        return self._abs(backend, model_variables, numeric_variables)


class MSE(Loss):
    """Mean Squared Error Loss for Univariate Targets."""

    def __init__(self, binary: bool = False, name: str = 'mean_squared_error'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

        :param name:
            The name of the loss.
        """
        super(MSE, self).__init__(name=name)

        # given the model variables <m> and the numeric variables <n>, if <m> is binary we have that:
        #     sqr(m, n) = [m * (1 - n) + (1 - m) * n]^2 =
        #               = [m * (1 - n)]^2        + [(1 - m) * n]^2	    - 2 * [m * (1 - n) * (1 - m) * n] =
        #               = [m^2 * (1 - 2n + n^2)] + [n^2 * (1 - 2m + m^2)] - 2 * [m * (1 - m) * (1 - n) * n] =
        #               = [m - 2mn + mn^2] 	   + [n^2 - 2mn^2 + mn^2]   - 2 * [(m - m) * (n - n^2)] = --> as m^2 = m
        #               = [m - 2mn + mn^2]       + [n^2 - mn^2]           - 2 * [0] =
        #               = m - 2mn + n^2
        self._sqr: Callable = (lambda b, m, n: m - 2 * m * n + n ** 2) if binary else (lambda b, m, n: b.square(m - n))

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        return self._sqr(backend, model_variables, numeric_variables)


class HammingDistance(Loss):
    """Hamming Distance for Binary Targets."""

    def __init__(self, name: str = 'hamming_distance'):
        """
        :param name:
            The name of the loss.
        """
        super(HammingDistance, self).__init__(name=name)

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        # the hamming distance is computed on class values, not on probabilities
        numeric_variables = probabilities.get_classes(numeric_variables)
        first_term = backend.multiply(1 - numeric_variables, model_variables)
        second_term = backend.multiply(numeric_variables, 1 - model_variables)
        return first_term + second_term


class CrossEntropy(Loss):
    """Negative Log-Likelihood Loss for Binary Targets."""

    def __init__(self, clip_value: float = 1e-15, name: str = 'crossentropy'):
        """
        :param clip_value:
            The clipping value to be used to avoid numerical errors.

        :param name:
            The name of the metric.
        """
        super(CrossEntropy, self).__init__(name=name)

        self.clip_value: float = clip_value
        """The clipping value to be used to avoid numerical errors."""

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        numeric_variables = numeric_variables.clip(min=self.clip_value, max=1 - self.clip_value)
        first_term = backend.multiply(model_variables, -np.log(numeric_variables))
        second_term = backend.multiply(1 - model_variables, -np.log(1 - numeric_variables))
        return first_term + second_term


aliases: dict = {
    # Mean Absolute Error
    'mae': MAE,
    'mean_absolute_error': MAE,
    'mean absolute error': MAE,
    # Mean Squared Error
    'mse': MSE,
    'mean_squared_error': MSE,
    'mean squared error': MSE,
    # Hamming Distance
    'hd': HammingDistance,
    'hamming_distance': HammingDistance,
    'hamming distance': HammingDistance,
    'bhd': HammingDistance,
    'binary_hamming': HammingDistance,
    'binary hamming': HammingDistance,
    'chd': HammingDistance,
    'categorical_hamming': HammingDistance,
    'categorical hamming': HammingDistance,
    # CrossEntropy
    'ce': CrossEntropy,
    'crossentropy': CrossEntropy,
    'bce': CrossEntropy,
    'binary_crossentropy': CrossEntropy,
    'binary crossentropy': CrossEntropy,
    'cce': CrossEntropy,
    'categorical_crossentropy': CrossEntropy,
    'categorical crossentropy': CrossEntropy,
    'll': CrossEntropy,
    'log_likelihood': CrossEntropy,
    'log likelihood': CrossEntropy,
    'nll': CrossEntropy,
    'negative_log_likelihood': CrossEntropy,
    'negative log likelihood': CrossEntropy,
}
"""Dictionary which associates to each loss alias the respective class type."""
