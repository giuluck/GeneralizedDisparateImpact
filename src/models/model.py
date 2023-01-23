from typing import Any, Dict

import numpy as np
import pandas as pd


class Model:
    def __init__(self, name: str, **config):
        """
        :param name:
            The model name.

        :param config:
            A dictionary of parameters that represent the configuration of the model.
        """

        self.__name__: str = name
        """The model name."""

        self.is_fit: bool = False
        """Whether or not the model has been fitted."""

        self.config: Dict[str, Any] = {'type': self.__class__.__name__}
        """A dictionary of parameters that represent the configuration of the model."""

        for k, v in config.items():
            self.config[k] = 'None' if v is None else v

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        """Fits the model.

        :param x:
            THe input data.

        :param y:
            The output target.
        """
        raise NotImplementedError("Please implement method '_fit'")

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predicts the target given an input dataset.

        :param x:
            The input data.

        :return:
            The predicted targets.
        """
        raise NotImplementedError("Please implement method '_predict'")

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        """Fits the model.

        :param x:
            The input data.

        :param y:
            The output target.
        """
        assert not self.is_fit, "Model has been already fitted"
        self._fit(x, y)
        self.is_fit = True

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predicts the target given an input dataset.

        :param x:
            The input data.

        :return:
            The predicted targets.
        """
        assert self.is_fit, "Model has not been fitted yet"
        return self._predict(x)
