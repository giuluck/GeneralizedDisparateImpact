from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

from moving_targets.util.typing import Dataset


class Model:
    def __init__(self, name: str, **config):
        """
        :param name:
            The model name.

        :param config:
            A dictionary of parameters that represent the configuration of the model.
        """

        self.is_fit: bool = False
        """Whether or not the model has been fitted."""

        self.config: Dict[str, Any] = {'model': name, **config}
        """A dictionary of parameters that represent the configuration of the model."""

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
