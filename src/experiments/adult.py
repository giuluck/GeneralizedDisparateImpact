import importlib.resources
from typing import Tuple

import numpy as np
import pandas as pd

from moving_targets.util.scalers import Scaler
from src.experiments.experiment import Experiment


class Adult(Experiment):
    @staticmethod
    def load_data(scale: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        with importlib.resources.path('data', 'adult.csv') as filepath:
            df = pd.read_csv(filepath)
        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        if scale:
            return Scaler('none', **{f: 'std' for f in numerical_features}).fit_transform(x), y
        else:
            return x, y

    def __init__(self, continuous: bool):
        """
        :param continuous:
            Whether the excluded feature is binary or continuous.
        """
        super(Adult, self).__init__(
            excluded='age' if continuous else 'sex',
            continuous=continuous,
            classification=True,
            units=[32, 32, 32]
        )
