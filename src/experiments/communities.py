import importlib.resources
from typing import Tuple

import numpy as np
import pandas as pd

from moving_targets.util.scalers import Scaler
from src.experiments.experiment import Experiment


class Communities(Experiment):
    @staticmethod
    def load_data(scale: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        if scale:
            return Scaler('std', race='none').fit_transform(x), Scaler('norm').fit_transform(y)
        else:
            return x, y

    def __init__(self, continuous: bool):
        """
        :param continuous:
            Whether the excluded feature is binary or continuous.
        """
        super(Communities, self).__init__(
            name='communities ' + 'continuous' if continuous else 'categorical',
            excluded='pctBlack' if continuous else 'race',
            continuous=continuous,
            classification=False,
            units=[256, 256]
        )
