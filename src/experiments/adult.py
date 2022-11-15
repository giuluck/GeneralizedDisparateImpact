import importlib.resources
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from moving_targets.metrics import DIDI, Metric
from moving_targets.util.scalers import Scaler
from src.experiments.experiment import Experiment
from src.metrics import BinnedDIDI
from src.models import Model


class Adult(Experiment):

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
        with importlib.resources.path('data', 'adult.csv') as filepath:
            df = pd.read_csv(filepath)
        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        return Scaler('none', **{f: 'std' for f in numerical_features}).fit_transform(x), y

    def __init__(self, excluded: Union[str, List[str]], metrics: List[Metric]):
        """
        :param excluded:
            Either a single feature or the list of features to exclude.

        :param metrics:
            The list of task-specific evaluation metrics.
        """
        super(Adult, self).__init__(classification=True,
                                    metrics=metrics,
                                    excluded=excluded,
                                    threshold=0.2,
                                    units=[32, 32, 32])


class AdultCategorical(Adult):
    def __init__(self):
        """"""
        categories = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
        super(AdultCategorical, self).__init__(excluded=[f'race_{r}' for r in categories], metrics=[
            DIDI(protected='race', classification=True, percentage=True, name='rel_didi'),
            DIDI(protected='race', classification=True, percentage=False, name='abs_didi')
        ])


class AdultContinuous(Adult):
    def __init__(self, bins: Tuple[int] = (2, 3, 5, 10)):
        """Adult dataset with 'age' as protected feature.

        :param bins:
            The number of bins to be used in the BinnedDIDI metric.
        """
        super(AdultContinuous, self).__init__(excluded='age', metrics=[
            *[BinnedDIDI(bins=b, protected='age', classification=True, percentage=True, name='rel_didi') for b in bins],
            *[BinnedDIDI(bins=b, protected='age', classification=True, percentage=False, name='abs_didi') for b in bins]
        ])

    def get_model(self, model: str, **kwargs) -> Model:
        # handle tasks-specific default degree for continuous fairness scenarios
        if model == 'sbr cov' or model.startswith('mt '):
            kwargs['degrees'] = kwargs.get('degrees') or 3
        return super(AdultContinuous, self).get_model(model, **kwargs)
