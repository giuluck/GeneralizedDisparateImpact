import importlib.resources
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from moving_targets.metrics import DIDI, Metric
from src.experiments.experiment import Experiment
from src.metrics import BinnedDIDI
from src.models.model import Model
from src.models.mt import MT


class Adult(Experiment):
    THRESHOLD: float = 0.2

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
        with importlib.resources.path('data', 'adult.csv') as filepath:
            df = pd.read_csv(filepath)
        x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        x = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)
        return x, y

    def __init__(self, excluded: Union[str, List[str]], metrics: List[Metric]):
        """
        :param excluded:
            Either a single feature or the list of features to exclude.

        :param metrics:
            The list of task-specific evaluation metrics.
        """
        super(Adult, self).__init__(excluded=excluded, classification=True, metrics=metrics)

    def get_model(self, model: str, **kwargs) -> Model:
        if model == 'mt':
            # default arguments for moving targets model which might be overridden by custom kwargs
            args = dict(
                degrees=1,
                learner='lr',
                iterations=15,
                metrics=self.metrics
            )
            args.update(kwargs)
            return MT(classification=True, features=self.excluded, thresholds=self.THRESHOLD, **args)
        else:
            raise AssertionError(f"Unknown model alias '{model}'")


class AdultRace(Adult):
    def __init__(self):
        """"""
        metrics = [DIDI(protected='race', classification=True)]
        categories = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
        super(AdultRace, self).__init__(excluded=[f'race_{r}' for r in categories], metrics=metrics)


class AdultAge(Adult):
    def __init__(self, bins: Tuple[int] = (2, 3, 5, 10)):
        """Adult dataset with 'age' as protected feature.

        :param bins:
            The number of bins to be used in the BinnedDIDI metric.
        """
        metrics = [BinnedDIDI(bins=b, protected='age', classification=True) for b in bins]
        super(AdultAge, self).__init__(excluded='age', metrics=metrics)

    def get_model(self, model: str, **kwargs) -> Model:
        if model == 'mt':
            # when using moving targets in this scenario the default argument for the degree is 3
            kwargs['degrees'] = kwargs.get('degrees') or 3
        return super(AdultAge, self).get_model(model, **kwargs)
