import importlib.resources
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from moving_targets.metrics import DIDI, Metric
from src.experiments.experiment import Experiment
from src.metrics import BinnedDIDI
from src.models.model import Model
from src.models.mt import MT


class Communities(Experiment):
    THRESHOLD: float = 0.2

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df[['violentPerPop']].values
        x = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)
        y = MinMaxScaler().fit_transform(y).flatten()
        return x, y

    def __init__(self, excluded: Union[str, List[str]], metrics: List[Metric]):
        """
        :param excluded:
            Either a single feature or the list of features to exclude.

        :param metrics:
            The list of task-specific evaluation metrics.
        """
        super(Communities, self).__init__(excluded=excluded, classification=False, metrics=metrics)

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
            return MT(classification=False, features=self.excluded, thresholds=self.THRESHOLD, **args)
        else:
            raise AssertionError(f"Unknown model alias '{model}'")


class CommunitiesRace(Communities):
    def __init__(self):
        """"""
        metrics = [DIDI(protected='race', classification=False)]
        super(CommunitiesRace, self).__init__(excluded='race', metrics=metrics)


class CommunitiesIncome(Communities):
    def __init__(self, bins: Tuple[int] = (2, 3, 5, 10)):
        """Communities & Crime dataset with 'income' as protected feature.

        :param bins:
            The number of bins to be used in the BinnedDIDI metric.
        """
        metrics = [BinnedDIDI(bins=b, protected='perCapInc', classification=False) for b in bins]
        super(CommunitiesIncome, self).__init__(excluded='perCapInc', metrics=metrics)

    def get_model(self, model: str, **kwargs) -> Model:
        if model == 'mt':
            # when using moving targets in this scenario the default argument for the degree is 3
            kwargs['degrees'] = kwargs.get('degrees') or 3
        return super(CommunitiesIncome, self).get_model(model, **kwargs)
