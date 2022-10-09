import importlib.resources
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from moving_targets.metrics import DIDI, Metric
from moving_targets.util.scalers import Scaler
from src.experiments.experiment import Experiment
from src.metrics import BinnedDIDI
from src.models.model import Model
from src.models.mt import MT
from src.models.sbr import SBR


class Communities(Experiment):
    THRESHOLD: float = 0.2

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        return Scaler('std', race='none').fit_transform(x), Scaler('norm').fit_transform(y)

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
            # handle tasks-specific default arguments
            learner = kwargs.get('learner') or 'lr'
            metrics = kwargs.get('metrics') or self.metrics
            return MT(classification=False,
                      excluded=self.excluded,
                      thresholds=self.THRESHOLD,
                      learner=learner,
                      metrics=metrics,
                      **kwargs)
        elif model == 'sbr':
            return SBR(classification=False, excluded=self.excluded, threshold=self.THRESHOLD, **kwargs)
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
            # handle tasks-specific default degree for continuous fairness scenarios
            kwargs['degrees'] = kwargs.get('degrees') or 3
        return super(CommunitiesIncome, self).get_model(model, **kwargs)
