import importlib.resources
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from moving_targets.metrics import DIDI, Metric
from moving_targets.util.scalers import Scaler
from src.experiments.experiment import Experiment
from src.metrics import BinnedDIDI
from src.models import Model


class Communities(Experiment):
    classification = False

    @staticmethod
    def load_data(scale: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        if scale:
            return Scaler('std', race='none').fit_transform(x), Scaler('norm').fit_transform(y)
        else:
            return x, y

    def __init__(self, excluded: Union[str, List[str]], metrics: List[Metric]):
        """
        :param excluded:
            Either a single feature or the list of features to exclude.

        :param metrics:
            The list of task-specific evaluation metrics.
        """
        super(Communities, self).__init__(metrics=metrics,
                                          excluded=excluded,
                                          threshold=0.2,
                                          units=[256, 256])


class CommunitiesCategorical(Communities):
    def __init__(self):
        """"""
        super(CommunitiesCategorical, self).__init__(excluded='race', metrics=[
            DIDI(protected='race', classification=self.classification, percentage=True, name='rel_didi'),
            DIDI(protected='race', classification=self.classification, percentage=False, name='abs_didi')
        ])


class CommunitiesContinuous(Communities):
    def __init__(self, bins: Tuple[int] = (2, 3, 5, 10)):
        """Communities & Crime dataset with 'pctBlack' as protected feature.

        :param bins:
            The number of bins to be used in the BinnedDIDI metric.
        """
        super(CommunitiesContinuous, self).__init__(excluded='pctBlack', metrics=[
            *[BinnedDIDI(
                bins=b,
                protected='pctBlack',
                classification=self.classification,
                percentage=True,
                name='rel_didi'
            ) for b in bins],
            *[BinnedDIDI(
                bins=b,
                protected='pctBlack',
                classification=self.classification,
                percentage=False,
                name='abs_didi'
            ) for b in bins]
        ])

    def get_model(self, model: str, **kwargs) -> Model:
        # handle tasks-specific default degree for continuous fairness scenarios
        if model == 'sbr cov' or model.startswith('mt '):
            kwargs['degrees'] = kwargs.get('degrees') or 3
        return super(CommunitiesContinuous, self).get_model(model, **kwargs)
