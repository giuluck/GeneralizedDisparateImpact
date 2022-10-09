import random
import re
import time
from typing import List, Tuple, Union
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from moving_targets.metrics import Metric, MSE, R2, CrossEntropy, Accuracy
from moving_targets.util.typing import Dataset
from src.models.model import Model


class Experiment:
    SEED: int = 0
    ENTITY: str = 'giuluck'
    PROJECT: str = 'non_causal_exclusion'

    @staticmethod
    def setup(seed: int):
        """Sets the random seed of the experiment.

        :param seed:
            The random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
        """Loads the dataset.

        :return:
            A tuple (x, y) containing the input data and the target vector.
        """
        raise NotImplementedError("please implement static method 'load_data'")

    def __init__(self, excluded: Union[str, List[str]], classification: bool, metrics: List[Metric]):
        """
        :param excluded:
            Either a single feature or the list of features to exclude.

        :param classification:
            Whether this is a classification or a regression task.

        :param metrics:
            The list of task-specific evaluation metrics.
        """

        task_metrics = [Accuracy(), CrossEntropy()] if classification else [R2(), MSE()]

        self.__name__: str = '_'.join(re.split('(?=[A-Z])', self.__class__.__name__)).lower()
        """The dataset name."""

        self.data: Tuple[pd.DataFrame, np.ndarray] = self.load_data()
        """The tuple (x, y) containing the input data and the target vector."""

        self.excluded: List[str] = excluded if isinstance(excluded, list) else [excluded]
        """The list of features whose causal effect should be excluded."""

        self.classification: bool = classification
        """Whether this is a classification or a regression task."""

        self.metrics: List[Metric] = [*task_metrics, *metrics]
        """The list of evaluation metrics."""

    def get_model(self, model: str, **kwargs) -> Model:
        """Returns a model instance.

        :param model:
            The model alias.

        :param kwargs:
            The model custom arguments.

        :return:
            The model instance.
        """
        raise NotImplementedError("please implement method 'get_model'")

    def get_folds(self, folds: Optional[int] = None, seed: int = 0) -> Union[List[Dataset], Dataset]:
        """Gets the data split in folds.

        With folds = None returns a dictionary of type {'train': (x, y)}.
        With folds = 1 returns a dictionary of type {'train': (xtr, ytr), 'test': (xts, yts)}, with 0.3 test split.
        With folds > 1 returns a list of dictionaries of type {'train': (xtr, ytr), 'val': (xvl, yvl)}.

        :param folds:
            The number of folds for k-fold cross-validation.

        :param seed:
            The splitting random seed.

        :return:
            Either a single tuple, a pair of tuples, or a list of tuples.
        """
        x, y = self.data
        if folds is None:
            return {'train': (x, y)}
        elif folds == 1:
            stratify = y if self.classification else None
            xtr, xts, ytr, yts = train_test_split(x, y, test_size=0.3, stratify=stratify, random_state=seed)
            return {'train': (xtr, ytr), 'test': (xts, yts)}
        else:
            kf = StratifiedKFold if self.classification else KFold
            idx = kf(n_splits=folds, shuffle=True, random_state=seed).split(x, y)
            return [{'train': (x.iloc[tr], y[tr]), 'val': (x.iloc[ts], y[ts])} for tr, ts in idx]

    def evaluate(self, model: Model, fold: Dataset) -> pd.DataFrame:
        """Evaluates the model on the given fold based on the task-specific metrics.

        :param model:
            The model to evaluate.

        :param fold:
            The fold on which to evaluate the model.

        :return:
            A panda Dataframe with metrics values as entries per each dataset in the fold.
        """
        metrics = {}
        for split, (x, y) in fold.items():
            p = model.predict(x)
            metrics[split] = {m.__name__: m(x, y, p) for m in self.metrics}
        return pd.DataFrame.from_dict(data=metrics)

    def run(self, model: str, folds: Optional[int], show: bool = True, log: bool = True, **kwargs):
        """Runs the experiment.

        :param model:
            The model alias.

        :param folds:
            The number of folds for cross-validation (folds = None means training set only, folds = 1 means 70/30
            train/test split, folds > 1 performs actual k-fold cross-validation).

        :param show:
            Whether or not to show the results on the console at the end of each instance run.

        :param log:
            Whether or not to log the results of each instance run on Weights & Biases.

        :param kwargs:
            The model custom arguments.
        """
        # if there is a single fold, run the experiment with fold index = None, otherwise indicate the correct index
        folds = self.get_folds(folds=folds, seed=self.SEED)
        if isinstance(folds, dict):
            x, y = folds['train']
            mdl = self.get_model(model, **kwargs)
            self.run_instance(x=x, y=y, model=mdl, fold=folds, index=None, show=show, log=log)
        else:
            for idx, fold in enumerate(folds):
                x, y = fold['train']
                mdl = self.get_model(model, **kwargs)
                self.run_instance(x=x, y=y, model=mdl, fold=fold, index=idx, show=show, log=log)

    def run_instance(self,
                     x: pd.DataFrame,
                     y: np.ndarray,
                     model: Model,
                     fold: Dataset,
                     index: Optional[int],
                     show: bool,
                     log: bool):
        """Runs a single instance of k-fold cross-validation experiment.

        :param x:
            The input data.

        :param y:
            The output targets.

        :param model:
            The machine learning model.

        :param fold:
            The fold on which to evaluate the model.

        :param index:
            The (optional) fold index.

        :param show:
            Whether or not to show the results on the console at the end of each instance run.

        :param log:
            Whether or not to log the results of each instance run on Weights & Biases.

        """
        # LOGGING & PRINTING
        if log:
            wandb.init(name=self.__name__,
                       entity=self.ENTITY,
                       project=self.PROJECT,
                       config={'fold': index, **model.config})
        # EXPERIMENT RUN
        self.setup(seed=self.SEED)
        start_time = time.time()
        model.fit(x=x, y=y)
        elapsed_time = time.time() - start_time
        metrics = self.evaluate(model=model, fold=fold)
        # LOGGING & PRINTING
        if log:
            logs = {'elapsed_time': elapsed_time}
            for split in metrics.columns:
                logs += {f'{split}/{metric}' for metric, value in metrics[split].items()}
            wandb.log(logs)
            wandb.finish()
        if show:
            if index is None:
                print('RESULTS:')
                print(metrics)
            else:
                print('-------------------------------------------------------')
                print(f'FOLD {index}:')
                print(metrics)
                print('-------------------------------------------------------')
