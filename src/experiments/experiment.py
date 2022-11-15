import random
import re
import time
from typing import List, Tuple, Union
from typing import Optional

import numpy as np
import pandas as pd
import torch.random
import wandb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from moving_targets.metrics import Metric, MSE, R2, CrossEntropy, Accuracy
from moving_targets.util.typing import Dataset
from src.metrics import HGR, RegressionWeight
from src.models import Model, RandomForest, GradientBoosting, NeuralNetwork, MovingTargets, NeuralSBR


class Experiment:
    SEED: int = 0
    """The random seed."""

    ENTITY: str = 'giuluck'
    """The Weights&Biases entity name."""

    @staticmethod
    def setup(seed: int):
        """Sets the random seed of the experiment.

        :param seed:
            The random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, np.ndarray]:
        """Loads the dataset.

        :return:
            A tuple (x, y) containing the input data and the target vector.
        """
        raise NotImplementedError("please implement static method 'load_data'")

    def __init__(self,
                 classification: bool,
                 metrics: List[Metric],
                 excluded: Union[str, List[str]],
                 threshold: float,
                 units: List[int]):
        """
        :param classification:
            Whether this is a classification or a regression task.

        :param metrics:
            The list of task-specific evaluation metrics.

        :param excluded:
            Either a single feature or the list of features to exclude.

        :param threshold:
            The threshold for the feature to exclude.

        :param units:
            The neural networks default units.
        """

        task_metrics = [Accuracy(), CrossEntropy()] if classification else [R2(), MSE()]

        self.__name__: str = ' '.join(re.split('(?=[A-Z])', self.__class__.__name__)).lower().strip(' ')
        """The dataset name."""

        self.data: Tuple[pd.DataFrame, np.ndarray] = self.load_data()
        """The tuple (x, y) containing the input data and the target vector."""

        self.classification: bool = classification
        """Whether this is a classification or a regression task."""

        self.excluded: List[str] = excluded if isinstance(excluded, list) else [excluded]
        """The list of features whose causal effect should be excluded."""

        self.threshold: float = threshold
        """The threshold for the feature to exclude."""

        self.units: List[int] = units
        """The neural networks default units."""

        self.metrics: List[Metric] = [
            *task_metrics,
            *metrics,
            HGR(features=excluded, percentage=True, chi2=False, name='rel_hgr'),
            HGR(features=excluded, percentage=False, chi2=False, name='abs_hgr'),
            HGR(features=excluded, percentage=True, chi2=True, name='rel_chi2'),
            HGR(features=excluded, percentage=False, chi2=True, name='abs_chi2'),
            *[RegressionWeight(feature=f, degree=5, percentage=True, name=f'rel_{f}') for f in self.excluded],
            *[RegressionWeight(feature=f, degree=5, percentage=False, name=f'abs_{f}') for f in self.excluded]
        ]
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
        if model == 'rf':
            return RandomForest(classification=self.classification, **kwargs)
        elif model == 'gb':
            return GradientBoosting(classification=self.classification, **kwargs)
        elif model == 'nn':
            kwargs['hidden_units'] = kwargs.get('hidden_units') or self.units
            kwargs['batch_size'] = kwargs.get('batch_size') or 128
            kwargs['epochs'] = kwargs.get('epochs') or 200
            return NeuralNetwork(classification=self.classification, **kwargs)
        elif model == 'sbr hgr':
            kwargs['threshold'] = kwargs.get('threshold') or self.threshold
            kwargs['hidden_units'] = kwargs.get('hidden_units') or self.units
            kwargs['batch_size'] = kwargs.get('batch_size') or 128
            kwargs['epochs'] = kwargs.get('epochs') or 200
            return NeuralSBR(penalty='hgr', excluded=self.excluded, classification=self.classification, **kwargs)
        elif model == 'sbr cov':
            kwargs['threshold'] = kwargs.get('threshold') or self.threshold
            kwargs['hidden_units'] = kwargs.get('hidden_units') or self.units
            kwargs['epochs'] = kwargs.get('epochs') or 500
            return NeuralSBR(
                penalty='cov',
                excluded=self.excluded,
                classification=self.classification,
                batch_size=len(self.data[0]),
                **kwargs
            )
        elif model.startswith('mt '):
            learner = model[3:]
            if learner == 'nn':
                kwargs['hidden_units'] = kwargs.get('hidden_units') or self.units
                kwargs['batch_size'] = kwargs.get('batch_size') or len(self.data[0])
                kwargs['epochs'] = kwargs.get('epochs') or 500
            kwargs['threshold'] = kwargs.get('threshold') or self.threshold
            return MovingTargets(learner=learner, classification=self.classification, excluded=self.excluded, **kwargs)
        else:
            raise AssertionError(f"Unknown model alias '{model}'")

    def get_folds(self, folds: Optional[int] = None) -> Union[List[Dataset], Dataset]:
        """Gets the data split in folds.

        With folds = None returns a dictionary of type {'train': (x, y)}.
        With folds = 1 returns a dictionary of type {'train': (xtr, ytr), 'test': (xts, yts)}, with 0.3 test split.
        With folds > 1 returns a list of dictionaries of type {'train': (xtr, ytr), 'val': (xvl, yvl)}.

        :param folds:
            The number of folds for k-fold cross-validation.

        :return:
            Either a single tuple, a pair of tuples, or a list of tuples.
        """
        x, y = self.data
        if folds is None:
            return {'train': (x, y)}
        elif folds == 1:
            stratify = y if self.classification else None
            xtr, xts, ytr, yts = train_test_split(x, y, test_size=0.3, stratify=stratify, random_state=self.SEED)
            return {'train': (xtr, ytr), 'test': (xts, yts)}
        else:
            kf = StratifiedKFold if self.classification else KFold
            idx = kf(n_splits=folds, shuffle=True, random_state=self.SEED).split(x, y)
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
            results = {}
            for metric in self.metrics:
                value = metric(x, y, p)
                if isinstance(value, (int, float, np.number)):
                    results[f'{metric.__name__}'] = value
                elif isinstance(value, dict):
                    for k, v in value.items():
                        results[f'{metric.__name__}_{k}'] = v
            metrics[split] = results
        return pd.DataFrame.from_dict(data=metrics)

    def run(self, model: str, folds: Optional[int], show: bool = True, log: Optional[str] = None, **kwargs):
        """Runs the experiment.

        :param model:
            The model alias.

        :param folds:
            The number of folds for cross-validation (folds = None means training set only, folds = 1 means 70/30
            train/test split, folds > 1 performs actual k-fold cross-validation).

        :param show:
            Whether or not to show the results on the console at the end of each instance run.

        :param log:
            Either a Weights & Biases project name on which to log results or None for no logging.

        :param kwargs:
            The model custom arguments.
        """
        # if there is a single fold, run the experiment with fold index = None, otherwise indicate the correct index
        folds = self.get_folds(folds=folds)
        if isinstance(folds, dict):
            x, y = folds['train']
            mdl = self.get_model(model, x=x, y=y, **kwargs)
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
                     log: Optional[str]):
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
            Either a Weights & Biases project name on which to log results or None for no logging.

        """
        # LOGGING & PRINTING
        if log is not None:
            wandb.init(name=f'{model.__name__} - {self.__name__} ({index})',
                       entity=self.ENTITY,
                       project=log,
                       config={'dataset': self.__name__, 'model': model.__name__, 'fold': index, **model.config})
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
                logs.update({f'{split}/{metric}': value for metric, value in metrics[split].items()})
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
