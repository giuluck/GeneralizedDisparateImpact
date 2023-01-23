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

from moving_targets.metrics import Metric, MSE, R2, CrossEntropy, Accuracy, DIDI
from moving_targets.util.typing import Dataset
from src.metrics import HGR, BinnedDIDI, GeneralizedDIDI, RegressionWeights
from src.models import Model, RandomForest, GradientBoosting, NeuralNetwork, MovingTargets, NeuralSBR


class Experiment:
    SEED: int = 0
    """The random seed."""

    ENTITY: str = 'shape-constraints'
    """The Weights&Biases entity name."""

    BINS: List[int] = [2, 3, 5, 10]
    """The number of bins to be used in the BinnedDIDI metric."""

    DEGREES: List[int] = [1, 2, 3, 4, 5]
    """The kernel degrees to be used in the GeneralizedDIDI metric for continuous protected features."""

    TRHESHOLD: float = 0.2
    """The default threshold for the feature to exclude."""

    RELATIVE: int = 1
    """The default kernel on which to compute the relative threshold."""

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
    def load_data(scale: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """Loads the dataset.

        :param scale:
            Whether or not to scale the data in order to have properly standardized/normalized values.

        :return:
            A tuple (x, y) containing the input data and the target vector.
        """
        raise NotImplementedError("please implement static method 'load_data'")

    def __init__(self, name: str, continuous: bool, classification: bool, excluded: str, units: List[int]):
        """
        :param name:
            The experiment name.

        :param continuous:
            Whether the excluded feature is binary or continuous.

        :param classification:
            Whether this is a classification or a regression task.

        :param excluded:
            Either a single feature or the list of features to exclude.

        :param units:
            The neural networks default units.
        """
        metrics = [Accuracy(), CrossEntropy()] if classification else [R2(), MSE()]
        metrics += [RegressionWeights(classification, excluded, degree=d, name=f'k{d}-alpha') for d in self.DEGREES]
        metrics += [
            HGR(feature=excluded, relative=True, chi2=False, name='rel_hgr'),
            HGR(feature=excluded, relative=False, chi2=False, name='abs_hgr'),
            HGR(feature=excluded, relative=True, chi2=True, name='rel_chi2'),
            HGR(feature=excluded, relative=False, chi2=True, name='abs_chi2')
        ]
        if continuous:
            for b in self.BINS:
                metrics += [
                    BinnedDIDI(classification, excluded, bins=b, relative=True, name=f'rel_binned_didi_{b}'),
                    BinnedDIDI(classification, excluded, bins=b, relative=False, name=f'abs_binned_didi_{b}')
                ]
            for d in self.DEGREES:
                metrics += [
                    GeneralizedDIDI(classification, excluded, degree=d, relative=1, name=f'rel_generalized_didi_{d}'),
                    GeneralizedDIDI(classification, excluded, degree=d, relative=0, name=f'abs_generalized_didi_{d}'),
                ]
        else:
            metrics += [
                DIDI(classification, excluded, percentage=True, name='rel_didi'),
                DIDI(classification, excluded, percentage=False, name='abs_didi')
            ]

        self.__name__: str = name
        """The dataset name."""

        self.data: Tuple[pd.DataFrame, np.ndarray] = self.load_data()
        """The tuple (x, y) containing the input data and the target vector."""

        self.continuous: bool = continuous
        """Whether the excluded feature is binary or continuous."""

        self.classification: bool = classification
        """Whether this is a classification or a regression task."""

        self.excluded: str = excluded
        """The list of features whose causal effect should be excluded."""

        self.units: List[int] = units
        """The neural network default units."""

        self.epochs: int = 200
        """The neural network default units."""

        self.batch: int = 128
        """The neural network default batch size."""

        self.degree: int = 5 if continuous else 1
        """The default kernel degree."""

        self.metrics: List[Metric] = metrics
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
            kwargs['batch_size'] = kwargs.get('batch_size') or self.batch
            kwargs['epochs'] = kwargs.get('epochs') or self.epochs
            return NeuralNetwork(classification=self.classification, **kwargs)
        elif model == 'sbr hgr':
            kwargs['relative'] = kwargs.get('relative') or self.RELATIVE
            kwargs['threshold'] = kwargs.get('threshold') or self.TRHESHOLD
            kwargs['hidden_units'] = kwargs.get('hidden_units') or self.units
            kwargs['batch_size'] = kwargs.get('batch_size') or self.batch
            kwargs['epochs'] = kwargs.get('epochs') or self.epochs
            return NeuralSBR(penalty='hgr', excluded=self.excluded, classification=self.classification, **kwargs)
        elif model.startswith('sbr'):
            _, penalty = model.split(' ')
            kwargs['relative'] = kwargs.get('relative') or self.RELATIVE
            kwargs['threshold'] = kwargs.get('threshold') or self.TRHESHOLD
            kwargs['hidden_units'] = kwargs.get('hidden_units') or self.units
            kwargs['epochs'] = kwargs.get('epochs') or int(2.5 * self.epochs)
            kwargs['degree'] = kwargs.get('degree') or self.degree
            return NeuralSBR(penalty=penalty, excluded=self.excluded, classification=self.classification,
                             batch_size=len(self.data[0]), **kwargs)
        elif model.startswith('mt '):
            _, learner, master = model.split(' ')
            if learner == 'nn':
                kwargs['hidden_units'] = kwargs.get('hidden_units') or self.units
                kwargs['batch_size'] = kwargs.get('batch_size') or self.batch
                kwargs['epochs'] = kwargs.get('epochs') or self.epochs
            kwargs['relative'] = kwargs.get('relative') or self.RELATIVE
            kwargs['threshold'] = kwargs.get('threshold') or self.TRHESHOLD
            kwargs['metrics'] = kwargs.get('metrics') or self.metrics
            kwargs['degree'] = kwargs.get('degree') or self.degree
            return MovingTargets(
                master=master,
                learner=learner,
                classification=self.classification,
                excluded=self.excluded,
                **kwargs
            )
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
            results = {'predictions': list(p)}
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
