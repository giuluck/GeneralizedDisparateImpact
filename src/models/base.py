from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from torch import nn
from torch.utils.data import DataLoader

from moving_targets.learners import RandomForestRegressor, TorchMLP
from moving_targets.metrics import Metric
from src.models import Model


class RandomForest(Model):
    def __init__(self,
                 classification: bool,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 **kwargs):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param n_estimators:
            The number of trees in the forest.

        :param max_depth:
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        :param kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestRegressor instance.
        """

        super(RandomForest, self).__init__(
            name='rf',
            classification=classification,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **kwargs
        )

        rf = RandomForestClassifier if classification else RandomForestRegressor
        self.model: Optional = rf(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        """The random forest model."""

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x)


class GradientBoosting(Model):
    def __init__(self,
                 classification: bool,
                 n_estimators: int = 100,
                 min_samples_leaf: Union[int, float] = 1,
                 **kwargs):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param n_estimators:
            The number of trees in the forest.

        :param min_samples_leaf:
            The minimum number of samples required to be at a leaf node. If int, then consider `min_samples_leaf` as
            the minimum number, otherwise `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are
            the minimum number of samples for each node.

        :param kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestRegressor instance.
        """

        super(GradientBoosting, self).__init__(
            name='gb',
            classification=classification,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )

        rf = GradientBoostingClassifier if classification else GradientBoostingRegressor
        self.model: Optional = rf(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **kwargs)
        """The random forest model."""

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        self.model.fit(x, y)

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x)


class NeuralNetwork(Model, TorchMLP):
    class WandbLogger:
        def __init__(self, fold: Dict[str, Tuple[Any, np.ndarray]], metrics: List[Metric], run: str, **config):
            super(NeuralNetwork.WandbLogger, self).__init__()

            self.fold: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = fold
            "The evaluation fold."

            self.metrics: List[Metric] = metrics
            """The evaluation metrics."""

            self.run: str = run
            """The Weights&Biases run name."""

            self.config: Dict[str, Any] = config
            """Additional Weights&Biases configuration."""

        def init(self, model):
            config = self.config.copy()
            config.update(model.config)
            wandb.init(project='nci_calibration', entity='shape-constraints', name=self.run, config=config)

        def log(self, model):
            log = {}
            for split, (x, y) in self.fold.items():
                p = model.model(torch.tensor(np.array(x), dtype=torch.float32)).detach().numpy().squeeze()
                log.update({f'{split}/{metric.__name__}': metric(x, y, p) for metric in self.metrics})
            wandb.log(log)

        @staticmethod
        def close():
            wandb.finish()

    def __init__(self,
                 classification: bool,
                 hidden_units: List[int],
                 batch_size: int,
                 epochs: int,
                 validation_split: float = 0.0,
                 logger: Optional[WandbLogger] = None,
                 verbose: bool = False):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param hidden_units:
            The neural network hidden units.

        :param batch_size:
            The neural network batch size.

        :param epochs:
            The neural network training epochs.

        :param validation_split:
            The neural network validation split.

        :param logger:
            An optional callback for logging the training history on Weights&Biases.

        :param verbose:
            The neural network verbosity.
        """
        super(NeuralNetwork, self).__init__(
            name='nn',
            classification=classification,
            validation_split=validation_split,
            hidden_units=hidden_units,
            batch_size=batch_size,
            epochs=epochs
        )

        super(Model, self).__init__(
            loss=nn.BCELoss() if classification else nn.MSELoss(),
            activation=nn.Sigmoid() if classification else None,
            hidden_units=hidden_units,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            verbose=verbose,
            optimizer='Adam',
            x_scaler=None,
            y_scaler=None,
            stats=False
        )

        self.logger: Optional[NeuralNetwork.WandbLogger] = logger
        """An optional callback for logging the training history on Weights&Biases."""

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        ds = TorchMLP.Dataset(x=np.array(x), y=np.array(y))
        loader = DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        batches = np.ceil(len(ds) / self.batch_size).astype(int)
        optimizer = self._build_model(input_units=x.shape[1])
        self.model.train()
        if self.logger is not None:
            self.logger.init(self)
        for epoch in np.arange(self.epochs) + 1:
            epoch_loss = 0.0
            for batch, (inp, out) in enumerate(loader):
                optimizer.zero_grad()
                pred = self.model(inp)
                loss = self.loss(pred, out)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inp.size(0)
                self._print(epoch=epoch, info={'loss': loss.item()}, batch=(batch + 1, batches))
            self._print(epoch=epoch, info={'loss': epoch_loss / len(ds)})
            if self.logger is not None:
                self.logger.log(self)
        if self.logger is not None:
            self.logger.close()
        self.model.eval()

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        x = torch.tensor(np.array(x), dtype=torch.float32)
        return self.model(x).detach().numpy().squeeze()
