from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset

from moving_targets.learners import RandomForestRegressor
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
            name='rf',
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


class NeuralNetwork(Model):
    class _Dataset(Dataset):
        def __init__(self, x, y):
            assert len(x) == len(y), f"Data should have the same length, but len(x) = {len(x)} and len(y) = {len(y)} "
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(np.expand_dims(y, axis=-1), dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    def __init__(self,
                 input_units: int,
                 classification: bool,
                 validation_split: float = 0.0,
                 hidden_units: List[int] = (128, 128),
                 batch_size: int = 128,
                 epochs: int = 250,
                 verbose: bool = False):
        """
        :param input_units:
            The number of input units.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param validation_split:
            The neural network validation split.

        :param hidden_units:
            The neural network hidden units.

        :param batch_size:
            The neural network batch size.

        :param epochs:
            The neural network training epochs.

        :param verbose:
            The neural network verbosity.
        """
        super(NeuralNetwork, self).__init__(
            name='nn',
            input_units=input_units,
            classification=classification,
            validation_split=validation_split,
            hidden_units=hidden_units,
            batch_size=batch_size,
            epochs=epochs
        )

        layers = [nn.Linear(in_features=input_units, out_features=hidden_units[0]), nn.ReLU()]
        for i, h in enumerate(hidden_units[1:]):
            layers += [nn.Linear(in_features=hidden_units[i - 1], out_features=hidden_units[i]), nn.ReLU()]
        layers += [nn.Linear(in_features=hidden_units[-1], out_features=1)]
        layers += [nn.Sigmoid()] if classification else []

        self.model: Optional[nn.Module] = nn.Sequential(*layers)
        """The torch model."""

        self.loss: nn.Module = nn.BCELoss() if classification else nn.MSELoss()
        """The neural network loss function."""

        self.optimizer: optim.Optimizer = optim.Adam(params=self.model.parameters())
        """The neural network optimizer."""

        self.epochs: int = epochs
        """The number of training epochs."""

        self.validation_split: float = validation_split
        """The validation split for neural network training."""

        self.batch_size: int = batch_size
        """The batch size for neural network training."""

        self.verbose: bool = verbose
        """Whether or not to print information during the neural network training."""

    def _print(self, epoch: int, loss: float, batch: Optional[Tuple[int, int]] = None):
        if not self.verbose:
            return

        # carriage return to write in the same line
        print(f'\r', end='')
        # print the epoch number with trailing spaces on the left to match the maximum epoch
        print(f'Epoch {epoch:{len(str(self.epochs))}}', end=' ')
        # print the loss value (either loss for this single batch or for the whole epoch)
        print(f'- loss = {loss:.4f}', end='')
        # check whether this is the information of a single batch or of the whole epoch and for the latter case (batch
        # is None) print a new line, while for the former (batch is a tuple) print the batch number and no new line
        if batch is None:
            print()
        else:
            batch, batches = batch
            print(f' (batch {batch:{len(str(batches))}} of {batches})', end='')

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        ds = NeuralNetwork._Dataset(x=np.array(x), y=np.array(y))
        loader = DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=True)
        batches = np.ceil(len(ds) / self.batch_size).astype(int)
        self.model.train()
        for epoch in np.arange(self.epochs) + 1:
            epoch_loss = 0.0
            for batch, (inp, out) in enumerate(loader):
                self.optimizer.zero_grad()
                pred = self.model(inp)
                loss = self.loss(pred, out)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * inp.size(0)
                self._print(epoch=epoch, loss=loss.item(), batch=(batch + 1, batches))
            self._print(epoch=epoch, loss=epoch_loss / len(ds))
        self.model.eval()

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        x = torch.tensor(np.array(x), dtype=torch.float32)
        return self.model(x).detach().numpy().squeeze()
