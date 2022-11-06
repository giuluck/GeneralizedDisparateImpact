from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import wandb
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense

from moving_targets.metrics import Metric
from moving_targets.util.typing import Dataset
from src.models.model import Model


class KerasWandbLogger(Callback):
    def __init__(self, fold: Dataset, metrics: List[Metric], run: str, **config):
        super(KerasWandbLogger, self).__init__()
        self.fold: Dataset = fold
        "The evaluation fold."

        self.metrics: List[Metric] = metrics
        """The evaluation metrics."""

        self.run: str = run
        """The Weights&Biases run name."""

        self.config: Dict = config
        """The Weights&Biases run configuration."""

    def on_train_begin(self, logs=None):
        wandb.init(project='nci_calibration', entity='giuluck', name=self.run, config=self.config)

    def on_epoch_end(self, epoch, logs=None):
        log = {}
        for split, (x, y) in self.fold.items():
            p = self.model.predict(x).flatten()
            log.update({f'{split}/{metric.__name__}': metric(x, y, p) for metric in self.metrics})
        wandb.log(log)

    def on_train_end(self, logs=None):
        wandb.finish()


class MLP(Model):
    def __init__(self,
                 classification: bool,
                 val_split: float = 0.0,
                 units: List[int] = (128, 128),
                 batch_size: int = 128,
                 epochs: int = 200,
                 verbose: bool = False,
                 callbacks: List[Callback] = ()):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param val_split:
            The neural network validation split.

        :param batch_size:
            The neural network batch size.

        :param units:
            The neural network hidden units.

        :param epochs:
            The neural network training epochs.

        :param verbose:
            The neural network verbosity.

        :param callbacks:
            The neural network callbacks.
        """
        super(MLP, self).__init__(
            name='mlp',
            classification=classification,
            val_split=val_split,
            batch_size=batch_size,
            units=units,
            epochs=epochs,
            callbacks=callbacks
        )

        self.net: Optional[Sequential] = None
        """The neural model."""

        self.act: str = 'sigmoid' if classification else 'linear'
        """The neural model's output activation."""

        self.units: List[int] = units
        """The neural network hidden units."""

        self.compile_args: Dict[str, Any] = {
            'loss': 'binary_crossentropy' if classification else 'mse',
            'optimizer': 'adam'
        }
        """Custom arguments to be passed to the 'compile' method."""

        self.fit_args: Dict[str, Any] = {
            'epochs': epochs,
            'verbose': verbose,
            'callbacks': callbacks,
            'batch_size': batch_size,
            'validation_split': val_split
        }
        """Custom arguments to be passed to the 'fit' method."""

    def _fit(self, x: pd.DataFrame, y: np.ndarray):
        self.net = Sequential([Dense(u, activation='relu') for u in self.units] + [Dense(1, activation=self.act)])
        self.net.compile(**self.compile_args)
        self.net.fit(x, y, **self.fit_args)

    def _predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.net.predict(x)
