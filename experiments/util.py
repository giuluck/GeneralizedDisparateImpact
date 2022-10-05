import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.metrics import Metric, DIDI
from moving_targets.util.typing import Dataset
from scipy.stats import pearsonr
from tensorflow.python.keras.callbacks import ModelCheckpoint


class DistributionCallback(Callback):
    def __init__(self, feature, plot='predictions'):
        super(DistributionCallback, self).__init__()
        if plot == 'predictions':
            plot_p, plot_z = True, False
        elif plot == 'adjusted':
            plot_p, plot_z = False, True
        elif plot in ['both', 'all']:
            plot_p, plot_z = True, True
        else:
            raise AssertionError(f"Unexpected string '{plot}' for plot parameter")

        self.feature = feature
        self.plot_adjusted = plot_z
        self.plot_prediction = plot_p
        self.iterations = []
        self.data = None

    def on_process_start(self, macs, x, y, val_data):
        self.data = pd.DataFrame.from_dict({self.feature: x[self.feature], 'y': y})

    def on_iteration_start(self, macs, x, y, val_data):
        self.iterations.append(macs.iteration)

    def on_training_end(self, macs, x, y, p, val_data):
        self.data[f'pred_{macs.iteration}'] = p

    def on_adjustment_end(self, macs, x, y, z, val_data):
        self.data[f'adj_{macs.iteration}'] = z

    def on_process_end(self, macs: MACS, x: pd.DataFrame, y: np.ndarray, val_data: Optional[Dataset]):
        plt.figure(figsize=(16, 9), tight_layout=True)
        num_columns = max(np.sqrt(16 / 9 * len(self.iterations)).round().astype(int), 1)
        num_rows = np.ceil(len(self.iterations) / num_columns).astype(int)
        ax = None
        for it in self.iterations:
            ax = plt.subplot(num_rows, num_columns, it + 1, sharex=ax, sharey=ax)
            if self.plot_prediction:
                plt.scatter(x=self.data[self.feature], y=self.data[f'pred_{it}'], label='predictions')
            if self.plot_adjusted and f'adj_{it}' in self.data:
                plt.scatter(x=self.data[self.feature], y=self.data[f'adj_{it}'], label='adjusted')
            ax.set(title=str(it), xlabel='', ylabel='')
            plt.legend()
        plt.suptitle(f'Distribution Plots for {self.feature}')
        plt.show()


class ICECallback(Callback):
    def __init__(self, feature, samples=-1, steps=20):
        super(ICECallback, self).__init__()
        self.feature = feature
        self.samples = samples
        self.steps = steps
        self.data = None
        self.columns = None

    def on_process_start(self, macs, x, y, val_data):
        space = np.linspace(x[self.feature].min(), x[self.feature].max(), self.steps)
        if self.samples != -1:
            x = x.sample(n=self.samples, random_state=0)
        self.data = pd.concat([x] * len(space))
        self.data[self.feature] = space.repeat(len(x))
        self.columns = self.data.columns

    def on_training_end(self, macs, x, y, p, val_data):
        self.data[f'pred_{macs.iteration}'] = macs.predict(self.data[self.columns])

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(figsize=(16, 9), tight_layout=True)
        iterations = len(self.data.columns) - len(self.columns)
        num_columns = max(np.sqrt(16 / 9 * iterations).round().astype(int), 1)
        num_rows = np.ceil(iterations / num_columns).astype(int)
        self.data = self.data.reset_index()
        ax = None
        for it in range(iterations):
            ax = plt.subplot(num_rows, num_columns, it + 1, sharex=ax, sharey=ax)
            for i, group in self.data.groupby('index'):
                sns.lineplot(data=group, x=self.feature, y=f'pred_{it}', color='black', alpha=0.4)
            sns.lineplot(data=self.data, x=self.feature, y=f'pred_{it}', ci=None, color='red', linewidth=3, label='PD')
            ax.set(xlabel='', ylabel='')
            ax.set_title(str(it))
        plt.suptitle(f'ICE Plots for {self.feature}')
        plt.show()


class BestEpoch(ModelCheckpoint):
    def __init__(self, monitor='loss', filepath='.weights.h5'):
        super(BestEpoch, self).__init__(filepath, monitor=monitor, save_weights_only=True, save_best_only=True)

    def on_train_begin(self, logs=None):
        self.best = -np.Inf if self.monitor_op == np.greater else np.Inf

    def on_train_end(self, logs=None):
        super(BestEpoch, self).on_train_end(logs=logs)
        self.model.load_weights(self.filepath)
        os.remove(self.filepath)


class Pearson(Metric):
    def __init__(self, feature, name='pearson'):
        super(Pearson, self).__init__(name=name)
        self.feature = feature

    def __call__(self, x, y, p):
        return pearsonr(x[self.feature], p)[0]


class BinnedDIDI(DIDI):
    def __init__(self, classification, protected, bins=2, percentage=True, name='didi'):
        super(BinnedDIDI, self).__init__(classification, protected, percentage, name=f'{name}_{bins}')
        self.bins = bins

    def __call__(self, x, y, p):
        x = x.copy()
        x[self.protected] = pd.qcut(x[self.protected], q=self.bins).cat.codes
        return super(BinnedDIDI, self).__call__(x, y, p)


def compute_monotonicities(samples, references, directions, eps: float = 1e-6):
    assert samples.ndim <= 2, f"'samples' should have 2 dimensions at most, but it has {samples.ndim}"
    assert references.ndim <= 2, f"'references' should have 2 dimensions at most, but it has {references.ndim}"
    samples, references = np.atleast_2d(samples), np.atleast_2d(references)
    samples = np.hstack([samples] * len(references)).reshape((len(samples), len(references), -1))
    differences = samples - references
    differences[np.abs(differences) < eps] = 0.
    num_differences = np.sign(np.abs(differences)).sum(axis=-1)
    monotonicities = np.sign(directions * differences).sum(axis=-1)
    monotonicities = np.squeeze(monotonicities * (num_differences == 1)).astype('int')
    return np.int32(monotonicities) if monotonicities.ndim == 0 else monotonicities


def run(x, y, features, master, learner, metrics=(), callbacks=(), iterations=5, verbose=True, plot=True, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    cbs = []
    for c in callbacks:
        if isinstance(c, Callback):
            cbs.append(c)
        elif c == 'dist':
            cbs += [DistributionCallback(feature=f) for f in features]
        elif c == 'ice':
            cbs += [ICECallback(feature=f, samples=50, steps=20) for f in features]
        else:
            raise AssertionError(f"Unknown callback alias '{c}'")
    model = MACS(init_step='pretraining', learner=learner, master=master, metrics=metrics)
    history = model.fit(x=x, y=y, iterations=iterations, callbacks=cbs, verbose=verbose)
    if isinstance(plot, dict):
        history.plot(**plot)
    elif plot is True:
        history.plot()
