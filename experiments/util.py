import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from moving_targets.callbacks import Callback
from moving_targets.metrics import Metric
from scipy.stats import pearsonr


class ICECallback(Callback):
    def __init__(self, feature, space):
        super(ICECallback, self).__init__()
        self.feature = feature
        self.space = space
        self.data = None
        self.columns = None

    def on_process_start(self, macs, x, y, val_data):
        self.data = pd.concat([x] * len(self.space))
        self.data[self.feature] = self.space.repeat(len(x))
        self.columns = self.data.columns

    def on_training_end(self, macs, x, y, p, val_data):
        self.data[f'p{macs.iteration}'] = macs.predict(self.data[self.columns])

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(figsize=(16, 9), tight_layout=True)
        iterations = len(self.data.columns) - len(self.columns)
        num_columns = max(np.sqrt(16 / 9 * iterations).round().astype(int), 1)
        num_rows = np.ceil(iterations / num_columns).astype(int)
        self.data = self.data.reset_index()
        ax = None
        for idx in range(iterations):
            ax = plt.subplot(num_rows, num_columns, idx + 1, sharex=ax, sharey=ax)
            for i, group in self.data.groupby('index'):
                sns.lineplot(data=group, x=self.feature, y=f'p{idx}', color='black', alpha=0.4)
            sns.lineplot(data=self.data, x=self.feature, y=f'p{idx}', ci=None, color='red', linewidth=3, label='PD')
            ax.set(xlabel='', ylabel='')
            ax.set_title(str(idx))
        plt.suptitle(f'ICE Plots for {self.feature}')
        plt.show()


class Pearson(Metric):
    def __init__(self, feature, name='pearson'):
        super(Pearson, self).__init__(name=name)
        self.feature = feature

    def __call__(self, x, y, p):
        return pearsonr(x[self.feature], p)[0]


def config(seed: int = 0):
    random.seed(0)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sns.set_style('whitegrid')
    sns.set_context('notebook')
