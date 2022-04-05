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
    @staticmethod
    def plot_ice(x, model, feature, space):
        plt.figure(figsize=(16, 9), tight_layout=True)
        df = pd.concat([x] * len(space))
        df[feature] = space.repeat(len(x))
        df['y'] = model.predict(df)
        df = df.reset_index()
        for i, group in df.groupby('index'):
            sns.lineplot(data=group, x=feature, y='y', color='black', alpha=0.4)
        sns.lineplot(data=df, x=feature, y='y', ci=None, color='red', linewidth=3, label='PD')
        plt.title(f'ICE Plot for {feature}')
        plt.show()

    def __init__(self, feature, space):
        super(ICECallback, self).__init__()
        self.feature = feature
        self.space = space
        self.x = None

    def on_process_start(self, macs, x, y, val_data):
        self.x = x

    def on_process_end(self, macs, val_data):
        self.plot_ice(x=self.x, model=macs, feature=self.feature, space=self.space)


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
