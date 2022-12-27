from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.metrics import DIDI
from moving_targets.util.typing import Dataset
from src.experiments import get
from src.metrics import RegressionWeight
from src.models import MovingTargets

sns.set_context('notebook')
sns.set_style('whitegrid')


class DistributionCallback(Callback):
    def __init__(self, feature, classification=True, predictions=True):
        super(DistributionCallback, self).__init__()
        self.feature = feature
        self.classification = classification
        self.predictions = predictions
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
            column = f'pred_{it}' if self.predictions else (f'adj_{it}' if it > 0 else 'y')
            sns.scatterplot(
                data=self.data.groupby([self.feature, column]).size().to_frame('size').reset_index(),
                x=self.feature,
                y=column,
                size='size',
                sizes=(30, 600),
                alpha=0.7,
                edgecolor='black',
                color='tab:cyan',
                legend=None
            )
            weights = []
            for d in [1, 2, 3, 4]:
                rw = RegressionWeight(self.feature, classification=self.classification, degree=d)
                w = rw(self.data, self.data['y'], self.data[column])
                weights.append([round(w, 2)] if d == 1 else [round(wi, 2) for wi in w.values()])
                sns.regplot(self.data, x=self.feature, y=column, order=d, ci=None, scatter=False, label=f'lr {d}')
            ax.set(title=' - '.join([str(ws) for ws in weights]), xlabel='', ylabel='')
            plt.legend()
        plt.suptitle(f"{self.feature} ({'predictions' if self.predictions else 'adjusted'})")
        plt.show()


if __name__ == '__main__':
    exp = get('adult continuous')
    fold = exp.get_folds(folds=1)
    degrees = 1 if 'categorical' in exp.__name__ else 5
    model = exp.get_model(
        model='mt rf',
        fold=fold,
        degrees=degrees,
        iterations=10,
        # metrics=[m for m in exp.metrics if isinstance(m, DIDI) or isinstance(m, RegressionWeight)],
        # metrics=[m for m in exp.metrics if isinstance(m, DIDI)],
        # history=dict(features=None, orient_rows=True, excluded=['predictions/*', 'train/*', 'test/*']),
        history=dict(features=None, orient_rows=True, excluded=['predictions/*', 'adjusted/*']),
        verbose=1
    )
    assert isinstance(model, MovingTargets), f"Model should be MovingTargets, got '{type(model)}'"
    # for f in exp.excluded:
    #     model.add_callback(callback=DistributionCallback(feature=f, classes=exp.classification, predictions=False))
    #     model.add_callback(callback=DistributionCallback(feature=f, classes=exp.classification, predictions=True))
    print('MODEL CONFIGURATION:')
    print(f'  > model: {model.__name__}')
    print(f'  > dataset: {exp.__name__}')
    for k, v in model.config.items():
        print(f'  > {k} --> {v}')
    print('-------------------------------------------------')
    xf, yf = fold['train']
    exp.run_instance(model=model, x=xf, y=yf, fold=fold, index=None, log=None, show=False)
