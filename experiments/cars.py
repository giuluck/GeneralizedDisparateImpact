import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from moving_targets.callbacks import DataLogger
from moving_targets.learners import TensorflowMLP
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import MSE, R2, MonotonicViolation
from moving_targets.util.scalers import Scaler

from experiments import util
from experiments.util import compute_monotonicities
from src.constraints import *
from src.master import DefaultMaster, Shape, ExplicitZerosMaster, CovarianceBasedMaster
from src.metrics import SoftShape

theta = 0.0
kernels = [1, 2, 3, 5, 10, 20, 50, 100]
iterations = 5
units = [8, 8]
master = 'covariance'
callbacks = False
preprocess = True
weights = True
backend = GurobiBackend(time_limit=30)
verbose = 1
plot = dict(orient_rows=True, features=[['predictions/mse', 'predictions/r2', 'predictions/monotonicity'],
                                        ['adjusted/shape_w0', 'adjusted/shape_w1', 'adjusted/shape_w+']])


# noinspection PyShadowingNames
class CarsCallback(DataLogger):
    def __init__(self, title=None):
        super(CarsCallback, self).__init__()
        self.title = title

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(figsize=(16, 9), tight_layout=True)
        num_columns = max(np.sqrt(16 / 9 * len(self.iterations)).round().astype(int), 1)
        num_rows = np.ceil(len(self.iterations) / num_columns).astype(int)
        self.data = self.data.sort_values('price').reset_index()
        ax = None
        for idx, it in enumerate(self.iterations):
            ax = plt.subplot(num_rows, num_columns, idx + 1, sharex=ax, sharey=ax)
            sns.lineplot(data=self.data, x='price', y=f'p{it}', linewidth=3, label='predictions')
            if f'z{it}' in self.data:
                sns.scatterplot(data=self.data, x='price', y=f'z{it}', alpha=0.6, label='adjusted')
            else:
                sns.scatterplot(data=self.data, x='price', y='y', alpha=0.6, label='original')
            ax.set_title(str(it))
        plt.suptitle(self.title)
        plt.show()


if __name__ == '__main__':
    # handle data and build learner
    with importlib.resources.path('data', 'cars.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df[['price']], df['sales'].values
        if preprocess:
            x, y = Scaler('std').fit_transform(x), Scaler('norm').fit_transform(y)
    metrics = [MSE(), R2(), MonotonicViolation(lambda v: compute_monotonicities(v, v, directions=[-1]))]
    cst = Lower(theta)

    # test different polynomial kernels
    print('----------------------------------------------------------------------------')
    for k in kernels:
        print(f'KERNEL {k}')
        # build learner
        lrn = TensorflowMLP(loss='mse', hidden_units=units, epochs=1000, verbose=False)

        # build master
        if master == 'default':
            shapes = [Shape('price', constraints=[None, cst, *[Null() for _ in range(1, k)]], kernel=k)]
            mst = DefaultMaster(shapes=shapes, backend=backend, binary=False)
        elif master == 'zeros':
            mst = ExplicitZerosMaster(feature='price', constraint=cst, degree=k, backend=backend, binary=False)
        elif master == 'covariance':
            mst = CovarianceBasedMaster(feature='price', constraint=cst, degree=k, backend=backend, binary=False)
        else:
            raise AssertionError(f"Unknown master '{master}'")

        # run experiment
        title = dict(title=f'KERNEL {k}')
        post = lambda wv: {'w0': wv[0], 'w1': wv[1], 'w+': np.abs(wv[2:]).max() if len(wv) > 2 else np.nan}
        util.run(x=x, y=y, features=['price'], learner=lrn, master=mst, iterations=iterations, verbose=verbose,
                 metrics=metrics + ([SoftShape('price', kernels=k, postprocessing=post)] if weights else []),
                 plot={**plot, **title} if isinstance(plot, dict) else (title if plot else False),
                 callbacks=[CarsCallback(**title)] if callbacks else [])
        print('----------------------------------------------------------------------------')
