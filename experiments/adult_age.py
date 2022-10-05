import importlib.resources

import numpy as np
import pandas as pd
from moving_targets.learners import LogisticRegression
from moving_targets.metrics import Accuracy, CrossEntropy
from moving_targets.util.scalers import Scaler

from experiments import util
from src.master import CausalExclusionMaster
from src.metrics import BinnedDIDI, RegressionWeight

threshold = 0.2
bins = [2, 3, 5, 10]
degrees = [1, 2, 3, 5]
iterations = 5
verbose = 1
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data and learner
    with importlib.resources.path('data', 'adult.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        x = Scaler('std').fit_transform(x)
    mtr = [Accuracy(), CrossEntropy()]
    mtr += [BinnedDIDI(classification=True, protected='age', bins=b) for b in bins]

    # compute the relative accepted violation
    z = x[['age']].values
    z = np.concatenate((np.ones_like(z), z), axis=1)
    th, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
    th = abs(threshold * th[1])

    # test different polynomial kernels
    print('----------------------------------------------------------------------------')
    for d in degrees:
        print(f'KERNEL {d}')
        lrn = LogisticRegression(max_iter=10000)
        mst = CausalExclusionMaster(features='age', thresholds=th, degrees=d, classification=True)
        title = dict(title=f'KERNEL {d}')
        util.run(x=x, y=y, learner=lrn, master=mst, iterations=iterations, verbose=verbose,
                 metrics=[*mtr, RegressionWeight(feature='age', degree=d)],
                 plot={**plot, **title} if isinstance(plot, dict) else (title if plot else False))
        print('----------------------------------------------------------------------------')
