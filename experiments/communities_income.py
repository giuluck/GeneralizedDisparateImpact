import importlib.resources

import numpy as np
import pandas as pd
from moving_targets.learners import LinearRegression
from moving_targets.metrics import R2, MSE
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
    # handle data
    with importlib.resources.path('data', 'communities.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        x, y = Scaler('std').fit_transform(x), Scaler('norm').fit_transform(y)
    mtr = [R2(), MSE()]
    mtr += [BinnedDIDI(classification=False, protected='perCapInc', bins=b) for b in bins]

    # compute the relative accepted violation
    z = x[['perCapInc']].values
    z = np.concatenate((np.ones_like(z), z), axis=1)
    th, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
    th = abs(threshold * th[1])

    # test different polynomial kernels
    print('----------------------------------------------------------------------------')
    for d in degrees:
        print(f'KERNEL {d}')
        lrn = LinearRegression(polynomial=2)
        mst = CausalExclusionMaster(features='perCapInc', thresholds=th, degrees=d, classification=False)
        title = dict(title=f'KERNEL {d}')
        util.run(x=x, y=y, learner=lrn, master=mst, iterations=iterations, verbose=verbose,
                 metrics=[*mtr, RegressionWeight(feature='perCapInc', degree=d)],
                 plot={**plot, **title} if isinstance(plot, dict) else (title if plot else False))
        print('----------------------------------------------------------------------------')
