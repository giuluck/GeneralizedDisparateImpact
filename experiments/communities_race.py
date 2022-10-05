import importlib.resources

import numpy as np
import pandas as pd

from experiments import util
from moving_targets.learners import LinearRegression
from moving_targets.metrics import DIDI, R2, MSE
from moving_targets.util.scalers import Scaler
from src.master import CausalExclusionMaster
from src.metrics import RegressionWeight

threshold = 0.2
iterations = 5
verbose = True
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data and learner
    with importlib.resources.path('data', 'communities.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        x, y = Scaler('std').fit_transform(x), Scaler('norm').fit_transform(y)
    lrn = LinearRegression(polynomial=2)
    mtr = [R2(), MSE(), DIDI(protected='race', classification=False), RegressionWeight(feature='race')]

    # compute the relative accepted violation
    z = x[['race']].values
    z = np.concatenate((np.ones_like(z), z), axis=1)
    th, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
    th = abs(threshold * th[1])

    # build the master and run the experiment
    mst = CausalExclusionMaster(features='race', thresholds=th, degrees=1, classification=False)
    util.run(x=x, y=y, learner=lrn, master=mst, metrics=mtr, iterations=iterations, verbose=verbose, plot=plot)
