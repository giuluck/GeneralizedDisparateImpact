import importlib.resources

import numpy as np
import pandas as pd

from experiments import util
from moving_targets.learners import LogisticRegression
from moving_targets.metrics import Accuracy, CrossEntropy, DIDI
from moving_targets.util.scalers import Scaler
from src.master import CausalExclusionMaster
from src.metrics import RegressionWeight

threshold = 0.2
iterations = 5
verbose = True
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data and learner
    with importlib.resources.path('data', 'adult.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        x = Scaler('std').fit_transform(x)
    ftr = [f for f in x.columns if f.startswith('race')]
    lrn = LogisticRegression(max_iter=10000)
    mtr = [Accuracy(), CrossEntropy(), DIDI(protected='race', classification=True)]
    mtr += [RegressionWeight(feature=f, name=f[5:]) for f in ftr]

    # compute relative accepted violations
    th = {}
    for f in ftr:
        z = x[[f]].values
        z = np.concatenate((np.ones_like(z), z), axis=1)
        w, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
        th[f] = abs(threshold * w[1])

    # build the master and run the experiment
    mst = CausalExclusionMaster(features=list(th.keys()), thresholds=list(th.values()), degrees=1, classification=True)
    util.run(x=x, y=y, learner=lrn, master=mst, metrics=mtr, iterations=iterations, verbose=verbose, plot=plot)
