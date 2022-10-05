import importlib.resources

import numpy as np
import pandas as pd
from moving_targets.learners import LogisticRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import Accuracy, CrossEntropy, DIDI

from experiments import util
from src.constraints import Smaller
from src.master import Shape, DefaultMaster

theta = 0.2
iterations = 5
backend = GurobiBackend(time_limit=30)
verbose = True
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data and learner
    with importlib.resources.path('data', 'adult.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
    learner = LogisticRegression(max_iter=10000, x_scaler='std')
    metrics = [Accuracy(), CrossEntropy(), DIDI(protected='race', classification=True, percentage=True, name='didi')]

    # config shape-related metrics with (relative) accepted violations
    violations = {}
    for f, c in x.transpose().iterrows():
        f = str(f)
        if f.startswith('race'):
            c = c.values.reshape((-1, 1))
            c = np.concatenate((np.ones_like(c), c), axis=1)
            w, _, _, _ = np.linalg.lstsq(c, y, rcond=None)
            violations[f] = abs(theta * w[1])

    # build master (constraint each protected feature to have first-order degree weight lower than the violation)
    shapes = [Shape(p, constraints=[None, Smaller(v)], kernel=1) for p, v in violations.items()]
    master = DefaultMaster(shapes=shapes, backend=backend, binary=True)

    util.run(x=x, y=y, features=list(violations.keys()), learner=learner, master=master, metrics=metrics,
             iterations=iterations, verbose=verbose, plot=plot)
