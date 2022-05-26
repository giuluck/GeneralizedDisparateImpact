import importlib.resources

import numpy as np
import pandas as pd
from moving_targets.learners import LinearRegression
from moving_targets.metrics import DIDI, R2, MSE

from experiments import util
from src.constraints import Smaller
from src.master import DefaultMaster, Shape

theta = 0.2
iterations = 5
backend = 'gurobi'
verbose = True
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data and learner
    with importlib.resources.path('data', 'communities.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
    learner = LinearRegression(x_scaler='std', y_scaler='norm')
    metrics = [R2(), MSE(), DIDI(protected='race', classification=False, percentage=True, name='didi')]

    # build master (constraint protected feature to have first-order degree weight lower than the relative violation)
    c = x[['race']].values
    c = np.concatenate((np.ones_like(c), c), axis=1)
    w, _, _, _ = np.linalg.lstsq(c, y, rcond=None)
    v = abs(theta * w[1])
    shapes = [Shape('race', constraints=[None, Smaller(v)], kernel=1)]
    master = DefaultMaster(shapes=shapes, backend=backend, binary=False)

    util.run(x=x, y=y, features=['race'], learner=learner, master=master, metrics=metrics, iterations=iterations,
             verbose=verbose, plot=plot)
