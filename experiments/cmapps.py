import importlib.resources

import pandas as pd
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE

from experiments import util
from src.constraints import Smaller
from src.master import DefaultMaster, Shape

iterations = 5
theta = 1e-3
degree = 1
backend = GurobiBackend(time_limit=10)
verbose = True
callbacks = []
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data
    with importlib.resources.path('data', 'cmapps.csv') as filepath:
        df = pd.read_csv(filepath)
    x, y = df.drop(columns=['src', 'machine', 'cycle', 'rul']), df['rul'].values
    features = ['p1', 'p2', 'p3']

    # build model
    learner = LinearRegression(polynomial=degree)
    shapes = [Shape(f, constraints={i + 1: Smaller(theta) for i in range(degree)}, kernel=degree) for f in features]
    master = DefaultMaster(shapes=shapes, backend=backend, loss='mse')
    postprocessing = lambda w: {f'w{i + 1}': v for i, v in enumerate(w[1:])}
    metrics = [R2(), MSE(), *[util.Pearson(feature=f, name=f) for f in features]]

    util.run(x=x, y=y, features=features, learner=learner, master=master, metrics=metrics, callbacks=callbacks,
             iterations=iterations, verbose=verbose, plot=plot)
