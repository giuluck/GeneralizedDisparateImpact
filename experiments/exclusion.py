import numpy as np
from moving_targets.learners import LinearRegression
from moving_targets.metrics import R2, MSE

from data.exclusion import generator
from experiments import util
from src.constraints import Smaller
from src.master import DefaultMaster, Shape

iterations = 5
theta = 1e-3
degree = 1
backend = 'gurobi'
verbose = True
callbacks = []
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data
    df = generator(fy=lambda qh: 3 + np.exp(qh), fq=lambda p, qh: qh + 2 * p).generate()
    x, y = df[['p', 'q']], df['y'].values
    features = ['p']

    # build learner and master
    learner = LinearRegression(polynomial=degree)
    shapes = [Shape(f, constraints={i + 1: Smaller(theta) for i in range(degree)}, kernel=degree) for f in features]
    master = DefaultMaster(shapes=shapes, backend=backend, loss='mse')
    metrics = [R2(), MSE(), *[util.Pearson(feature=f, name=f) for f in features]]

    util.run(x=x, y=y, features=features, learner=learner, master=master, metrics=metrics, callbacks=callbacks,
             iterations=iterations, verbose=verbose, plot=plot)
