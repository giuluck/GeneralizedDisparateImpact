import numpy as np
from moving_targets.learners import LinearRegression, TensorflowMLP
from moving_targets.metrics import MSE, R2

from data.sdl import load_data
from experiments import util
from src.constraints import *
from src.master import DefaultMaster, Shape
from src.metrics import SoftShape

features = 1
samples = 50
augmentation = [None, 200]
iterations = 5
thr = 1e-9
learner = 'lr 2'
shapes = [
    # Shape('x1', constraints=[None, Negative(), Greater(4.0)], kernel=2),  # convex (inflection point on the right)
    # Shape('x1', constraints=[None, Positive(), Lower(-4.0)], kernel=2),  # concave (inflection point on the right)
    # Shape('x1', constraints=[None, Greater(3.0), Smaller(thr)], kernel=2),  # increasing
    # Shape('x1', constraints=[None, Lower(-3.0), Smaller(thr)], kernel=2),  # decreasing
    # Shape('x1', constraints=[None, Smaller(thr), Smaller(thr)], kernel=2),  # negligible
    Shape('x1', constraints=[None, Smaller(thr)], kernel=1)  # negligible (first order kernel)
]
backend = 'gurobi'
verbose = 1
callbacks = ['ice']
plot = False

if __name__ == '__main__':
    for aug in augmentation:
        # handle data
        x, y = load_data(samples=samples, features=features, augmentation=aug, noise=0.01)

        # build learner
        if learner == 'mlp':
            lrn = TensorflowMLP(loss='mse', hidden_units=[8, 8], epochs=1000, verbose=False, mask=np.nan)
        elif 'lr' in learner:
            lrn = learner.split(' ')
            lrn = 1 if len(lrn) == 1 else int(lrn[1])
            lrn = LinearRegression(polynomial=lrn, mask=np.nan)
        else:
            raise AssertionError(f"Unknown learner '{learner}'")

        # build master and metrics
        mst = DefaultMaster(shapes=shapes, backend=backend)
        mtr = [MSE(), R2(), *[SoftShape(feature=f'x{i + 1}', kernels=2) for i in range(features)]]

        util.run(x=x, y=y, features=['u'] + [f'x{i + 1}' for i in range(features)], learner=lrn, master=mst,
                 metrics=mtr, callbacks=callbacks, iterations=iterations, verbose=verbose, plot=plot)
