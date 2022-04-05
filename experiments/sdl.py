import numpy as np
from moving_targets import MACS
from moving_targets.learners import LinearRegression, MultiLayerPerceptron
from moving_targets.masters.backends import GurobiBackend

from data.sdl import load_data
from experiments.util import config, ICECallback
from src.constraints import Greater, Lower, Smaller
from src.master import ShapeConstrainedMaster

samples = 200
features = 5
iterations = 5
degree = 2
threshold = 1e-3
learner = 'mlp'
constraints = {
    'x1': [None, Smaller(threshold), Greater(0.2)],  # convex
    'x2': [None, Smaller(threshold), Lower(-0.6)],   # concave
    'x3': [None, Greater(0.7), Smaller(threshold)],  # increasing
    'x4': [None, Lower(-0.3), Smaller(threshold)],   # decreasing
    'x5': [None, Smaller(0.1), Smaller(0.1)],        # negligible
}
callbacks = [1, 2, 3, 4, 5]
backend = GurobiBackend(time_limit=10)
m_stats = False
t_stats = False

if __name__ == '__main__':
    config()

    # handle data and model
    x, y = load_data(samples=samples, features=features)
    if learner == 'lr':
        learner = LinearRegression(polynomial=degree).fit(x, y)
    elif learner == 'mlp':
        learner = MultiLayerPerceptron(loss='mse', hidden_units=[64, 64], epochs=1000, verbose=False)
    else:
        raise AssertionError(f"Unknown learner '{learner}'")
    master = ShapeConstrainedMaster(constraints=constraints, kernels=degree, backend=backend, stats=m_stats)
    model = MACS(
        init_step='pretraining',
        learner=learner,
        master=master
    )

    # handle callbacks and examine
    space = np.linspace(0, 1, samples)
    model.fit(x, y, iterations=iterations, callbacks=[ICECallback(feature=f'x{i}', space=space) for i in callbacks])
