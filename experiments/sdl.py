import numpy as np
from moving_targets import MACS
from moving_targets.learners import LinearRegression, MultiLayerPerceptron
from moving_targets.masters.backends import GurobiBackend

from data.sdl import load_data
from experiments.util import config, ICECallback
from src.constraints import *
from src.master import ShapeConstrainedMaster
from src.metrics import SoftShape

# NOTE:
# moving targets prevents from replicating the experiments in the paper because while the authors of SDL change the
# weights of the mlp model via a custom loss function, we aim at regularizing them by modifying the distribution of
# the targets; still, since <x_1>, ..., <x_k> are correlated (they are all built as <u> + gaussian noise), if we change
# the distribution of <y> to match the expected shape of <x1>, then we force all the other shapes to be the same as
# that of <x1>, thus if we constraint them to be different there is no solution

samples = 50
features = 1
iterations = 5
degree = 2
thr = 1e-9
learner = 'mlp'
constraints = {
    # 'x1': [None, Lower(-2.0), Greater(1.0)],   # convex with inflection point on the right (-b / 2a > 0 -> b < 0)
    'x1': [None, Greater(2.0), Lower(-1.0)],   # concave with inflection point on the right (-b / 2a > 0 -> b > 0)
    # 'x1': [None, Greater(0.5), Smaller(thr)],  # increasing
    # 'x1': [None, Lower(-0.5), Smaller(thr)],   # decreasing
    # 'x1': [None, Smaller(0.1), Smaller(thr)],  # negligible
}
callbacks = np.arange(features) + 1
backend = GurobiBackend(time_limit=30)
m_stats = False
t_stats = False
plot = False

if __name__ == '__main__':
    config()

    # handle data and learner
    x, y = load_data(samples=samples, features=features)
    if learner == 'lr':
        learner = LinearRegression(polynomial=degree).fit(x, y)
    elif learner == 'mlp':
        learner = MultiLayerPerceptron(loss='mse', hidden_units=[64, 64], epochs=1000, verbose=False)
    else:
        raise AssertionError(f"Unknown learner '{learner}'")

    # build model
    model = MACS(
        init_step='pretraining',
        learner=learner,
        master=ShapeConstrainedMaster(constraints=constraints, kernels=degree, backend=backend, stats=m_stats),
        metrics=[SoftShape(feature=f'x{i}', kernels=degree, name=f'x{i}') for i in np.arange(features) + 1]
    )

    # handle callbacks and examine
    space = np.linspace(0, 1, samples)
    h = model.fit(x, y, iterations=iterations, callbacks=[ICECallback(feature=f'x{i}', space=space) for i in callbacks])
    if plot is True:
        h.plot()
    elif isinstance(plot, dict):
        h.plot(**plot)
