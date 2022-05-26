from moving_targets.learners import LinearRegression, TensorflowMLP
from moving_targets.metrics import MSE, R2

from data.sdl import load_data
from experiments import util
from src.constraints import *
from src.master import DefaultMaster, Shape
from src.metrics import SoftShape

samples = 50
features = 1
iterations = 5
thr = 1e-9
learner = 'lr 1'
shapes = [
    # Shape('x1', constraints=[None, Lower(-2.0), Greater(1.0)], kernel=2),  # convex (inflection point on the right)
    # Shape('x1', constraints=[None, Greater(1.0), Lower(-1.0)], kernel=2),  # concave (inflection point on the right)
    # Shape('x1', constraints=[None, Greater(0.5), Smaller(thr)], kernel=2),  # increasing
    # Shape('x1', constraints=[None, Lower(-0.5), Smaller(thr)], kernel=2),  # decreasing
    # Shape('x1', constraints=[None, Smaller(0.1), Smaller(thr)], kernel=2),  # negligible
    Shape('x1', constraints=[None, Smaller(thr)], kernel=1)
]
backend = 'gurobi'
verbose = True
callbacks = ['dist', 'ice']
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data
    x, y = load_data(samples=samples, features=features, noise=0.01)
    x, y = util.augment_data(x, y)

    # build learner
    if learner == 'mlp':
        learner = TensorflowMLP(loss='mse', hidden_units=[64, 64], epochs=1000, verbose=False)
    elif 'lr' in learner:
        learner = learner.split(' ')
        learner = 1 if len(learner) == 1 else int(learner[1])
        learner = LinearRegression(polynomial=learner)
    else:
        raise AssertionError(f"Unknown learner '{learner}'")

    # build master and metrics
    master = DefaultMaster(shapes=shapes, backend=backend)
    metrics = [MSE(), R2(), *[SoftShape(feature=f'x{i + 1}', kernels=2) for i in range(features)]]

    util.run(x=x, y=y, features=[f'x{i + 1}' for i in range(features)], learner=learner, master=master, metrics=metrics,
             callbacks=callbacks, iterations=iterations, verbose=verbose, plot=plot)

    # TODO: remove debug
    # Problem: we can effectively change the output shape but we have *NO GUARANTEES* on what the model will learn if
    # there is any kind of correlation between two input features. E.g., in this case, we have <u> ~ <x1>, thus by
    # changing the output distribution, the constraints on <x1> will be reflected on <u> as well, which should be fine
    # by the way. The problem is that, if we aim at cancelling <x1> (and, as a consequence, cancelling <u>) the model
    # may still learn a weight vector [w, -w], which will cancel out the effect but it is semantically wrong.
    #   > Of course, if the input distribution does not vary that much, i.e., the correlation between <u> and <x1>
    #     remains in the test set as well, the problem only regards the semantic of the mode, while predictions are
    #     guaranteed to be correct; this is not the case if the test distribution varies.
    #   > Would it make sense to move the *INPUT DATA* instead of the targets? In this way, we could have total control
    #     on the shape of <x1>; however, this changes will not affect the relation between <u> and the target, and it
    #     should be investigated when and how this behaviour may be preferred.
    if isinstance(learner, LinearRegression):
        print(learner.model.coef_)
