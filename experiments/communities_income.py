import importlib.resources

import numpy as np
import pandas as pd
from moving_targets.learners import LinearRegression, TensorflowMLP
from moving_targets.metrics import R2, MSE
from moving_targets.util.scalers import Scaler

from experiments import util
from src.constraints import Smaller, Null
from src.master import Shape, CovarianceBasedMaster, DefaultMaster, ExplicitZerosMaster
from src.metrics import SoftShape

theta = 0.2
bins = [2, 3, 5, 10, 20]
kernels = [1, 2, 3, 5, 10]
iterations = 5
learner = 'lr 2'
master = 'covariance'
reg_1 = None
reg_2 = None
reg_inf = None
preprocess = True
weights = False
backend = 'gurobi'
verbose = 1
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data
    with importlib.resources.path('data', 'communities.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        if preprocess:
            x, y = Scaler('std').fit_transform(x), Scaler('norm').fit_transform(y)
    metrics = [R2(), MSE(), util.Pearson(feature='perCapInc')]
    metrics += [util.BinnedDIDI(classification=False, protected='perCapInc', bins=b) for b in bins]

    # compute relative violation
    c = x[['perCapInc']].values
    c = np.concatenate((np.ones_like(c), c), axis=1)
    w, _, _, _ = np.linalg.lstsq(c, y, rcond=None)
    v = abs(theta * w[1])
    cst = Smaller(v)

    # test different polynomial kernels
    print('----------------------------------------------------------------------------')
    for k in kernels:
        print(f'KERNEL {k}')
        # build learner
        if learner == 'mlp':
            # best epoch callback is needed due to massive loss fluctuations
            lrn = TensorflowMLP(loss='mse', hidden_units=[128, 128], epochs=300, verbose=False,
                                callbacks=[util.BestEpoch(monitor='loss')])
        elif 'lr' in learner:
            lrn = learner.split(' ')
            lrn = 1 if len(lrn) == 1 else int(lrn[1])
            lrn = LinearRegression(polynomial=lrn)
        else:
            raise AssertionError(f"Unknown learner '{learner}'")

        # build master
        if master == 'default':
            shapes = [Shape('perCapInc', constraints=[None, cst, *[Null() for _ in range(1, k)]], kernel=k)]
            mst = DefaultMaster(shapes=shapes, backend=backend, reg_1=reg_1, reg_2=reg_2, reg_inf=reg_inf, binary=False)
        elif master == 'zeros':
            mst = ExplicitZerosMaster(feature='perCapInc', constraint=cst, degree=k, backend=backend,
                                      reg_1=reg_1, reg_2=reg_2, reg_inf=reg_inf, binary=False)
        elif master == 'covariance':
            mst = CovarianceBasedMaster(feature='perCapInc', constraint=cst, degree=k, backend=backend,
                                        reg_1=reg_1, reg_2=reg_2, reg_inf=reg_inf, binary=False)
        else:
            raise AssertionError(f"Unknown master '{master}'")

        # run experiment
        title = dict(title=f'KERNEL {k}')
        post = lambda wv: {f'w{degree}': weight for degree, weight in enumerate(wv)}
        util.run(x=x, y=y, features=['perCapInc'], learner=lrn, master=mst, iterations=iterations, verbose=verbose,
                 metrics=metrics + ([SoftShape('perCapInc', kernels=k, postprocessing=post)] if weights else []),
                 plot={**plot, **title} if isinstance(plot, dict) else (title if plot else False))
        print('----------------------------------------------------------------------------')
