import importlib.resources

import numpy as np
import pandas as pd
from moving_targets.learners import LogisticRegression, TensorflowMLP
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import Accuracy, CrossEntropy
from moving_targets.util.scalers import Scaler

from experiments import util
from src.constraints import Smaller, Null
from src.master import Shape, CovarianceBasedMaster, DefaultMaster, ExplicitZerosMaster
from src.metrics import SoftShape

theta = 0.2
bins = [2, 3, 5, 10, 20]
kernels = [1, 2, 3, 5, 10]
iterations = 5
learner = 'lr'
master = 'covariance'
preprocess = True
weights = False
backend = GurobiBackend(time_limit=30)
verbose = 1
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data and learner
    with importlib.resources.path('data', 'adult.csv') as filepath:
        df = pd.read_csv(filepath)
        x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        if preprocess:
            x = Scaler('std').fit_transform(x)
    metrics = [Accuracy(), CrossEntropy(), util.Pearson(feature='age')]
    metrics += [util.BinnedDIDI(classification=False, protected='age', bins=b) for b in bins]

    # compute relative violation
    c = x[['age']].values
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
            lrn = TensorflowMLP(loss='mse', output_activation='sigmoid', hidden_units=[128, 128], epochs=300,
                                verbose=False, callbacks=[util.BestEpoch(monitor='loss')])
        elif 'lr' in learner:
            lrn = learner.split(' ')
            lrn = 1 if len(lrn) == 1 else int(lrn[1])
            lrn = LogisticRegression(polynomial=lrn, max_iter=10000)
        else:
            raise AssertionError(f"Unknown learner '{learner}'")

        # build master
        # TODO: which one is the parameter to increase numerical precision in Gurobi? The documentation mentions only
        #  the tolerances to be used as stopping criteria or to distinguish between integer and floating point values
        #  but these are explicitly indicated as not useful to increase precision. In fact, the documentation argues
        #  that the best way to avoid numerical errors is to reformulate the master (indeed, the covariance based
        #  formulation works very well even for higher-order polynomial kernels differently from both the default
        #  and the explicit zeros formulation, and it is even a lot faster).
        if master == 'default':
            shapes = [Shape('age', constraints=[None, cst, *[Null() for _ in range(1, k)]], kernel=k)]
            mst = DefaultMaster(shapes=shapes, backend=backend, binary=True)
        elif master == 'zeros':
            mst = ExplicitZerosMaster(feature='age', constraint=cst, degree=k, backend=backend, binary=True)
        elif master == 'covariance':
            mst = CovarianceBasedMaster(feature='age', constraint=cst, degree=k, backend=backend, binary=True)
        else:
            raise AssertionError(f"Unknown master '{master}'")

        # run experiment
        title = dict(title=f'KERNEL {k}')
        post = lambda wv: {f'w{degree}': weight for degree, weight in enumerate(wv)}
        util.run(x=x, y=y, features=['age'], learner=lrn, master=mst, iterations=iterations, verbose=verbose,
                 metrics=metrics + ([SoftShape('age', kernels=k, postprocessing=post)] if weights else []),
                 plot={**plot, **title} if isinstance(plot, dict) else (title if plot else False))
        print('----------------------------------------------------------------------------')
