import importlib.resources
import os

import pandas as pd
from moving_targets import MACS
from moving_targets.learners import LogisticRegression
from moving_targets.metrics import CrossEntropy, Accuracy, AUC

from experiments import util
from src.constraints import Smaller
from src.master import DefaultMaster, Shape

dataset = 0
iterations = 5
theta = 1e-3
degree = 1
learner = 'lr'
backend = 'gurobi'
verbose = True
callbacks = []
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    # handle data
    with importlib.resources.path('data', 'marconi') as filepath:
        files = [filepath.joinpath(gzip) for gzip in os.listdir(filepath)]
    df = pd.read_parquet(files[dataset]).rename(columns={'timestamp': 'index'}).set_index('index')

    # TODO: keeping avg information only for the moment, may need to change it
    x = df.drop(columns=['label', 'New_label']).astype('float32')
    x = x[[f for f in x.columns if 'avg:' in f]].rename(columns=lambda f: f[4:])
    y = df['New_label'].astype('category').cat.codes.values

    # TODO: which features to exclude?
    #   > 'cpu_idle' and 'cpu_aidle' (and btw what is the difference?)
    #   > 'load_one', 'load_five' and 'load_fifteen' (where are the others?)
    features = [f for f in x.columns if 'load' in f]

    # build learner
    if learner == 'mlp':
        raise AssertionError("Need to test configuration before")
        # learner = TensorflowMLP(output_activation='sigmoid', loss='binary_crossentropy')
    elif 'lr' in learner:
        learner = LogisticRegression().fit(x, y)
    else:
        raise AssertionError(f"Unknown learner '{learner}'")

    # build master
    shapes = [Shape(f, constraints={i + 1: Smaller(theta) for i in range(degree)}, kernel=degree) for f in features]
    master = DefaultMaster(shapes=shapes, backend=backend, loss='hd', binary=True)
    metrics = [CrossEntropy(), Accuracy(), AUC(), *[util.Pearson(f, name=f) for f in features]]
    model = MACS(init_step='pretraining', learner=learner, master=master, metrics=metrics)

    util.run(x=x, y=y, features=features, learner=learner, master=master, metrics=metrics, callbacks=callbacks,
             iterations=iterations, verbose=verbose, plot=plot)
