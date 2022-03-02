import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from moving_targets import MACS
from moving_targets.learners import LinearRegression, LogisticRegression
from moving_targets.masters.backends import CplexBackend
from moving_targets.metrics import Accuracy, CrossEntropy, DIDI, R2, MSE

from fairness.util import DidiMaster, LrMaster, LrWeight

classification = False
master = 'lr'
theta = 0.2
loss = 'mse'
alpha = 'harmonic'
iterations = 5
backend = CplexBackend(time_limit=30)
m_stats = True
t_stats = False
verbose = True
plot = dict()  # num_subplots=4, orient_rows=True)
metrics = [
    LrWeight(protected='race', aggregate=False, percentage=True, name='percentage_weight'),
    DIDI(protected='race', classification=classification, percentage=True, name='percentage_didi'),
    LrWeight(protected='race', aggregate=False, percentage=False, name='absolute_weight'),
    DIDI(protected='race', classification=classification, percentage=False, name='absolute_didi')
]

if __name__ == '__main__':
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    np.random.seed(0)

    # handle data and learner
    if classification:
        with importlib.resources.path('data', 'adult.csv') as filepath:
            df = pd.read_csv(filepath)
            x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        learner = LogisticRegression(max_iter=10000, x_scaler='std')
        metrics = [Accuracy(), CrossEntropy(), *metrics]
    else:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            df = pd.read_csv(filepath)
            x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        learner = LinearRegression(x_scaler='std', y_scaler='norm')
        metrics = [R2(), MSE(), *metrics]

    # handle master
    if master == 'didi':
        master = DidiMaster(classification, backend=backend, loss=loss, alpha=alpha, stats=m_stats)
    elif master == 'lr':
        master = LrMaster(classification, theta=theta, backend=backend, loss=loss, alpha=alpha, stats=m_stats)
    else:
        raise AssertionError(f"Unknown master '{master}")

    # handle macs model
    model = MACS(init_step='pretraining', learner=learner, master=master, metrics=metrics, stats=t_stats)
    history = model.fit(x=x, y=y, iterations=iterations, verbose=verbose)
    if plot is True:
        history.plot()
    elif isinstance(plot, dict):
        history.plot(**plot)
