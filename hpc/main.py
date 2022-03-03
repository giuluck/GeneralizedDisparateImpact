import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE

from hpc.synthetic_data import generator
from hpc.util import EffectCancellation, CausalIndependence, Pearson

dataset = 'synthetic'
iterations = 20
theta = 1e-3
pearson = True
time_limit = 10
m_stats = False
t_stats = False

if __name__ == '__main__':
    # config
    np.random.seed(0)
    sns.set_style('whitegrid')
    sns.set_context('notebook')

    if dataset == 'cmapps':
        with importlib.resources.path('data', 'cmapps.csv') as filepath:
            df = pd.read_csv(filepath)
        x, y = df.drop(columns=['src', 'machine', 'cycle', 'rul']), df['rul'].values
        features = ['p1', 'p2', 'p3']
        metrics = [CausalIndependence(f, name=f'{f}_weight') for f in features]
    elif dataset == 'synthetic':
        df = generator(fy=lambda qh: 3 + np.exp(qh), fq=lambda p, qh: qh + 2 * p).generate()
        x, y = df[['p', 'q']], df['y'].values
        features = ['p']
        metrics = [CausalIndependence('p', name='p_weights')]
    else:
        raise AssertionError(f"Unknown dataset '{dataset}'")

    # build model
    learner = LinearRegression()
    master = EffectCancellation(
        theta=theta,
        features=features,
        backend=GurobiBackend(time_limit=time_limit),
        stats=m_stats,
        ub=float('inf'),
        lb=-float('inf'),
        loss='mse'
    )
    model = MACS(
        init_step='pretraining',
        learner=learner,
        master=master,
        metrics=[R2(), MSE(), *metrics] + ([Pearson(f, name=f'{f}_pearson') for f in features] if pearson else []),
        stats=t_stats
    )

    # fit and examine
    history = model.fit(iterations=iterations, x=x, y=y)
    history.plot(figsize=(16, 9), orient_rows=True)
