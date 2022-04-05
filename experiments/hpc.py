import importlib.resources

import numpy as np
import pandas as pd
import seaborn as sns
from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE

from data.synthetic import generator
from experiments.util import Pearson, config
from src.constraints import Smaller
from src.master import ShapeConstrainedMaster
from src.metrics import SoftShape

dataset = 'cmapps'
iterations = 10
theta = 1e-3
degree = 1
shape = True
pearson = True
backend = GurobiBackend(time_limit=10)
m_stats = False
t_stats = False

if __name__ == '__main__':
    config()

    # handle data
    if dataset == 'cmapps':
        with importlib.resources.path('data', 'cmapps.csv') as filepath:
            df = pd.read_csv(filepath)
        x, y = df.drop(columns=['src', 'machine', 'cycle', 'rul']), df['rul'].values
        features = ['p1', 'p2', 'p3']
    elif dataset == 'synthetic':
        df = generator(fy=lambda qh: 3 + np.exp(qh), fq=lambda p, qh: qh + 2 * p).generate()
        x, y = df[['p', 'q']], df['y'].values
        features = ['p']
    else:
        raise AssertionError(f"Unknown dataset '{dataset}'")

    # build model
    learner = LinearRegression()
    master = ShapeConstrainedMaster(constraints={f: {i + 1: Smaller(theta) for i in range(degree)} for f in features},
                                    backend=backend,
                                    stats=m_stats)
    metrics = [R2(), MSE()]
    if shape:
        postprocessing = lambda w: {f'w{i + 1}': v for i, v in enumerate(w[1:])}
        metrics += [SoftShape(feature=f, postprocessing=postprocessing, name=f) for f in features]
    if pearson:
        metrics += [Pearson(f, name=f'{f}_pearson') for f in features]
    model = MACS(
        init_step='pretraining',
        learner=learner,
        master=master,
        metrics=metrics,
        stats=t_stats
    )

    # fit and examine
    history = model.fit(iterations=iterations, x=x, y=y)
    history.plot(figsize=(16, 9), orient_rows=True)
