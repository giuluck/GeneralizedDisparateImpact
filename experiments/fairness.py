import importlib.resources

import pandas as pd
from moving_targets import MACS
from moving_targets.learners import LinearRegression, LogisticRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import Accuracy, CrossEntropy, DIDI, R2, MSE

from experiments.util import config
from src.constraints import Smaller
from src.master import ShapeConstrainedMaster, Shape
from src.metrics import SoftShape

classification = False
theta = 0.2
iterations = 5
backend = GurobiBackend(time_limit=30)
m_stats = False
t_stats = False
verbose = True
plot = dict(features=None, excluded=['adjusted/*'])

if __name__ == '__main__':
    config()

    # handle data and learner
    didi = DIDI(protected='race', classification=False, percentage=True, name='didi')
    if classification:
        with importlib.resources.path('data', 'adult.csv') as filepath:
            df = pd.read_csv(filepath)
            x, y = df.drop('income', axis=1), df['income'].astype('category').cat.codes.values
        learner = LogisticRegression(max_iter=10000, x_scaler='std')
        metrics = [Accuracy(), CrossEntropy(), didi]
    else:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            df = pd.read_csv(filepath)
            x, y = df.drop('violentPerPop', axis=1), df['violentPerPop'].values
        learner = LinearRegression(x_scaler='std', y_scaler='norm')
        metrics = [R2(), MSE(), didi]

    # handle shape-related metrics and compute (relative) accepted violations
    violations = {}
    for c in x.columns:
        if c.startswith('race'):
            metric = SoftShape(feature=c, name=f'w_{c.replace("race_", "")}')
            metrics.append(metric)
            violations[c] = abs(theta * metric(x, y, y))

    # handle master (constraint each protected feature to have first-order degree weight lower than the violation)
    shapes = [Shape(p, constraints=[None, Smaller(v)], kernel=1) for p, v in violations.items()]
    master = ShapeConstrainedMaster(shapes=shapes, binary=classification, backend=backend, stats=m_stats)

    # handle macs model
    model = MACS(init_step='pretraining', learner=learner, master=master, metrics=metrics, stats=t_stats)
    history = model.fit(x=x, y=y, iterations=iterations, verbose=verbose)
    if plot is True:
        history.plot()
    elif isinstance(plot, dict):
        history.plot(**plot)
