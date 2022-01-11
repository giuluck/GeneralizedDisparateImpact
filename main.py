import numpy as np
import pandas as pd
from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE

from util import CausalExclusion, PearsonCorrelation

excluded_features = ['p1', 'p2']
theta = 1.0 * np.ones_like(excluded_features, dtype=float)

if __name__ == '__main__':
    # get data
    df = pd.read_csv('data/cmapps.csv')
    # df = df[df['src'] == 'train_FD001']

    x = df.drop(columns=['src', 'machine', 'cycle', 'p3', 'rul'])
    y = df['rul'].values

    # build model
    model = MACS(
        learner=LinearRegression(),
        master=CausalExclusion(backend=GurobiBackend(time_limit=10), excluded_features=excluded_features, theta=theta),
        metrics=[R2(), MSE(), PearsonCorrelation(feature='p1'), PearsonCorrelation(feature='p2')],
        stats=True
    )
    history = model.fit(iterations=10, x=x, y=y)
    history.plot(figsize=(16, 9), orient_rows=True)
