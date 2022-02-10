import numpy as np
import pandas as pd
import seaborn as sns
from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE, CausalIndependence

from util import EffectCancellation

features = ['p1', 'p2', 'p3']
theta = 2 * np.ones_like(features, dtype='int')

if __name__ == '__main__':
    # config
    np.random.seed(0)
    sns.set_style('whitegrid')
    sns.set_context('notebook')

    # get data
    df = pd.read_csv('data/cmapps.csv')
    x, y = df.drop(columns=['src', 'machine', 'cycle', 'rul']), df['rul'].values

    # build model
    model = MACS(
        learner=LinearRegression(),
        master=EffectCancellation(backend=GurobiBackend(time_limit=30), features=features, theta=theta),
        metrics=[R2(), MSE(), CausalIndependence(features=features, aggregation=None)],
        init_step='pretraining'
    )

    # fit and examine
    history = model.fit(iterations=5, x=x, y=y)
    history.plot(figsize=(16, 9), orient_rows=True)