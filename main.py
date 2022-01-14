import pandas as pd
import seaborn as sns
from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE

from util import *

FEATURES = ['p1', 'p2']
THETA = 0

if __name__ == '__main__':
    # config
    np.random.seed(0)
    sns.set_style('whitegrid')
    sns.set_context('notebook')

    # get data
    df = pd.read_csv('data/cmapps.csv')
    x, y = df.drop(columns=['src', 'machine', 'cycle', 'p3', 'rul']), df['rul']
    features = x.columns.get_indexer(FEATURES)
    theta = THETA * np.ones_like(features)

    # build model
    metrics = [PearsonCorrelation(feature=i, name=f'pearson_{f}') for i, f in zip(features, FEATURES)]
    model = MACS(
        learner=LinearRegression(),
        master=CausalExclusionCovariance(backend=GurobiBackend(time_limit=30), features=features, theta=theta),
        metrics=[R2(), MSE(), ZeroWeightCorrelation(features=features, name='constraint'), *metrics],
        init_step='pretraining',
        stats=True
    )

    # fit and examine
    history = model.fit(iterations=5, x=x.values, y=y.values)
    history.plot(figsize=(16, 9), orient_rows=True)
