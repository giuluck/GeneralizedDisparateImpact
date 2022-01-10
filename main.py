import numpy as np
from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE, Metric
from scipy.stats import pearsonr

from src.master import CausalExclusion
from src.util import load_cmapss_data


class PearsonCorrelation(Metric):
    def __init__(self, feature, name):
        super(PearsonCorrelation, self).__init__(name=name)
        self.feature = feature

    def __call__(self, x, y, p):
        return pearsonr(x[self.feature], p)[0]


if __name__ == '__main__':
    df = load_cmapss_data(data_folder='./data')
    df = df[df['src'] == 'train_FD001']
    x = df.drop(columns=['src', 'machine', 'cycle', 'p3', 'rul'])
    y = df['rul'].values

    excluded_features = ['p1', 'p2']
    theta = 0.00001 * np.ones(len(excluded_features))
    model = MACS(
        learner=LinearRegression(),
        master=CausalExclusion(backend=GurobiBackend(time_limit=10), excluded_features=excluded_features, theta=theta),
        metrics=[R2(), MSE()] + [PearsonCorrelation(feature=f, name=f'pearson_{f}') for f in excluded_features],
        stats=True
    )
    history = model.fit(iterations=10, x=x, y=y)
    history.plot(figsize=(16, 9), orient_rows=True)
