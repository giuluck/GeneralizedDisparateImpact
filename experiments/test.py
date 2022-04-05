import numpy as np
import seaborn as sns
from moving_targets import MACS
from moving_targets.learners import LinearRegression
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import R2, MSE

from data.synthetic import generator
from experiments.util import Pearson, config
from src.constraints import Smaller
from src.master import ShapeConstrainedMaster


class CustomLearner(LinearRegression):
    def __init__(self, expected_weights):
        super(CustomLearner, self).__init__(stats=True)
        self.expected_weights = np.array(expected_weights)

    def on_training_end(self, macs, x, y, p, val_data):
        weights = np.concatenate(([macs.learner.model.intercept_], macs.learner.model.coef_))
        self.log(**{f'weight/{i}': weight for i, weight in enumerate(weights)})
        self.log(**{f'difference/{i}': diff for i, diff in enumerate(weights - self.expected_weights)})


if __name__ == '__main__':
    config()

    # build dataset
    a, b, c, d, e = [2, 5, 1, 5, 1]
    w = [e - c * d / b, -a * d / b, d / b]
    df = generator(fq=lambda p, qh: a * p + b * qh + c, fy=lambda qh: d * qh + e).generate()
    print('Covariance(p, q_hat):', np.cov(df['p'], df['q_hat'])[0, 1])
    print('\n\n\n\n')

    # build model
    model = MACS(
        init_step='pretraining',
        learner=CustomLearner(expected_weights=w),
        master=ShapeConstrainedMaster(
            constraints={'p': [None, Smaller(1e-9)]},
            backend=GurobiBackend(time_limit=10),
            binary=False,
            stats=False
        ),
        metrics=[MSE(), R2(), Pearson(feature='p')]
    )

    # fit and examine
    history = model.fit(iterations=10, x=df[['p', 'q']], y=df['y'].values, verbose=True)
    history.plot(figsize=(16, 9), orient_rows=True)
