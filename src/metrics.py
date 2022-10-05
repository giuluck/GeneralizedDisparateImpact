from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from moving_targets.metrics import Metric, DIDI


class RegressionWeight(Metric):
    """Computes the weight(s) of a linear regression model trained uniquely on the given feature after it gets
    processed via a polynomial kernel with the given degree."""

    def __init__(self, feature: str, degree: int = 1, higher_orders: str = 'max', name: str = 'weight'):
        """
        :param feature:
            The name of the feature to inspect.

        :param degree:
            The degree of the polynomial kernel.

        :param higher_orders:
            The policy on how to present the weights of higher-order degrees if a kernel > 1 is passed. Options are
            'none' to get just the first-order degree,  'all' to get all the higher-order degrees, or 'max' to get
            just the maximal higher-order degree.

        :param name:
            The name of the metric.
        """
        super(RegressionWeight, self).__init__(name=name)

        # handle higher-orders postprocessing
        if degree == 1 or higher_orders == 'none':
            higher_orders = lambda w: w[1]
        elif higher_orders == 'max':
            higher_orders = lambda w: {'w1': w[1], 'w+': np.max(w[2:])}
        elif higher_orders == 'all':
            higher_orders = lambda w: {f'w{i + 1}': v for i, v in enumerate(w[1:])}
        else:
            raise AssertionError(f"Unknown higher orders option '{higher_orders}'")

        self.feature = feature
        """The name of the feature to inspect."""

        self.kernel: Callable = PolynomialFeatures(degree=degree, include_bias=True).fit_transform
        """The kernel function to transform each constrained feature."""

        self.higher_orders: Callable = higher_orders
        """The postprocessing function to return the higher-orders weights."""

    def __call__(self, x, y, p):
        a = self.kernel(x[[self.feature]])
        w, _, _, _ = np.linalg.lstsq(a, p, rcond=None)
        return self.higher_orders(np.abs(w))


class BinnedDIDI(DIDI):
    """Performs a binning operation on a continuous protected attribute then computes the DIDI on each bin."""

    def __init__(self,
                 classification: bool,
                 protected: str,
                 bins: int = 2,
                 percentage: bool = True,
                 name: str = 'didi'):
        """
        :param classification:
            Whether the DIDI is computed for a classification or a regression task.

        :param protected:
            The name of the protected feature.

        :param bins:
            The number of bins.

        :param percentage:
            Whether the DIDI is computed as an absolute or a relative value.

        :param name:
            The name of the metric.
        """
        super(BinnedDIDI, self).__init__(classification, protected, percentage, name=f'{name}_{bins}')
        self.bins = bins

    def __call__(self, x, y, p):
        x = x.copy()
        x[self.protected] = pd.qcut(x[self.protected], q=self.bins).cat.codes
        return super(BinnedDIDI, self).__call__(x, y, p)
