from math import pi, sqrt
from typing import Callable

import numpy as np
import pandas as pd
import torch
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


class HGR(Metric):
    """Torch-based implementation of the HGR metric obtained from the official repository of "Fairness-Aware Learning
    for Continuous Attributes and Treatments" (https://github.com/criteo-research/continuous-fairness/)."""

    class KDE:
        """A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization. Keep in mind that
        KDE are not scaling well with the number of dimensions and this implementation is not really optimized..."""

        def __init__(self, x_train):
            n, d = x_train.shape
            self.n = n
            self.d = d
            self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            self.std = self.bandwidth
            self.train_x = x_train

        @staticmethod
        def _unsqueeze_multiple_times(inp, axis, times):
            out = inp
            for i in range(times):
                out = out.unsqueeze(axis)
            return out

        def pdf(self, x):
            s = x.shape
            d = s[-1]
            s = s[:-1]
            assert d == self.d
            data = x.unsqueeze(-2)
            train_x = self._unsqueeze_multiple_times(self.train_x, 0, len(s))
            # noinspection PyTypeChecker
            gaussian_values = torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
            pdf_values = gaussian_values.mean(dim=-1) / sqrt(2 * pi) / self.bandwidth
            return pdf_values

    @staticmethod
    def joint_2(x, y, damping=1e-10):
        # the check on x_std and y_std to be != 0 allows to avoid nan vectors in case of very degraded solutions
        x_std, y_std = x.std(), y.std()
        x = (x - x.mean()) / (1 if x_std == 0.0 else x_std)
        y = (y - y.mean()) / (1 if y_std == 0.0 else y_std)
        data = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)
        joint_density = HGR.KDE(data)
        n_bins = int(min(50, 5. / joint_density.std))
        x_centers = torch.linspace(-2.5, 2.5, n_bins)
        y_centers = torch.linspace(-2.5, 2.5, n_bins)
        xx, yy = torch.meshgrid([x_centers, y_centers], indexing='ij')
        grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
        h2d = joint_density.pdf(grid) + damping
        h2d /= h2d.sum()
        return h2d

    def __init__(self, feature: str, percentage: bool = True, name: str = 'hgr'):
        """
        :param feature:
            The name of the feature to inspect.

        :param percentage:
            Whether the HGR is computed as an absolute or a relative value.

        :param name:
            The name of the metric.
        """
        super(HGR, self).__init__(name=name)

        self.percentage: bool = percentage
        """Whether the HGR is computed as an absolute or a relative value."""

        self.feature: str = feature
        """The name of the feature to inspect."""

    def __call__(self, x, y, p):
        p = torch.tensor(p, dtype=torch.float)
        z = torch.tensor(x[self.feature].values, dtype=torch.float)
        h2d = self.joint_2(p, z)
        marginal_p = h2d.sum(dim=1).unsqueeze(1)
        marginal_z = h2d.sum(dim=0).unsqueeze(0)
        q = h2d / (torch.sqrt(marginal_p) * torch.sqrt(marginal_z))
        hgr_p = torch.svd(q)[1].numpy()[1]
        if self.percentage:
            y = torch.tensor(y, dtype=torch.float)
            h2d = self.joint_2(y, z)
            marginal_y = h2d.sum(dim=1).unsqueeze(1)
            marginal_z = h2d.sum(dim=0).unsqueeze(0)
            q = h2d / (torch.sqrt(marginal_y) * torch.sqrt(marginal_z))
            hgr_y = torch.svd(q)[1].numpy()[1]
            if hgr_y == 0.0:
                return 0.0 if hgr_p == 0.0 else float('inf')
            else:
                return hgr_p / hgr_y
        else:
            return hgr_p
