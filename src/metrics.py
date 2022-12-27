from math import pi, sqrt
from typing import Callable, Union, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PolynomialFeatures

from moving_targets.metrics import Metric, DIDI
from moving_targets.util import probabilities


class RegressionWeight(Metric):
    """Computes the weight(s) of a linear regression model trained uniquely on the given feature after it gets
    processed via a polynomial kernel with the given degree."""

    def __init__(self,
                 feature: str,
                 classification: bool,
                 degree: int = 1,
                 higher_orders: str = 'all',
                 percentage: bool = True,
                 name: str = 'weight'):
        """
        :param feature:
            The name of the feature to inspect.

        :param classification:
            Whether this is for a classification or a regression task (in the first scenario, binarize the predictions).

        :param degree:
            The degree of the polynomial kernel.

        :param higher_orders:
            The policy on how to present the weights of higher-order degrees if a kernel > 1 is passed. Options are
            'none' to get just the first-order degree,  'all' to get all the higher-order degrees, or 'max' to get
            just the maximal higher-order degree.

        :param percentage:
            Whether the weight is computed as an absolute or a relative value.

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

        self.feature: str = feature
        """The name of the feature to inspect."""

        self.classification: bool = classification
        """Whether this is for a classification or a regression task."""

        self.kernel: Callable = PolynomialFeatures(degree=degree, include_bias=True).fit_transform
        """The kernel function to transform each constrained feature."""

        self.higher_orders: Callable = higher_orders
        """The postprocessing function to return the higher-orders weights."""

        self.percentage: bool = percentage
        """Whether the HGR is computed as an absolute or a relative value."""

    def __call__(self, x, y, p):
        a = self.kernel(x[[self.feature]])
        p = probabilities.get_classes(p) if self.classification else p
        wp, _, _, _ = np.linalg.lstsq(a, p, rcond=None)
        if self.percentage:
            wy, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
            # when wy == 0.0, change wp float('inf') if its value is not null and then change wy to 1.0
            wp[np.logical_and(wy == 0.0, wp != 0.0)] = float('inf')
            wy[wy == 0.0] = 1.0
            return self.higher_orders(np.abs(wp / wy))
        else:
            return self.higher_orders(np.abs(wp))


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

        self.bins: int = bins
        """The number of bins."""

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
    def joint_2(x: torch.Tensor, y: torch.Tensor, damping: float = 1e-10, eps: float = 1e-9) -> torch.Tensor:
        # add an eps value to avoid nan vectors in case of very degraded solutions
        x = (x - x.mean()) / (x.std(dim=None) + eps)
        y = (y - y.mean()) / (y.std(dim=None) + eps)
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

    @staticmethod
    def hgr(x: torch.Tensor, y: torch.Tensor, chi2: bool) -> torch.Tensor:
        h2d = HGR.joint_2(x, y)
        marginal_x = h2d.sum(dim=1).unsqueeze(1)
        marginal_y = h2d.sum(dim=0).unsqueeze(0)
        q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
        if chi2:
            return (q ** 2).sum(dim=[0, 1]) - 1.0
        else:
            return torch.linalg.svd(q)[1][1]

    def __init__(self, features: Union[str, List[str]], percentage: bool = True, chi2: bool = False, name: str = 'hgr'):
        """
        :param features:
            The name of the features to inspect.

        :param percentage:
            Whether the HGR is computed as an absolute or a relative value.

        :param chi2:
            Whether to return the chi^2 approximation of the HGR or its actual value.

        :param name:
            The name of the metric.
        """
        super(HGR, self).__init__(name=name)

        self.chi2: bool = chi2
        """Whether to return the chi^2 approximation of the HGR or its actual value."""

        self.percentage: bool = percentage
        """Whether the HGR is computed as an absolute or a relative value."""

        self.features: Union[str, List[str]] = features
        """The name of the feature to inspect."""

    def _compute_val(self, z: torch.Tensor, y: torch.Tensor, p: torch.Tensor):
        hgr_p = HGR.hgr(p, z, chi2=self.chi2).item()
        if self.percentage:
            hgr_y = HGR.hgr(y, z, chi2=self.chi2).item()
            if hgr_y == 0.0:
                return 0.0 if hgr_p == 0.0 else float('inf')
            else:
                return hgr_p / hgr_y
        else:
            return hgr_p

    def __call__(self, x, y, p):
        p = torch.tensor(p, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        if isinstance(self.features, str):
            z = torch.tensor(x[self.features].values, dtype=torch.float)
            return self._compute_val(z, y, p)
        metrics = {}
        for feature in self.features:
            z = torch.tensor(x[feature].values, dtype=torch.float)
            metrics[feature] = self._compute_val(z, y, p)
        return metrics
