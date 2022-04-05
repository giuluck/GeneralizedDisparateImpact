from typing import Callable, Union

import numpy as np
from moving_targets.metrics import Metric
from sklearn.preprocessing import PolynomialFeatures


class SoftShape(Metric):
    """Measures the shape of a feature as the weight(s) of the linear regression model trained on (a kernel of) that
    feature with respect to the target.

    The kernel function allows to inspect non-linear shapes of that feature. For example, if we want to inspect the
    concavity we may use a polynomial kernel with degree two (i.e., k(x) -> 1 + x + x^2), and eventually pass a
    postprocessing function that returns only the weight associated to the third feature, which is indeed x^2.
    """

    def __init__(self,
                 feature: str,
                 kernels: Union[None, int, Callable] = 1,
                 postprocessing: Union[None, Callable] = None,
                 name='shape'):
        """
        :param feature:
            The name of the feature to inspect.

        :param kernels:
            If None is passed, the input feature is not transformed. If an integer is passed, the input feature is
            transformed via a polynomial kernel up to the given degree, with bias included (the default behaviour is,
            indeed, to transform feature f into [1|f]). Otherwise, an explicit kernel function must be passed.

        :param postprocessing:
            A function f(w) -> w' that processes the weights of the linear regression model. It may either return a
            dictionary associating a key to each weight (or a subset of such) or a single scalar representing the
            aggregated metric value. If None, returns all the weights related to each transformation of the feature
            labelled as 'w0', 'w1' and so on (actually, if the kernel is polynomial, automatically removes the
            intercept from the list), or a single scalar value if the kernel is such that a single feature is used as
            input of the linear regression model.

        :param name:
            The name of the metric.
        """
        super(SoftShape, self).__init__(name=name)

        # handle default postprocessing function
        # - if linear kernel, returns the weight of the feature
        # - if polynomial kernel, returns each weight apart from the intercept
        # - if custom or no kernel, returns either a single scalar or a dictionary of features
        if postprocessing is None:
            if kernels == 1:
                postprocessing = lambda w: w[1]
            elif isinstance(kernels, int):
                postprocessing = lambda w: {f'w{i + 1}': v for i, v in enumerate(w[1:])}
            else:
                postprocessing = lambda w: w[0] if w.size == 1 else {f'w{i}': v for i, v in enumerate(w)}

        # handle default kernel function
        # - if no kernel, uses the identity function (i.e., simply retrieves the feature)
        # - if int, builds a polynomial kernel and applies it
        if kernels is None:
            kernels = lambda x, f: x[[f]]
        elif isinstance(kernels, int):
            kernel = PolynomialFeatures(degree=kernels, include_bias=True)
            kernels = lambda x, f: kernel.fit_transform(x[[f]])

        self.feature = feature
        """The name of the feature to inspect."""

        self.kernels: Callable = kernels
        """The kernel function to transform each constrained feature."""

        self.postprocessing: Callable = postprocessing
        """The postprocessing function to return the selected weight(s)."""

    def __call__(self, x, y, p):
        a = self.kernels(x, self.feature)
        w, _, _, _ = np.linalg.lstsq(a, p, rcond=None)
        return self.postprocessing(w)
