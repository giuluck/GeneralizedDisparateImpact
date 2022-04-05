import numpy as np
import pandas as pd


def load_data(samples=200, features=5):
    """Creates a synthetic dataset as described in Section 5.1 of "Sensitivity Direction Learning with Neural Networks
    Using Domain Knowledge as Soft Shape Constraints":

    In order to clearly confirm that SDL actually works to learn about the model with user-specified relationships, the
    dataset should be have relationships differed from user-specified relationships. Considering this requirement, we
    generated the dataset which consists of 200 instances and six columns as following steps: (i) the 1-d vector U is
    defined to be between 0.0 and 1.0 in 0.005 steps, and (ii) y is set to U, and each feature (x1 ~ x5) is set to the
    sum of U and Gaussian noise (sigma = 0.1). As is clear from above steps, there are only strong correlations between
    each of five features (x1 ~ x5) and the output y. Therefore it can be said that if a model is learned without shape
    constraints, there is very little possibility that the trained model has convexity relationships.
    """
    u = np.linspace(0, 1, num=samples, endpoint=False)
    rng = np.random.default_rng(0)
    return pd.DataFrame().from_dict({'u': u, **{f'x{i + 1}': u + rng.normal(0, 0.1) for i in range(features)}}), u
