import random

import numpy as np
import seaborn as sns
import tensorflow as tf

from moving_targets import MACS


def run(x, y, master, learner, metrics=(), callbacks=(), iterations=5, verbose=True, plot=True, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    model = MACS(init_step='pretraining', learner=learner, master=master, metrics=metrics)
    history = model.fit(x=x, y=y, iterations=iterations, callbacks=callbacks, verbose=verbose)
    if isinstance(plot, dict):
        history.plot(**plot)
    elif plot is True:
        history.plot()
