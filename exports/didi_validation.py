import time
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from moving_targets.metrics import DIDI
from src.experiments import get
from src.metrics import BinnedDIDI
from src.models import AbstractMaster

sns.set_context('poster')
sns.set_style('whitegrid')

# categorical orderings for datasets, kernels, and bins
DATASETS = ['Communities & Crimes', 'Adult']
BINS = [2, 3, 5, 10]

folder = '../temp'
save_plot = True
show_plot = True


def plot(df: dict, x_col: str, y_col: str, order: list, name: str, title: Optional[str] = None):
    df = pd.DataFrame.from_dict(df).transpose().stack().reset_index()
    df = df.rename(columns={'level_0': 'Dataset', 'level_1': 'Constraint Bins', 'level_2': x_col, 0: y_col})
    df['Constraint Bins'] = df['Constraint Bins'].astype(int)
    df[x_col] = df[x_col].map(lambda s: int(s[4:]))
    fig = sns.catplot(
        data=df,
        kind='bar',
        x=x_col,
        y=y_col,
        order=order,
        hue='Constraint Bins',
        hue_order=BINS,
        col='Dataset',
        col_order=DATASETS,
        errorbar=None,
        palette='tab10',
        legend_out=False,
        aspect=1,
        height=7
    )
    fig.tight_layout()
    for a in fig.axes.flat:
        a.set_title(a.title.get_text().replace(f'Dataset = ', ''), size=28)
        a.set_ylim((0, 1))
    if title is not None:
        fig.figure.suptitle(title)
        fig.figure.subplots_adjust(top=0.82)
    if save_plot:
        plt.savefig(f'{folder}/{name}.eps', format='eps')
    if show_plot:
        plt.show()


class BinnedDIDIMaster(AbstractMaster):
    def _formulation(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, v: np.ndarray):
        assert self.relative, "BinnedDIDI implementation supports relative constraints only."
        # create dataframe by binning the protected attribute with the number of bins (stored in self.degree)
        x = pd.DataFrame(data=pd.qcut(x, q=self.degree).codes, columns=['x'])
        # as a first step, we need to compute the deviations between the average output for the total dataset and the
        # average output respectively to each protected class
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected='x')
        deviations = self.backend.add_continuous_variables(len(indicator_matrix), lb=0.0, name='deviations')
        # this is the average output target for the whole dataset
        total_avg = self.backend.mean(v)
        for g, protected_group in enumerate(indicator_matrix):
            # this is the subset of the variables having <label> as protected feature (i.e., the protected group)
            protected_vars = v[protected_group]
            if len(protected_vars) == 0:
                continue
            # this is the average output target for the protected group
            protected_avg = self.backend.mean(protected_vars)
            # eventually, the partial deviation is computed as the absolute value (which is linearized) of the
            # difference between the total average samples and the average samples within the protected group
            self.backend.add_constraint(deviations[g] >= total_avg - protected_avg)
            self.backend.add_constraint(deviations[g] >= protected_avg - total_avg)
        # finally, we compute the DIDI as the sum of this deviations, which is constrained to be lower or equal to the
        # given value (also, since we are computing the percentage DIDI, we need to scale for the original train_didi)
        didi = self.backend.sum(deviations)
        train_didi = DIDI.regression_didi(indicator_matrix=indicator_matrix, targets=y)
        self.backend.add_constraint(didi <= self.threshold * train_didi)


if __name__ == '__main__':
    results = {}
    for dataset in DATASETS:
        print(f'{dataset.upper()}:')
        exp = get(f'{dataset.split(" ")[0].lower()} continuous')
        xx, yy = exp.data
        mtr = [BinnedDIDI(exp.classification, exp.excluded, bins=b, name=f'bin {b}') for b in BINS]
        for b in BINS:
            print(f'  - b = {b}:', end=' ')
            mst = BinnedDIDIMaster(exp.classification, exp.excluded, degree=b, threshold=0.2, relative=True)
            start = time.time()
            adj = mst.adjust_targets(x=xx, y=yy, p=None)
            mse = mean_squared_error(yy, adj)
            print(f'{mse:.4f} ({time.time() - start:.2f}s)')
            results[(dataset, b)] = {m.__name__: m(xx, yy, adj) for m in mtr}

    plot(df=results, x_col='Evaluation Bins', y_col='% DIDI', order=BINS, name='bin_didi', title=None)
