"""This script is used to create the Binned DIDI plots after GeDI constraint (Figure 3)."""

import time
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from src.experiments import get
from src.metrics import BinnedDIDI, GeneralizedDIDI
from src.models import FineGrainedMaster, CoarseGrainedMaster

sns.set_context('poster')
sns.set_style('whitegrid')

# categorical orderings for datasets, kernels, and bins
DATASETS = ['Communities & Crimes', 'Adult']
KERNELS = [1, 2, 3, 4, 5]
BINS = [2, 3, 5, 10]

folder = '../temp'
save_plot = False
show_plot = True


def plot(df: dict, x_col: str, y_col: str, order: list, name: str, title: Optional[str] = None):
    df = pd.DataFrame.from_dict(df).transpose().stack().reset_index()
    df = df.rename(columns={'level_0': 'Dataset', 'level_1': 'Kernel Degree', 'level_2': x_col, 0: y_col})
    df['Kernel Degree'] = df['Kernel Degree'].astype(int)
    df[x_col] = df[x_col].map(lambda s: int(s[4:]))
    fig = sns.catplot(
        data=df,
        kind='bar',
        x=x_col,
        y=y_col,
        order=order,
        hue='Kernel Degree',
        hue_order=KERNELS,
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


if __name__ == '__main__':
    fine_didi, coarse_didi, fine_gedi, coarse_gedi = {}, {}, {}, {}
    for dataset in DATASETS:
        print(f'{dataset.upper()}:')
        exp = get(f'{dataset.split(" ")[0].lower()} continuous')
        x, y = exp.data
        mtr_bin = [BinnedDIDI(exp.classification, exp.excluded, bins=b, name=f'bin {b}') for b in BINS]
        mtr_gen = [GeneralizedDIDI(exp.classification, exp.excluded, degree=k, name=f'gen {k}') for k in KERNELS]
        for kernel in KERNELS:
            print(f'  - k = {kernel}:', end=' ')
            mst = CoarseGrainedMaster(exp.classification, exp.excluded, degree=kernel, threshold=0.2, relative=1)
            start = time.time()
            adj = mst.adjust_targets(x=x, y=y, p=None)
            mse = mean_squared_error(y, adj)
            print(f'coarse {mse:.4f} ({time.time() - start:.2f}s) &', end=' ')
            coarse_didi[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_bin}
            coarse_gedi[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_gen}
            mst = FineGrainedMaster(exp.classification, exp.excluded, degree=kernel, threshold=0.2, relative=1)
            start = time.time()
            adj = mst.adjust_targets(x=x, y=y, p=None)
            mse = mean_squared_error(y, adj)
            print(f'fine {mse:.4f} ({time.time() - start:.2f}s)')
            fine_didi[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_bin}
            fine_gedi[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_gen}

    plot(df=fine_didi, x_col='Bins', y_col='% DIDI', order=BINS, name='fine_didi', title=None)
    plot(df=coarse_didi, x_col='Bins', y_col='% DIDI', order=BINS, name='coarse_didi', title=None)
