import time
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.experiments import get
from src.metrics import BinnedDIDI, GeneralizedDIDI
from src.models import FirstOrderMaster, GeneralizedDIDIMaster

sns.set_context('poster')
sns.set_style('whitegrid')

# categorical orderings for datasets, kernels, and bins
DATASETS = ['Communities', 'Adult']
KERNELS = [1, 2, 3, 4, 5]
BINS = [2, 3, 5, 10]

folder = '../temp'
save_plot = True
show_plot = True


@contextmanager
def elapsed_time(prefix: str = '', suffix: str = '\n', fmt: str = '.2f'):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f'{prefix}{end - start:{fmt}}s{suffix}', end='')


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
        aspect=4 / 3,
        height=7
    )
    fig.tight_layout()
    for a in fig.axes.flat:
        a.set_title(a.title.get_text().replace(f'Dataset = ', ''), size=28)
        a.set_ylim((0, 1))
    if save_plot:
        plt.savefig(f'{folder}/{name}.png', format='png')
        plt.savefig(f'{folder}/{name}.svg', format='svg')
        plt.savefig(f'{folder}/{name}.eps', format='eps')
    if title is not None:
        fig.figure.suptitle(f'Percentage Disparate Impact for {title}')
        fig.figure.subplots_adjust(top=0.85)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    first_bin, didi_bin, first_gen, didi_gen = {}, {}, {}, {}
    for dataset in DATASETS:
        print(f'{dataset.upper()}:')
        exp = get(f'{dataset.lower()} continuous')
        x, y = exp.data
        mtr_bin = [BinnedDIDI(exp.classification, exp.excluded, bins=b, name=f'bin {b}') for b in BINS]
        mtr_gen = [GeneralizedDIDI(exp.classification, exp.excluded, degree=k, name=f'gen {k}') for k in KERNELS]
        for kernel in KERNELS:
            print(f'  - k = {kernel}:', end=' ')
            mst = FirstOrderMaster(exp.classification, exp.excluded, degree=kernel, threshold=0.2, relative=1)
            with elapsed_time(prefix='first (', suffix=') & '):
                adj = mst.adjust_targets(x=x, y=y, p=None)
            first_bin[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_bin}
            first_gen[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_gen}
            mst = GeneralizedDIDIMaster(exp.classification, exp.excluded, degree=kernel, threshold=0.2, relative=1)
            with elapsed_time(prefix='didi (', suffix=')\n'):
                adj = mst.adjust_targets(x=x, y=y, p=None)
            didi_bin[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_bin}
            didi_gen[(dataset, kernel)] = {m.__name__: m(x, y, adj) for m in mtr_gen}

    plot(df=first_bin, x_col='Bins', y_col='% DIDI', order=BINS, name='first_bin', title='Higher-order Exclusion')
    plot(df=didi_bin, x_col='Bins', y_col='% DIDI', order=BINS, name='didi_bin', title='Generalized DIDI')
    plot(df=first_gen, x_col='k', y_col='gDIDI_k', order=KERNELS, name='first_gen', title='Higher-order Exclusion')
    plot(df=didi_gen, x_col='k', y_col='gDIDI_k', order=KERNELS, name='didi_gen', title='Generalized DIDI')
