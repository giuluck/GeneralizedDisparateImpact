from typing import Optional

import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

sns.set_context('poster')
sns.set_style('whitegrid')

# data columns
COLUMNS = {
    'model': 'Model',
    'dataset': 'Dataset',
    'degrees': 'Kernel Degree',
    'adjusted/rel_didi_2': 'Adj. 2-DIDI',
    'adjusted/rel_didi_3': 'Adj. 3-DIDI',
    'adjusted/rel_didi_5': 'Adj. 5-DIDI',
    'adjusted/rel_didi_10': 'Adj. 10-DIDI',
    'predictions/rel_didi_2': 'Pred. 2-DIDI',
    'predictions/rel_didi_3': 'Pred. 3-DIDI',
    'predictions/rel_didi_5': 'Pred. 5-DIDI',
    'predictions/rel_didi_10': 'Pred. 10-DIDI'
}

# models and datasets aliases
MODELS = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'nn': 'Neural Network'}
DATASETS = {'communities continuous': 'Communities', 'adult continuous': 'Adult'}

# categorical orderings in plots
ORDERINGS = {
    'Model': list(MODELS.values()),
    'Dataset': list(DATASETS.values()),
    'Vector': ['Predictions', 'Projections'],
    'Kernel Degree': [1, 2, 3, 5],
    'Bins': [2, 3, 5, 10]
}

folder = '../temp'
save_plot = True
show_plot = True


def plot(data: pd.DataFrame,
         name: str,
         ylim: Optional[float] = None,
         row: Optional[str] = None,
         col: Optional[str] = None,
         title: Optional[str] = None):
    row_order = [None] if row is None else ORDERINGS[row]
    col_order = [None] if col is None else ORDERINGS[col]
    fig = sns.catplot(
        data=data,
        kind='bar',
        x='Bins',
        y='% DIDI',
        order=ORDERINGS['Bins'],
        hue='Kernel Degree',
        hue_order=ORDERINGS['Kernel Degree'],
        row=row,
        row_order=row_order,
        col=col,
        col_order=col_order,
        errorbar=None,
        palette='tab10',
        legend_out=False,
        aspect=4 / 3,
        height=6
    )
    fig.tight_layout()
    for a in fig.axes.flat:
        a.set_title(a.title.get_text().replace(f'{row} = ', '').replace(f'{col} = ', ''), size=28)
        if ylim is not None:
            a.set_ylim((0, ylim))
    if title is not None:
        fig.figure.suptitle(f'Percentage Disparate Impact for {title}')
        fig.figure.subplots_adjust(top=0.85)
    if save_plot:
        plt.savefig(f'{folder}/{name}.png', format='png')
        plt.savefig(f'{folder}/{name}.svg', format='svg')
        plt.savefig(f'{folder}/{name}.eps', format='eps')
    if show_plot:
        plt.show()


if __name__ == '__main__':
    # download results using wandb api
    runs = wandb.Api().runs('shape-constraints/nci_mt')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])
    df = df[COLUMNS.keys()].rename(columns=COLUMNS).dropna(axis=0).reset_index(drop=True)
    df['Model'] = df['Model'].map(MODELS)
    df['Dataset'] = df['Dataset'].map(DATASETS)
    df = df.pivot_table(index=['Kernel Degree', 'Model', 'Dataset'])
    df = df.stack().reset_index(name='% DIDI').rename(columns={'level_3': 'Bins'})
    df['Vector'] = ['Predictions' if 'Pred.' in v else 'Projections' for v in df['Bins']]
    df['Bins'] = [int(v.split(' ')[1][:-5]) for v in df['Bins']]

    # plot results for predictions and for projections separately on gradient boosting model only
    df = df[df['Model'] == 'Gradient Boosting']
    df_pred = df[df['Vector'] == 'Predictions']
    df_proj = df[df['Vector'] == 'Projections']
    lim = 1.1 * df['% DIDI'].max().max()
    plot(data=df_pred, ylim=lim, col='Dataset', name='predictions', title=None)
    plot(data=df_proj, ylim=lim, col='Dataset', name='projections', title=None)
