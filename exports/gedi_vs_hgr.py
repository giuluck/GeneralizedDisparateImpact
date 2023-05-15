"""This script is used to create the plots of computed HGR vs computed GeDI (Figure 4)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

sns.set_context('poster')
sns.set_style('whitegrid')

MODELS = {
    'rf': 'None',
    'gb': 'None',
    'nn': 'None',
    'mt fine rf': 'Fine',
    'mt fine gb': 'Fine',
    'mt fine nn': 'Fine',
    'sbr fine': 'Fine',
    'mt coarse rf': 'Coarse',
    'mt coarse gb': 'Coarse',
    'mt coarse nn': 'Coarse',
    'sbr coarse': 'Coarse'
}

constraints = ['None', 'Fine', 'Coarse']
split_constraints = True

folder = '../temp'
save_plot = False
show_plot = True

if __name__ == '__main__':
    # download results using wandb api and remove
    runs = wandb.Api().runs('shape-constraints/gedi-experiments')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])

    # discard train/val data by concatenating it
    train = df.rename(columns=lambda s: s.replace('train/', ''))
    val = df.rename(columns=lambda s: s.replace('val/', ''))
    df = pd.concat([train, val])
    df['Constraint'] = df['model'].map(MODELS)
    df = df[df['Constraint'].isin(constraints)].reset_index(drop=True)

    # change columns names and data types accordingly
    df['% HGR'] = df['rel_hgr']
    df['% GeDI'] = np.where(df['rel_didi'].isna(), df['rel_generalized_didi_5'], df['rel_didi'])
    df = df[['Constraint', '% GeDI', '% HGR']]

    # plot data
    plt.figure(figsize=(10, 6), tight_layout=True)
    fig = sns.scatterplot(
        data=df,
        x='% GeDI',
        y='% HGR',
        hue='Constraint' if split_constraints else None,
        hue_order=constraints,
        style='Constraint' if split_constraints else None,
        style_order=constraints,
        edgecolor='black',
        linewidth=0.0,
        s=100
    )
    sns.regplot(
        data=df,
        x='% GeDI',
        y='% HGR',
        line_kws={'linewidth': 2, 'color': 'black'},
        scatter=False,
        ci=None
    )
    fig.set_xlim(0.0)
    fig.set_ylim(0.0)

    if save_plot:
        plt.savefig(f'{folder}/gedi_hgr.eps', format='eps')
    if show_plot:
        plt.show()
