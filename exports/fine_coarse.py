import matplotlib.pyplot as plt
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

BINS = [2, 3, 5, 10]
CONSTRAINTS = ['Coarse', 'Fine']

split = False

folder = '../temp'
save_plot = True
show_plot = True

if __name__ == '__main__':
    # download results using wandb api and remove
    runs = wandb.Api().runs('shape-constraints/experiments')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])

    # discard train/val data by concatenating them
    df = pd.concat([df.rename(columns=lambda s: s.replace(f'{split}/', '')) for split in ['train', 'val']])

    # create splitting based on binned didi
    df_concat = []
    for b in BINS:
        df_b = df.rename(columns={f'rel_binned_didi_{b}': f'% DIDI'})
        df_b['Bins'] = b
        df_concat.append(df_b)
    df = pd.concat(df_concat)

    # create splitting based on constraint type and filter for continuous attributes tasks only
    df['Constraint'] = df['model'].map(MODELS)
    df = df[df['dataset'].map(lambda s: 'continuous' in s)]
    df = df[df['Constraint'] != 'None'].reset_index(drop=True)

    # change columns names and data types accordingly
    df['% GeDI'] = df['rel_generalized_didi_5']
    df = df[['Constraint', 'Bins', '% GeDI', '% DIDI']]

    # plot data
    fig = sns.relplot(
        data=df,
        kind='scatter',
        x='% GeDI',
        y='% DIDI',
        hue=None if split else 'Constraint',
        hue_order=CONSTRAINTS,
        style=None if split else 'Bins',
        style_order=BINS,
        row='Constraint' if split else None,
        row_order=CONSTRAINTS,
        col='Bins' if split else None,
        col_order=BINS,
        palette='tab10',
        edgecolor='black',
        linewidth=0.3,
        alpha=None if split else 0.9,
        facet_kws={'sharex': True, 'sharey': True},
        height=7,
        aspect=1,
        s=100
    )
    if save_plot:
        plt.savefig(f'{folder}/fine_coarse.eps', format='eps')
    if show_plot:
        plt.show()
