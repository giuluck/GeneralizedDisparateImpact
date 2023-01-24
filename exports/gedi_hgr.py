import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

sns.set_context('poster')
sns.set_style('whitegrid')

COLUMNS = {
    'split': 'Split',
    'values': 'Values',
    'output': 'Output',
    'protected': 'Protected',
    'dataset': 'Dataset',
    'model': 'Model',
    'hgr': 'HGR',
    'didi': 'DIDI'
}

# in order to avoid having five different column values for the adult categorical dataset
# we first retrieve them and then aggregate them via mean
RACES_CATEGORICAL = ['Other', 'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo']
HGR_CATEGORICAL = {
    'train/rel_hgr': [f'train/rel_hgr_race_{race}' for race in RACES_CATEGORICAL],
    'val/rel_hgr': [f'val/rel_hgr_race_{race}' for race in RACES_CATEGORICAL]
}

plot_regression = True

folder = '../temp'
save_plot = True
show_plot = True

if __name__ == '__main__':
    # download results using wandb api
    runs = wandb.Api().runs('shape-constraints/nci_experiments')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])
    # handle hgr in categorical fairness scenario
    for column, keys in HGR_CATEGORICAL.items():
        replacements = df[keys].mean(axis=1)
        df[column] = np.where(np.isnan(df[column]), replacements, df[column])
    df['output'] = df['dataset'].map(lambda s: 'categorical' if 'adult' in s else 'continuous')
    df['protected'] = df['dataset'].map(lambda s: 'categorical' if 'categorical' in s else 'continuous')
    df['model'] = df['model'].map(lambda s: s.replace('cov', 'lr'))
    # split and concatenate the train and validation data
    tr = df.rename(columns=lambda s: s.replace('train/', ''))
    vl = df.rename(columns=lambda s: s.replace('val/', ''))
    tr['split'] = 'train'
    vl['split'] = 'val'
    df = pd.concat([tr, vl]).reset_index(drop=True)
    # split and concatenate the relative and absolute data
    tr = df.rename(columns=lambda s: s.replace('rel_', ''))
    vl = df.rename(columns=lambda s: s.replace('abs_', ''))
    tr['values'] = 'Relative Values'
    vl['values'] = 'Absolute Values'
    df = pd.concat([tr, vl]).reset_index(drop=True)
    # change columns names and data types accordingly
    df['didi'] = np.where(np.isnan(df['didi']), df['didi_2'], df['didi'])
    df = df[COLUMNS.keys()].rename(columns=COLUMNS)

    # plot data
    fig = sns.relplot(
        data=df,
        kind='scatter',
        x='DIDI',
        y='HGR',
        hue='Output',
        hue_order=['categorical', 'continuous'],
        style='Protected',
        style_order=['categorical', 'continuous'],
        col='Values',
        palette='tab10',
        edgecolor='black',
        linewidth=0.3,
        facet_kws={'sharex': False, 'sharey': False},
        height=7,
        aspect=1,
        s=250
    )
    for ax, val in zip(fig.axes[0], ['Relative Values', 'Absolute Values']):
        if plot_regression:
            sns.regplot(
                data=df[df['Values'] == val],
                x='DIDI',
                y='HGR',
                color='black',
                line_kws={'linewidth': 2},
                scatter=False,
                ci=None,
                ax=ax
            )
        ax.set_title(ax.title.get_text().replace('Values = ', ''), size=28)
        ax.set_xlim(0.0)
        ax.set_ylim(0.0)
    if save_plot:
        plt.savefig(f'{folder}/didi_hgr.png', format='png')
        plt.savefig(f'{folder}/didi_hgr.svg', format='svg')
        plt.savefig(f'{folder}/didi_hgr.eps', format='eps')
    if show_plot:
        plt.show()
