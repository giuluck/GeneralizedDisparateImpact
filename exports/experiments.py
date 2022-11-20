from typing import Callable

import numpy as np
import pandas as pd
import wandb

# data columns
COLUMNS = {
    'model': 'Model',
    'dataset': 'Dataset',
    'crossentropy': 'CE',
    'accuracy': 'ACC',
    'mse': 'MSE',
    'r2': 'R2',
    'rel_hgr': 'HGR',
    'rel_didi': 'DIDI',
    'rel_didi_2': '2-DIDI',
    'rel_didi_3': '3-DIDI',
    'rel_didi_5': '5-DIDI',
    'rel_didi_10': '10-DIDI',
    'elapsed_time': 'Time'
}

# metrics whose best value is the higher instead of the lower
MAX_METRICS = ['ACC', 'R2']

# categorical orderings for sorting
MODELS_ORDERING = ['RF', 'GB', 'NN', 'MT RF', 'MT GB', 'MT NN', 'SBR COV', 'SBR HGR']
DATASET_ORDERING = ['communities categorical', 'communities continuous', 'adult categorical', 'adult continuous']

# in order to avoid having five different column values for the adult categorical dataset
# we first retrieve them and then aggregate them via mean
RACES_CATEGORICAL = ['Other', 'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo']
HGR_CATEGORICAL = {
    'train/rel_hgr': [f'train/rel_hgr_race_{race}' for race in RACES_CATEGORICAL],
    'val/rel_hgr': [f'val/rel_hgr_race_{race}' for race in RACES_CATEGORICAL]
}

folder = '../temp'


def bold_extreme_values(series: pd.Series, function: Callable = np.min, format_string: str = '.2f'):
    extrema = series != function(series)
    bolded = series.map(lambda x: f'\\textbf{{{x:{format_string}}}}')
    formatted = series.map(lambda x: f'{x:{format_string}}')
    return formatted.where(extrema, bolded)


if __name__ == '__main__':
    # download results using wandb api
    runs = wandb.Api().runs('shape-constraints/nci_experiments')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])
    # handle hgr in categorical fairness scenario
    for column, keys in HGR_CATEGORICAL.items():
        replacements = df[keys].mean(axis=1)
        df[column] = np.where(np.isnan(df[column]), replacements, df[column])
    # split and concatenate the train and validation data (this will be used for pivoting)
    tr = df.rename(columns=lambda s: s.replace('train/', ''))
    vl = df.rename(columns=lambda s: s.replace('val/', ''))
    tr['split'] = 'train'
    vl['split'] = 'val'
    df = pd.concat([tr, vl]).reset_index(drop=True)
    # change columns names and data types accordingly
    df = df[list(COLUMNS.keys()) + ['split']].rename(columns=COLUMNS)
    df['Dataset'] = pd.Categorical(df['Dataset'], categories=DATASET_ORDERING, ordered=True)
    df['Model'] = pd.Categorical(df['Model'].map(str.upper), categories=MODELS_ORDERING, ordered=True)
    df = df.sort_values(['Dataset', 'Model']).reset_index(drop=True)
    # group by dataset and return a latex table for each of them
    for dataset in DATASET_ORDERING:
        table = df[df['Dataset'] == dataset].drop(columns=['Dataset']).dropna(axis=1)
        table = table.pivot_table(index='Model', columns=['split'])[table.columns.drop(['Model', 'split'])]
        for col in table.columns.get_level_values(0).unique():
            fn = np.max if col in MAX_METRICS else np.min
            table[col] = table[col].apply(lambda s: bold_extreme_values(series=s, function=fn))
        latex = table.style.to_latex(
            hrules=True,
            multicol_align='l',
            position_float='centering',
            label=f'tab:{dataset.replace(" ", "_")}',
            caption=f'Results for \\textit{{{dataset}}} dataset.'
        )
        latex = latex.replace('split', '')
        latex = latex.replace('Model ' + '&  ' * len(table.columns) + '\\\\\n', '')
        with open(f'{folder}/{dataset.replace(" ", "_")}.tex', 'w') as f:
            f.write(latex)
