from typing import Callable

import numpy as np
import pandas as pd
import wandb

# data columns
COLUMNS = {
    'split': 'Split',
    'model': 'Model',
    'dataset': 'Dataset',
    # 'crossentropy': 'CE',
    'accuracy': 'ACC',
    # 'mse': 'MSE',
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
MODELS_ORDERING = ['RF', 'GB', 'NN', 'MT RF', 'MT GB', 'MT NN', 'SBR LR', 'SBR HGR']
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


def get_table(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    data = data[data['Dataset'] == dataset].drop(columns=['Dataset']).dropna(axis=1)
    data = data.pivot_table(index='Model', columns=['Split'])[data.columns.drop(['Model', 'Split'])]
    # convert float values to strings and apply boldness to min/max values
    for col in data.columns.get_level_values(0).unique():
        fn = np.max if col in MAX_METRICS else np.min
        data[col] = data[col].apply(lambda s: bold_extreme_values(series=s, function=fn))
    # use this workaround since renaming columns does not work properly for multi-index
    if 'Time' in data.columns.get_level_values(0).unique():
        times = data[('Time', 'train')]
        data = data.drop(columns=['Time'], level=0)
        data['Time'] = times
    return data


def to_latex(data: pd.DataFrame, name: str, caption: str):
    latex = data.style.to_latex(
        hrules=True,
        multicol_align='l',
        position_float='centering',
        label=f'table:{name}',
        caption=caption
    )
    latex = latex.replace('Split', '')
    latex = latex.replace('Model ' + '&  ' * len(data.columns) + '\\\\\n', '')
    with open(f'{folder}/{name}.tex', 'w') as f:
        f.write(latex)


if __name__ == '__main__':
    # download results using wandb api
    runs = wandb.Api().runs('shape-constraints/nci_experiments')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])
    # handle hgr in categorical fairness scenario
    for column, keys in HGR_CATEGORICAL.items():
        replacements = df[keys].mean(axis=1)
        df[column] = np.where(np.isnan(df[column]), replacements, df[column])
    df['model'] = df['model'].map(lambda s: s.replace('cov', 'lr'))
    # split and concatenate the train and validation data (this will be used for pivoting)
    tr = df.rename(columns=lambda s: s.replace('train/', ''))
    vl = df.rename(columns=lambda s: s.replace('val/', ''))
    tr['split'] = 'train'
    vl['split'] = 'val'
    df = pd.concat([tr, vl]).reset_index(drop=True)
    # change columns names and data types accordingly
    df = df[COLUMNS.keys()].rename(columns=COLUMNS)
    df['Dataset'] = pd.Categorical(df['Dataset'], categories=DATASET_ORDERING, ordered=True)
    df['Model'] = pd.Categorical(df['Model'].map(str.upper), categories=MODELS_ORDERING, ordered=True)
    df = df.sort_values(['Dataset', 'Model']).reset_index(drop=True)
    # export table for categorical datasets
    table_comm = get_table(data=df, dataset='communities categorical')
    table_adult = get_table(data=df, dataset='adult categorical')
    table = pd.concat((table_comm, table_adult), keys=['\\textit{Communities \\& Crimes}', '\\textit{Adult}'], axis=1)
    to_latex(data=table, name='categorical', caption='Results for datasets with categorical protected feature.')
    # export table for communities continuous
    table = get_table(data=df, dataset='communities continuous')
    to_latex(
        data=table,
        name='communities',
        caption='Results for \\textit{Communities \\& Crimes} with continuous protected feature.'
    )
    # export table for adult continuous
    table = get_table(data=df, dataset='adult continuous')
    to_latex(data=table, name='adult', caption='Results for \\textit{Adult} with continuous protected feature.')
