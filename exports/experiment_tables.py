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
    # 'abs_hgr': 'HGR',
    # 'rel_hgr': 'HGR \\%',
    'rel_didi': 'GeDI-V1 \\%',
    'rel_binned_didi_2': 'DIDI-2 \\%',
    'rel_binned_didi_3': 'DIDI-3 \\%',
    'rel_binned_didi_5': 'DIDI-5 \\%',
    'rel_binned_didi_10': 'DIDI-10 \\%',
    'rel_generalized_didi_5': 'GeDI-V5 \\%',
    'elapsed_time': 'Time'
}

MODELS = {
    'rf': ('RF', ''),
    'gb': ('GB', ''),
    'nn': ('NN', ''),
    'mt fine rf': ('MT RF', 'Fine'),
    'mt fine gb': ('MT GB', 'Fine'),
    'mt fine nn': ('MT NN', 'Fine'),
    'mt coarse rf': ('MT RF', 'Coarse'),
    'mt coarse gb': ('MT GB', 'Coarse'),
    'mt coarse nn': ('MT NN', 'Coarse'),
    'sbr fine': ('SBR', 'Fine'),
    'sbr coarse': ('SBR', 'Coarse')
}

# categorical orderings for sorting
DATASET_ORDERING = ['communities categorical', 'communities continuous', 'adult categorical', 'adult continuous']
MODEL_ORDERING = ['RF', 'GB', 'NN', 'MT RF', 'MT GB', 'MT NN', 'SBR']
CONSTRAINT_ORDERING = ['', 'Coarse', 'Fine']

# metrics whose best value is the higher instead of the lower
MAX_METRICS = ['ACC', 'R2']

folder = '../temp'


def bold_extreme_values(series: pd.Series, function: Callable = np.min, format_string: str = '.2f'):
    extrema = series != function(series)
    bolded = series.map(lambda x: f'\\textbf{{{x:{format_string}}}}')
    formatted = series.map(lambda x: f'{x:{format_string}}')
    return formatted.where(extrema, bolded)


def get_table(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    data = data[data['Dataset'] == dataset].drop(columns=['Dataset']).dropna(axis=1)
    columns_ordering = data.columns.drop(['Model', 'Constraint', 'Split'])
    data = data.pivot_table(index=['Model', 'Constraint'], columns=['Split'])[columns_ordering]
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
        multicol_align='c',
        multirow_align='t',
        position_float='centering',
        label=f'table:{name}',
        caption=caption
    )
    latex = latex.replace('{table}', '{table*}')
    latex = latex.replace('\\begin{tabular}', '\\small\n\\begin{tabular}')
    latex = latex.replace('{Split}', '{}')
    latex = latex.replace('{Model} ' + '& {} ' * len(data.columns) + '\\\\\n', '')
    latex = latex.replace('{Model} & {Constraint} ' + '& {} ' * len(data.columns) + '\\\\\n', '')
    with open(f'{folder}/{name}.tex', 'w') as f:
        f.write(latex)


if __name__ == '__main__':
    # download results using wandb api
    runs = wandb.Api().runs('shape-constraints/experiments')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])
    # split and concatenate the train and validation data (this will be used for pivoting)
    tr = df.rename(columns=lambda s: s.replace('train/', ''))
    vl = df.rename(columns=lambda s: s.replace('val/', ''))
    tr['split'] = 'train'
    vl['split'] = 'val'
    df = pd.concat([tr, vl]).reset_index(drop=True)
    # change columns names and data types accordingly
    df = df[COLUMNS.keys()].rename(columns=COLUMNS)
    df['Dataset'] = pd.Categorical(df['Dataset'], categories=DATASET_ORDERING, ordered=True)
    cst_mapping = {k: v for k, (_, v) in MODELS.items()}
    df['Constraint'] = pd.Categorical(df['Model'].map(cst_mapping), categories=CONSTRAINT_ORDERING, ordered=True)
    mdl_mapping = {k: v for k, (v, _) in MODELS.items()}
    df['Model'] = pd.Categorical(df['Model'].map(mdl_mapping), categories=MODEL_ORDERING, ordered=True)
    df = df.sort_values(['Dataset', 'Model', 'Constraint']).reset_index(drop=True)
    # export table for categorical datasets, then manually replace:
    #       \begin{tabular}{l|l...l|l...l}
    #       \toprule
    #       {} & \multicolumn{.}{c|}{\textit{Communities \& Crimes}} & \multicolumn{.}{c}{\textit{Adult}} \\
    table_comm = get_table(data=df, dataset='communities categorical')
    table_adult = get_table(data=df, dataset='adult categorical')
    table = pd.concat((table_comm, table_adult), keys=['\\textit{Communities \\& Crimes}', '\\textit{Adult}'], axis=1)
    table.index = table.index.get_level_values(0)
    to_latex(data=table, name='categorical', caption='Results for datasets with binary protected feature.')
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
