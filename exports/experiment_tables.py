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
    'rel_didi': 'GeDI-V1 \\%',
    # 'rel_binned_didi_2': 'DIDI-2 \\%',
    # 'rel_binned_didi_3': 'DIDI-3 \\%',
    # 'rel_binned_didi_5': 'DIDI-5 \\%',
    # 'rel_binned_didi_10': 'DIDI-10 \\%',
    'rel_generalized_didi_5': 'GeDI-V5 \\%',
    # 'abs_hgr': 'HGR',
    # 'rel_hgr': 'HGR \\%',
    'elapsed_time': 'Time'
}

MODELS = {
    'rf': ('RF', 'Plain'),
    'gb': ('GB', 'Plain'),
    'nn': ('NN', 'Plain'),
    'mt fine rf': ('RF MT', 'Fine'),
    'mt fine gb': ('GB MT', 'Fine'),
    'mt fine nn': ('NN MT', 'Fine'),
    'sbr fine': ('NN SBR', 'Fine'),
    'mt coarse rf': ('RF MT', 'Coarse'),
    'mt coarse gb': ('GB MT', 'Coarse'),
    'mt coarse nn': ('NN MT', 'Coarse'),
    'sbr coarse': ('NN SBR', 'Coarse')
}

# categorical orderings for sorting
DATASET_ORDERING = ['communities categorical', 'communities continuous', 'adult categorical', 'adult continuous']
MODEL_ORDERING = ['RF', 'RF MT', 'GB', 'GB MT', 'NN', 'NN MT', 'NN SBR']
CONSTRAINT_ORDERING = ['Plain', 'Coarse', 'Fine']

# metrics whose best value is the higher instead of the lower
MAX_METRICS = ['ACC', 'R2']

size = 'scriptsize'
bold = False
std = True

folder = '../temp'


def format_entries(column: str, series: pd.Series, function: Callable = np.min, string_format: str = '.2f'):
    if std:
        extrema = series.map(lambda x: x[0]) != function(series.map(lambda x: x[0]))
        if column == 'Time':
            bolded = series.map(lambda x: f'\\textbf{{{x[0]:{string_format}}}}')
            formatted = series.map(lambda x: f'{x[0]:{string_format}}')
        else:
            bolded = series.map(lambda x: f'\\textbf{{{x[0]:{string_format}}}} $\\pm$ {x[1]:{string_format}}')
            formatted = series.map(lambda x: f'{x[0]:{string_format}} $\\pm$ {x[1]:{string_format}}')
    else:
        extrema = series != function(series)
        bolded = series.map(lambda x: f'\\textbf{{{x:{string_format}}}}')
        formatted = series.map(lambda x: f'{x:{string_format}}')
    return formatted.where(extrema, bolded) if bold else formatted


def get_table(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    data = data[data['Dataset'] == dataset].drop(columns=['Dataset']).dropna(axis=1)
    columns_ordering = data.columns.drop(['Model', 'Constraint', 'Split'])
    aggregation = (lambda x: (np.mean(x), np.std(x))) if std else 'mean'
    data = data.pivot_table(index=['Model', 'Constraint'], columns=['Split'], aggfunc=aggregation)[columns_ordering]
    # convert float values to strings and apply boldness to min/max values
    for col in data.columns.get_level_values(0).unique():
        fn = np.max if col in MAX_METRICS else np.min
        data[col] = data[col].apply(lambda s: format_entries(column=col, series=s, function=fn))
    # use this workaround since renaming columns does not work properly for multi-index
    if 'Time' in data.columns.get_level_values(0).unique():
        times = data[('Time', 'train')]
        data = data.drop(columns=['Time'], level=0)
        data['Time'] = times
    return data


def to_latex(data: pd.DataFrame, name: str, caption: str):
    latex = data.style.to_latex(
        hrules=True,
        column_format='l' * data.index.nlevels + 'c' * data.shape[1],
        multicol_align='c',
        multirow_align='t',
        position_float='centering',
        label=f'table:{name}',
        caption=caption
    )
    latex = latex.replace('{table}', '{table*}')
    latex = latex.replace('\\begin{tabular}', f'\\{size}\n\\begin{{tabular}}')
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
    df = df[df['Model'].notna()].sort_values(['Dataset', 'Model', 'Constraint']).reset_index(drop=True)

    # export table for categorical tasks
    table_1 = get_table(data=df, dataset='communities categorical')
    table_2 = get_table(data=df, dataset='adult categorical')
    table = pd.concat((table_1, table_2), keys=['\\textit{Communities \\& Crimes}', '\\textit{Adult}'], axis=1)
    table.index = table.index.get_level_values(0)
    to_latex(
        data=table,
        name='categorical',
        caption='Results for tasks with \\textit{binary} protected attribute.'
    )

    # export table for continuous tasks
    table_1 = get_table(data=df, dataset='communities continuous')
    table_2 = get_table(data=df, dataset='adult continuous')
    table = pd.concat((table_1, table_2), keys=['\\textit{Communities \\& Crimes}', '\\textit{Adult}'], axis=1)
    to_latex(
        data=table,
        name='continuous',
        caption='Results for tasks with \\textit{continuous} protected attribute.'
    )

    # export table for continuous datasets with fine-grain results only
    table_1 = get_table(data=df, dataset='communities continuous')
    table_2 = get_table(data=df, dataset='adult continuous')
    table = pd.concat((table_1, table_2), keys=['\\textit{Communities \\& Crimes}', '\\textit{Adult}'], axis=1)
    table = table.reset_index(level=-1)
    table = table[table['Constraint'] != 'Coarse'].drop(columns=['Constraint'], level=0)
    to_latex(
        data=table,
        name='continuous_fine',
        caption='Results for tasks with \\textit{continuous} protected attribute.'
    )

    # # export table for communities tasks
    # keys = ['', 'Binary Protected Attribute', 'Continuous Protected Attribute']
    # table_1 = get_table(data=df, dataset='communities categorical')
    # table_2 = get_table(data=df, dataset='communities continuous')
    # table = pd.concat((table_1, table_2), keys=keys, axis=1)
    # table.index = table.index.get_level_values(0)
    # to_latex(
    #     data=table,
    #     name='communities',
    #     caption='Results on the \\textit{Communities \\& Crimes} dataset for binary (\\textit{race}) '
    #             'and continuous protected attribute (\\textit{pctBlack}).'
    # )
    #
    # # export table for adult tasks
    # keys = ['', 'Binary Protected Attribute', 'Continuous Protected Attribute']
    # table_1 = get_table(data=df, dataset='adult categorical')
    # table_2 = get_table(data=df, dataset='adult continuous')
    # table = pd.concat((table_1, table_2), keys=keys, axis=1)
    # table.index = table.index.get_level_values(0)
    # to_latex(
    #     data=table,
    #     name='adult',
    #     caption='Results on the \\textit{Adult} dataset for binary (\\textit{sex}) '
    #             'and continuous protected attribute (\\textit{age}).'
    # )
    #
    # # export table for communities continuous
    # table = get_table(data=df, dataset='communities continuous')
    # to_latex(
    #     data=table,
    #     name='communities_continuous',
    #     caption='Results for \\textit{Communities \\& Crimes} with continuous protected attribute.'
    # )
    #
    # # export table for adult continuous
    # table = get_table(data=df, dataset='adult continuous')
    # to_latex(
    #     data=table,
    #     name='adult_continuous',
    #     caption='Results for \\textit{Adult} with continuous protected attribute.'
    # )
