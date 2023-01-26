import pandas as pd
import seaborn as sns
import wandb

sns.set_context('poster')
sns.set_style('whitegrid')

MODELS = {
    'rf': ('RF', 'None'),
    'gb': ('GB', 'None'),
    'nn': ('NN', 'None'),
    'mt fine rf': ('RF MT', 'Fine'),
    'mt fine gb': ('GB MT', 'Fine'),
    'mt fine nn': ('NN MT', 'Fine'),
    'sbr fine': ('NN SBR', 'Fine'),
    'mt coarse rf': ('RF MT', 'Coarse'),
    'mt coarse gb': ('RF GB', 'Coarse'),
    'mt coarse nn': ('NN MT', 'Coarse'),
    'sbr coarse': ('NN SBR', 'Coarse')
}

DATASETS = {
    'communities categorical': ('Communities & Crimes', 'Binary', 'Continuous'),
    'communities continuous': ('Communities & Crimes', 'Continuous', 'Continuous'),
    'adult categorical': ('Adult', 'Binary', 'Binary'),
    'adult continuous': ('Adult', 'Continuous', 'Continuous')
}

COLUMNS = {
    'dataset': 'Dataset',
    'protected': 'Protected Attribute',
    'target': 'Output Target',
    'model': 'Model',
    'constraint': 'Constraint',
    'fold': 'Fold',
    'split': 'Split',
    'mse': 'MSE',
    'r2': 'R2',
    'crossentropy': 'Crossentropy',
    'accuracy': 'Accuracy',
    'rel_generalized_didi_1': '% DIDI-V1',
    'rel_generalized_didi_5': '% DIDI-V5',
    'rel_didi': '% DIDI',
    'rel_binned_didi_2': '% DIDI-2',
    'rel_binned_didi_3': '% DIDI-3',
    'rel_binned_didi_5': '% DIDI-5',
    'rel_binned_didi_10': '% DIDI-10',
    'rel_hgr': '% HGR'
}

folder = '../temp'

if __name__ == '__main__':
    runs = wandb.Api().runs('shape-constraints/experiments')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])

    train = df.rename(columns=lambda s: s.replace('train/', ''))
    val = df.rename(columns=lambda s: s.replace('val/', ''))
    train['split'] = 'Train'
    val['split'] = 'Validation'
    df = pd.concat([train, val])

    df['target'] = df['dataset'].map({k: v for k, (_, _, v) in DATASETS.items()})
    df['protected'] = df['dataset'].map({k: v for k, (_, v, _) in DATASETS.items()})
    df['dataset'] = df['dataset'].map({k: v for k, (v, _, _) in DATASETS.items()})
    df['constraint'] = df['model'].map({k: v for k, (_, v) in MODELS.items()})
    df['model'] = df['model'].map({k: v for k, (v, _) in MODELS.items()})
    df['rel_generalized_didi_1'] = df['rel_didi']

    df = df[COLUMNS.keys()].rename(columns=COLUMNS).reset_index(drop=True)
    df.to_csv(f'{folder}/results.csv')
