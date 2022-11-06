import seaborn as sns

from src.experiments import get
from src.metrics import RegressionWeight

sns.set_context('notebook')
sns.set_style('whitegrid')

if __name__ == '__main__':
    exp = get('communities categorical')
    fold = exp.get_folds(folds=1)
    degrees = 1 if 'race' in exp.__name__ else 3
    model = exp.get_model(
        model='mt',
        fold=fold,
        degrees=degrees,
        iterations=15,
        metrics=exp.metrics + [RegressionWeight(feature=f, degree=degrees, name=f) for f in exp.excluded],
        history=dict(features=None, orient_rows=True, excluded=['adjusted/*', 'predictions/*']),
        verbose=True
    )
    print('MODEL CONFIGURATION:')
    for k, v in model.config.items():
        print(f'  > {k} --> {v}')
    print('-------------------------------------------------')
    x, y = fold['train']
    exp.run_instance(model=model, x=x, y=y, fold=fold, index=None, log=None, show=True)
