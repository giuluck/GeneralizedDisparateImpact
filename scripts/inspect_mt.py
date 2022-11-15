import seaborn as sns

from src.experiments import get

sns.set_context('notebook')
sns.set_style('whitegrid')

if __name__ == '__main__':
    exp = get('adult categorical')
    fold = exp.get_folds(folds=5)[0]
    degrees = 1 if 'categorical' in exp.__name__ else 3
    model = exp.get_model(
        model='mt rf',
        fold=fold,
        degrees=degrees,
        iterations=3,
        metrics=[exp.metrics],
        # history=dict(features=None, orient_rows=True, excluded=['adjusted/*', 'predictions/*']),
        history=dict(features=['*abs_hgr*', '*rel_hgr*'], orient_rows=True, excluded=['adjusted/*', 'predictions/*']),
        verbose=True
    )
    print('MODEL CONFIGURATION:')
    print(f'  > model: {model.__name__}')
    print(f'  > dataset: {exp.__name__}')
    for k, v in model.config.items():
        print(f'  > {k} --> {v}')
    print('-------------------------------------------------')
    x, y = fold['train']
    exp.run_instance(model=model, x=x, y=y, fold=fold, index=None, log=None, show=True)
