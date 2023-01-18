import seaborn as sns

from moving_targets.metrics import DIDI
from src.experiments import get
from src.metrics import GeneralizedDIDI
from src.models import MovingTargets

sns.set_context('notebook')
sns.set_style('whitegrid')

if __name__ == '__main__':
    exp = get('communities continuous')
    fold = exp.get_folds(folds=1)
    degree = 5 if exp.continuous else 1
    model = exp.get_model(
        model='mt first rf',
        fold=fold,
        degree=degree,
        iterations=3,
        metrics=[m for m in exp.metrics if isinstance(m, DIDI) or isinstance(m, GeneralizedDIDI)],
        history=dict(features=None, orient_rows=True, excluded=['predictions/*', 'train/*', 'test/*']),
        # history=dict(features=None, orient_rows=True, excluded=['predictions/*', 'adjusted/*']),
        verbose=1
    )
    assert isinstance(model, MovingTargets), f"Model should be MovingTargets, got '{type(model)}'"
    print('MODEL CONFIGURATION:')
    print(f'  > model: {model.__name__}')
    print(f'  > dataset: {exp.__name__}')
    for k, v in model.config.items():
        print(f'  > {k} --> {v}')
    print('-------------------------------------------------')
    xf, yf = fold['train']
    exp.run_instance(model=model, x=xf, y=yf, fold=fold, index=None, log=None, show=False)
