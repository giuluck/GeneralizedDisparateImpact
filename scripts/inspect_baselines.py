import time

import seaborn as sns

from src.experiments import get

sns.set_context('notebook')
sns.set_style('whitegrid')

if __name__ == '__main__':
    exp = get('communities continuous')
    fold = exp.get_folds(folds=1)
    model = exp.get_model(model='sbr hgr', verbose=True)
    print('MODEL CONFIGURATION:')
    for k, v in model.config.items():
        print(f'  > {k} --> {v}')
    print('-------------------------------------------------')
    x, y = fold['train']
    start = time.time()
    exp.run_instance(model=model, x=x, y=y, fold=fold, index=None, log=None, show=True)
    print('-------------------------------------------------')
    print(f'Elapsed Time: {time.time() - start}')
