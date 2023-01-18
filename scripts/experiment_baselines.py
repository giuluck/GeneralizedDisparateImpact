import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from src.experiments import get

experiments = {
    'communities categorical': ['rf', 'gb', 'nn', 'sbr first', 'mt first rf', 'mt first gb', 'mt first nn'],
    'communities continuous': ['rf', 'gb', 'nn', 'sbr first', 'mt first rf', 'mt first gb', 'mt first nn'],
    'adult categorical': ['rf', 'gb', 'nn', 'sbr first', 'sbr didi',
                          'mt first rf', 'mt first gb', 'mt first nn',
                          'mt didi rf', 'mt didi gb', 'mt didi nn'],
    'adult continuous': ['rf', 'gb', 'nn', 'sbr first', 'sbr didi',
                         'mt first rf', 'mt first gb', 'mt first nn',
                         'mt didi rf', 'mt didi gb', 'mt didi nn'],
}

if __name__ == '__main__':
    print('-------------------------------------------------')
    for dataset, models in experiments.items():
        print(f' * DATASET: {dataset}')
        for i, model in enumerate(models):
            if i != 0:
                print()
            print(f'    - MODEL: {model}')
            exp = get(dataset)
            folds = exp.get_folds(folds=5)
            for idx, fold in enumerate(folds):
                print(f'      > FOLD: {idx}', end='')
                start = time.time()
                x, y = fold['train']
                mdl = exp.get_model(model=model)
                exp.run_instance(model=mdl, x=x, y=y, fold=fold, index=idx, log='experiments', show=False)
                print(f' -- elapsed time = {time.time() - start:.2f}s')
        print('-------------------------------------------------')
    shutil.rmtree('wandb')
