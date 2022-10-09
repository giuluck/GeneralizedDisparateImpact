import os
import shutil
import time

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.experiments import get

datasets = {
    'communities race': ['mt', 'sbr'],
    'adult race': ['mt', 'sbr'],
}

if __name__ == '__main__':
    print('-------------------------------------------------')
    for dataset, models in datasets.items():
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
                exp.run_instance(model=mdl, x=x, y=y, fold=fold, index=idx, log=True, show=False)
                print(f' -- elapsed time = {time.time() - start:.2f}s')
        print('-------------------------------------------------')
    shutil.rmtree('wandb')
