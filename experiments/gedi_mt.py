"""This script runs the experiments on moving targets approaches only and stores the results on Weights & Biases."""

import os

from experiments.config import WandBConfig
from src.models import MovingTargets

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from moving_targets.callbacks import WandBLogger
from src.experiments import get

models = ['rf', 'gb', 'nn']

datasets = {
    'communities categorical': [1],
    'communities continuous': [1, 2, 3, 4, 5],
    'adult categorical': [1],
    'adult continuous': [1, 2, 3, 4, 5]
}

if __name__ == '__main__':
    print('-------------------------------------------------')
    for model in models:
        for dataset, degrees in datasets.items():
            print(f' * MODEL: {model}, DATASET: {dataset}')
            for i, degree in enumerate(degrees):
                if i != 0:
                    print()
                print(f'    - DEGREE: {degree}')
                exp = get(dataset)
                folds = exp.get_folds(folds=5)
                for idx, fold in enumerate(folds):
                    print(f'      > FOLD: {idx}', end='')
                    start = time.time()
                    x, y = fold['train']
                    # build the model without callback and add it later in order to use the model config
                    mdl = exp.get_model(model=f'mt {model} fine', fold=fold, degree=degree)
                    assert isinstance(mdl, MovingTargets), f"There has been some errors with retrieved model {mdl}"
                    mdl.add_callback(WandBLogger(
                        project=WandBConfig.gedi_mt,
                        entity=WandBConfig.entity,
                        run_name=f'{model} - {dataset} - {degree} ({idx})',
                        dataset=dataset,
                        model=model,
                        fold=idx,
                        **mdl.config
                    ))
                    exp.run_instance(model=mdl, x=x, y=y, fold=fold, index=None, log=None, show=False)
                    print(f' -- elapsed time = {time.time() - start:.2f}s')
            print('-------------------------------------------------')
    shutil.rmtree('wandb')
